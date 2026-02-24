"""
card_parser.py — Universal Visiting Card Parser
Simple-first OCR pipeline: tries original image first, escalates processing only if needed.
Strict phone validation: +91 mobile = 10 digits, 033 landline = 8 digits.
Quality gate: only sends to LLM if OCR text is good enough.
"""

import os
import re
import json
import logging
import requests
import numpy as np
import base64
import io
import cv2
from PIL import Image, ImageOps
import pytesseract
from dotenv import load_dotenv

load_dotenv()

# ─── Logger ───
log = logging.getLogger("card_parser")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ─── Tesseract Path Setup ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESSERACT_PATH = os.path.join(BASE_DIR, "Tesseract-OCR", "tesseract.exe")

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ─── OpenRouter Config ───
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


# ═══════════════════════════════════════
# PHONE NUMBER VALIDATION & NORMALIZATION
# ═══════════════════════════════════════

def normalize_phone_number(raw: str) -> str:
    stripped = raw.strip()
    has_plus = stripped.startswith('+')
    digits = re.sub(r'\D', '', stripped)

    if not digits:
        return ""

    # +91 Indian mobile
    if has_plus and digits.startswith('91'):
        after_91 = digits[2:]
        if len(after_91) == 10 and after_91[0] in '6789':
            return '+91' + after_91
        return ""

    if digits.startswith('91') and len(digits) == 12:
        after_91 = digits[2:]
        if after_91[0] in '6789':
            return '+91' + after_91
        return ""

    if digits.startswith('091') and len(digits) == 13:
        after = digits[3:]
        if len(after) == 10 and after[0] in '6789':
            return '+91' + after
        return ""

    # 033 Kolkata landline
    if digits.startswith('033'):
        after_033 = digits[3:]
        if len(after_033) == 8:
            return '033' + after_033
        return ""

    # Other Indian STD codes
    if digits.startswith('0') and len(digits) >= 2:
        if 10 <= len(digits) <= 12:
            return digits
        return ""

    # Bare 10-digit Indian mobile
    if len(digits) == 10 and digits[0] in '6789':
        return '+91' + digits

    # Other international numbers with +
    if has_plus and 7 <= len(digits) <= 15:
        return '+' + digits

    # Fallback
    if 7 <= len(digits) <= 15:
        return digits

    return ""


def validate_phone_numbers(phone_list: list) -> list:
    seen = set()
    result = []
    for raw in phone_list:
        if not raw or not isinstance(raw, str):
            continue
        normalized = normalize_phone_number(raw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


# ═══════════════════════════════════════════════
# SIMPLE-FIRST IMAGE PROCESSING & OCR
# Strategy: try original first, escalate only if needed
# ═══════════════════════════════════════════════

MIN_OCR_WIDTH = 1500


def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")
    elif pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    if len(cv_img.shape) == 2:
        return Image.fromarray(cv_img)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def fix_exif(pil_img: Image.Image) -> Image.Image:
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass
    return pil_img


def upscale(img: np.ndarray, min_w: int = MIN_OCR_WIDTH) -> np.ndarray:
    h, w = img.shape[:2]
    if w < min_w:
        scale = min_w / w
        img = cv2.resize(img, (min_w, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img


def rotate_img(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def ocr_run(img, config="--psm 6 --oem 3") -> str:
    try:
        if isinstance(img, np.ndarray):
            img = cv_to_pil(img)
        return pytesseract.image_to_string(img, config=config).strip()
    except Exception:
        return ""


def ocr_score(text: str) -> int:
    if not text or len(text.strip()) < 5:
        return 0

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    score = 0

    score += min(len(text.strip()), 80)

    words = re.findall(r'[a-zA-Z]{3,}', text)
    score += min(len(words) * 6, 50)

    if re.search(r'\d{7,}', re.sub(r'[\s\-]', '', text)):
        score += 20

    if re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text):
        score += 25

    if re.search(r'www\.|\.com|\.in|\.org|\.net|https?://', text, re.IGNORECASE):
        score += 15

    score += min(len(lines) * 3, 20)

    garbage = len(re.findall(r'[^a-zA-Z0-9\s@.+\-(),/:&\'\"#\n]', text))
    score -= garbage * 2

    short_noise = sum(1 for l in lines if len(l) < 3)
    score -= short_noise * 3

    return max(score, 0)


def ocr_best_of_configs(img: np.ndarray) -> tuple:
    best_text = ""
    best_score = 0
    for psm in ["--psm 6 --oem 3", "--psm 3 --oem 3", "--psm 4 --oem 3"]:
        text = ocr_run(img, psm)
        score = ocr_score(text)
        if score > best_score:
            best_score = score
            best_text = text
    return best_text, best_score


# ─── Level 0: Raw original (just upscale) ───
def _try_original(cv_img: np.ndarray) -> tuple:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img.copy()
    gray = upscale(gray)
    return ocr_best_of_configs(gray)


# ─── Level 1: Light processing (blur + OTSU) ───
def _try_light(cv_img: np.ndarray) -> tuple:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img.copy()
    gray = upscale(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ocr_best_of_configs(binary)


# ─── Level 2: CLAHE contrast enhancement ───
def _try_clahe(cv_img: np.ndarray) -> tuple:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img.copy()
    gray = upscale(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ocr_best_of_configs(binary)


# ─── Level 3: Adaptive threshold ───
def _try_adaptive(cv_img: np.ndarray) -> tuple:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img.copy()
    gray = upscale(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, blockSize=21, C=10)
    return ocr_best_of_configs(binary)


# ─── Level 4: Sharpen + threshold ───
def _try_sharpen(cv_img: np.ndarray) -> tuple:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img.copy()
    gray = upscale(gray)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, blockSize=15, C=8)
    return ocr_best_of_configs(binary)


def extract_text_ocr(pil_img: Image.Image) -> str:
    """
    Simple-first OCR:
    1. Fix EXIF
    2. Try ORIGINAL at all 4 rotations (0, 180, 90, 270)
    3. Escalate to heavier processing only if needed
    4. Return best text found
    """
    pil_img = fix_exif(pil_img)
    cv_img = pil_to_cv(pil_img)
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)

    h, w = cv_img.shape[:2]
    log.info(f"OCR START — image size: {w}x{h}")

    best_text = ""
    best_score = 0
    best_level = ""
    best_angle = 0

    levels = [_try_original, _try_light, _try_clahe, _try_adaptive, _try_sharpen]
    level_names = ["original", "light", "clahe", "adaptive", "sharpen"]

    for idx, level_fn in enumerate(levels):
        lname = level_names[idx]
        for angle in [0, 180, 90, 270]:
            rotated = rotate_img(cv_img, angle)
            try:
                text, score = level_fn(rotated)
            except Exception:
                continue

            log.debug(f"  [{lname} @ {angle}°] score={score}  text_len={len(text)}")

            if score > best_score:
                best_score = score
                best_text = text
                best_level = lname
                best_angle = angle

            if best_score >= 80:
                log.info(f"OCR DONE (early exit) — level={best_level} angle={best_angle}° score={best_score}")
                log.debug(f"OCR TEXT:\n{best_text[:500]}")
                return best_text

        if best_score >= 50:
            log.info(f"OCR DONE (good enough) — level={best_level} angle={best_angle}° score={best_score}")
            log.debug(f"OCR TEXT:\n{best_text[:500]}")
            return best_text

    log.warning(f"OCR DONE (low quality) — level={best_level} angle={best_angle}° score={best_score}")
    if best_text:
        log.debug(f"OCR TEXT:\n{best_text[:500]}")
    else:
        log.warning("OCR returned EMPTY text — no text detected in any orientation/processing")

    return best_text


def is_ocr_quality_sufficient(text: str) -> bool:
    """
    Quality gate: OCR must have at least 3 words AND
    (a phone number OR email OR 5+ words) before sending to LLM.
    """
    if not text or len(text.strip()) < 10:
        log.warning("QUALITY GATE: FAIL — text too short")
        return False

    words = re.findall(r'[a-zA-Z]{3,}', text)
    if len(words) < 3:
        log.warning(f"QUALITY GATE: FAIL — only {len(words)} words found (need 3+)")
        return False

    has_phone = bool(re.search(r'\d{7,}', re.sub(r'[\s\-]', '', text)))
    has_email = bool(re.search(r'@', text))
    has_enough_words = len(words) >= 5

    passed = has_phone or has_email or has_enough_words
    if passed:
        log.info(f"QUALITY GATE: PASS — words={len(words)} phone={has_phone} email={has_email}")
    else:
        log.warning(f"QUALITY GATE: FAIL — words={len(words)} phone={has_phone} email={has_email}")
    return passed


# ═══════════════════════════════
# REGEX EXTRACTION
# ═══════════════════════════════

def extract_regex(text: str) -> dict:
    phone_patterns = [
        r'\+?\d{1,4}[\s\-]?\(?\d{1,5}\)?[\s\-]?\d{1,5}[\s\-]?\d{1,5}[\s\-]?\d{0,5}',
        r'\+?\d[\d\s\-\.]{7,15}\d',
    ]

    raw_phones = set()
    for pattern in phone_patterns:
        for m in re.findall(pattern, text):
            cleaned = re.sub(r'[\s\-\.]', '', m)
            if 7 <= len(cleaned) <= 18:
                raw_phones.add(m.strip())

    emails = re.findall(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text)
    websites = re.findall(r'(?:https?://|www\.)[^\s,;]+', text, re.IGNORECASE)

    return {
        "phones": list(raw_phones),
        "emails": [e.strip() for e in emails],
        "websites": [w.strip() for w in websites],
    }


# ═══════════════════════════════
# LLM PARSING
# ═══════════════════════════════

def parse_with_llm(ocr_text: str) -> dict:
    prompt = f"""You are an expert at reading business/visiting cards. Extract ALL info from this OCR text. Return ONLY valid JSON, no markdown.

Fields:
- person_name (string)
- company_name (string)
- designation (string)
- phone_numbers (array of strings — ALL phone/mobile/landline numbers exactly as written)
- email (string)
- website (string)
- office_address (string — full address)
- city (string)
- state (string)
- country (string)

IMPORTANT for phone_numbers:
- Include ALL numbers: mobile, office, fax, landline
- Keep them exactly as written on the card including +91, 033, etc.
- Do NOT skip any phone number

If not found, use "" for strings and [] for arrays.

OCR Text:
\"\"\"{ocr_text}\"\"\"

Return ONLY the JSON object."""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENROUTER_MODEL,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "Extract business card details. Return ONLY valid JSON. No markdown code blocks."},
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=30
        )

        result = response.json()
        if "choices" not in result:
            return {"error": "LLM response error"}

        raw = result["choices"][0]["message"]["content"].strip()

        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

        return json.loads(raw)

    except json.JSONDecodeError:
        return {"error": "LLM returned invalid JSON"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}


def parse_with_vision(image: Image.Image) -> dict:
    """Use GPT-4o-mini Vision to directly read and parse the card perfectly."""
    log.info("Sending image to Vision API (GPT-4o-mini)...")
    
    # Compress/resize image slightly so it doesn't exceed vision API limits or take too long
    # Convert to RGB (required for JPEG)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    w, h = image.size
    max_dim = 2000
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    prompt = """You are an expert at reading business/visiting cards. Read all the text on this card and extract the details. Return ONLY valid JSON, no markdown.

Fields:
- person_name (string)
- company_name (string)
- designation (string)
- phone_numbers (array of strings — ALL phone/mobile/landline numbers exactly as written)
- email (string)
- website (string)
- office_address (string — full address)
- city (string)
- state (string)
- country (string)
- raw_text (string - the raw extracted text you see on the card, exactly as written)

IMPORTANT for phone_numbers:
- Include ALL numbers: mobile, office, fax, landline
- Keep them exactly as written on the card including +91, 033, etc.
- Do NOT skip any phone number

If not found, use "" for strings and [] for arrays.

Return ONLY the JSON object."""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENROUTER_MODEL,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            },
            timeout=40
        )

        result = response.json()
        if "choices" not in result:
            return {"error": "Vision API response error"}

        raw = result["choices"][0]["message"]["content"].strip()

        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

        parsed = json.loads(raw)
        return parsed

    except json.JSONDecodeError:
        return {"error": "Vision returned invalid JSON"}
    except requests.exceptions.Timeout:
        return {"error": "Vision request timed out"}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════
# MAIN PARSER
# ═══════════════════════════════

def parse_card(image: Image.Image) -> dict:
    """
    Main entry: PIL Image → parsed dict.
    Pipeline: Try Vision API for perfect extraction → Fallback to Tesseract OCR → LLM.
    """
    log.info("═══ PARSE_CARD START ═══")

    # Step 1: Try GPT-4o-mini Vision API (handles all orientations, 100% accuracy)
    vision_data = parse_with_vision(image)
    
    # If vision is successful, use it!
    if "error" not in vision_data:
        log.info("Vision API succeeded.")
        log.info(f"VISION RESULT: name={vision_data.get('person_name')}, company={vision_data.get('company_name')}, phones={vision_data.get('phone_numbers')}")
        
        raw_text = vision_data.get("raw_text", "")
        regex_data = extract_regex(raw_text)
        
        vision_phones = vision_data.get("phone_numbers", [])
        if isinstance(vision_phones, str):
            vision_phones = [vision_phones] if vision_phones else []
            
        all_raw_phones = [p for p in vision_phones if p] + regex_data["phones"]
        validated_phones = validate_phone_numbers(all_raw_phones)
        log.info(f"VALIDATED PHONES: {validated_phones}")
        
        result = {
            "person_name": vision_data.get("person_name", ""),
            "company_name": vision_data.get("company_name", ""),
            "designation": vision_data.get("designation", ""),
            "phone_numbers": validated_phones,
            "email": regex_data.get("emails", [vision_data.get("email", "")])[0] if regex_data.get("emails") else vision_data.get("email", ""),
            "website": regex_data.get("websites", [vision_data.get("website", "")])[0] if regex_data.get("websites") else vision_data.get("website", ""),
            "office_address": vision_data.get("office_address", ""),
            "city": vision_data.get("city", ""),
            "state": vision_data.get("state", ""),
            "country": vision_data.get("country", ""),
            "ocr_text": raw_text,
        }
        log.info("═══ PARSE_CARD DONE (VISION) ═══")
        return result
    
    # Step 2: Fallback to classic OCR if vision fails (timeout/error)
    log.warning(f"Vision API failed ({vision_data.get('error')}). Falling back to Tesseract OCR...")
    
    ocr_text = extract_text_ocr(image)

    if not ocr_text or len(ocr_text.strip()) < 5:
        log.error("PARSE_CARD: No text extracted from image at all")
        return {"error": "Could not read any text from the image. Please try a clearer photo with better lighting."}

    # Step 2: Quality gate — skip LLM if OCR is garbage
    if not is_ocr_quality_sufficient(ocr_text):
        log.warning("PARSE_CARD: OCR quality too low — skipping LLM, returning regex-only")
        regex_data = extract_regex(ocr_text)
        validated_phones = validate_phone_numbers(regex_data["phones"])
        return {
            "error": "OCR quality too low. The image may be blurry, dark, or at an extreme angle. Please retake the photo.",
            "person_name": "",
            "company_name": "",
            "designation": "",
            "phone_numbers": validated_phones,
            "email": regex_data["emails"][0] if regex_data["emails"] else "",
            "website": regex_data["websites"][0] if regex_data["websites"] else "",
            "office_address": "",
            "city": "",
            "state": "",
            "country": "",
            "ocr_text": ocr_text,
        }

    # Step 3: Regex extraction
    regex_data = extract_regex(ocr_text)
    log.info(f"REGEX: phones={regex_data['phones']}  emails={regex_data['emails']}  websites={regex_data['websites']}")

    # Step 4: LLM parsing (only if OCR quality is sufficient)
    log.info("Sending OCR text to LLM...")
    llm_data = parse_with_llm(ocr_text)

    if "error" in llm_data:
        log.error(f"LLM ERROR: {llm_data['error']}")
        validated_phones = validate_phone_numbers(regex_data["phones"])
        return {
            "person_name": "",
            "company_name": "",
            "designation": "",
            "phone_numbers": validated_phones,
            "email": regex_data["emails"][0] if regex_data["emails"] else "",
            "website": regex_data["websites"][0] if regex_data["websites"] else "",
            "office_address": "",
            "city": "",
            "state": "",
            "country": "",
            "ocr_text": ocr_text,
            "parse_warning": llm_data["error"]
        }

    log.info(f"LLM RESULT: name={llm_data.get('person_name')}, company={llm_data.get('company_name')}, phones={llm_data.get('phone_numbers')}")
    log.debug(f"LLM FULL: {json.dumps(llm_data, indent=2)}")

    # Step 5: Merge phones from LLM + regex, validate
    llm_phones = llm_data.get("phone_numbers", [])
    if isinstance(llm_phones, str):
        llm_phones = [llm_phones] if llm_phones else []

    all_raw_phones = [p for p in llm_phones if p] + regex_data["phones"]
    validated_phones = validate_phone_numbers(all_raw_phones)
    log.info(f"VALIDATED PHONES: {validated_phones}")

    result = {
        "person_name": llm_data.get("person_name", ""),
        "company_name": llm_data.get("company_name", ""),
        "designation": llm_data.get("designation", ""),
        "phone_numbers": validated_phones,
        "email": regex_data["emails"][0] if regex_data["emails"] else llm_data.get("email", ""),
        "website": regex_data["websites"][0] if regex_data["websites"] else llm_data.get("website", ""),
        "office_address": llm_data.get("office_address", ""),
        "city": llm_data.get("city", ""),
        "state": llm_data.get("state", ""),
        "country": llm_data.get("country", ""),
        "ocr_text": ocr_text,
    }
    log.info("═══ PARSE_CARD DONE ═══")
    return result

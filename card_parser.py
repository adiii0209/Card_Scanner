"""
card_parser.py — Universal Visiting Card Parser
Handles OCR, image preprocessing, auto-rotation, regex extraction, LLM parsing.
Strict phone number validation: +91 mobile = 10 digits, 033 landline = 8 digits.
"""

import os
import re
import json
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
from dotenv import load_dotenv

load_dotenv()

# ─── Tesseract Path Setup ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESSERACT_PATH = os.path.join(BASE_DIR, "Tesseract-OCR", "tesseract.exe")

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ─── OpenRouter Config ───
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "openai/gpt-4o-mini"


# ═══════════════════════════════════════
# PHONE NUMBER VALIDATION & NORMALIZATION
# ═══════════════════════════════════════

def normalize_phone_number(raw: str) -> str:
    """
    Normalize and validate a single phone number string.
    Rules:
      - Indian mobile (+91): must have exactly 10 digits after +91, starting with 6-9
        → returns "+91XXXXXXXXXX"
      - Kolkata landline (033): must have exactly 8 digits after 033
        → returns "033XXXXXXXX"
      - Other STD codes (0XX): must have 8-10 digits after the code
        → returns as cleaned digits
      - Any number that is exactly 10 digits starting with 6-9 is assumed Indian mobile
        → returns "+91XXXXXXXXXX"
      - Otherwise returns the cleaned number if it has 7-15 digits
    Returns empty string "" if invalid.
    """
    # Strip all non-digit characters except leading +
    stripped = raw.strip()
    has_plus = stripped.startswith('+')
    digits = re.sub(r'\D', '', stripped)

    if not digits:
        return ""

    # ─── +91 Indian mobile ───
    # Could be: +91XXXXXXXXXX, 91XXXXXXXXXX, 091XXXXXXXXXX
    if has_plus and digits.startswith('91'):
        after_91 = digits[2:]
        if len(after_91) == 10 and after_91[0] in '6789':
            return '+91' + after_91
        return ""  # Invalid +91 number

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

    # ─── 033 Kolkata landline ───
    if digits.startswith('033'):
        after_033 = digits[3:]
        if len(after_033) == 8:
            return '033' + after_033
        return ""

    # ─── Other Indian STD codes (0XX or 0XXX) ───
    if digits.startswith('0') and len(digits) >= 2:
        # Common: 011 (Delhi), 022 (Mumbai), 044 (Chennai), etc.
        # STD code is 2-4 digits including the 0
        # Total should be 10-11 digits for Indian landlines
        if 10 <= len(digits) <= 12:
            return digits
        return ""

    # ─── Bare 10-digit Indian mobile ───
    if len(digits) == 10 and digits[0] in '6789':
        return '+91' + digits

    # ─── Other international numbers with + ───
    if has_plus and 7 <= len(digits) <= 15:
        return '+' + digits

    # ─── Fallback: accept 7-15 digit numbers ───
    if 7 <= len(digits) <= 15:
        return digits

    return ""


def validate_phone_numbers(phone_list: list) -> list:
    """
    Take a list of raw phone strings, normalize each, remove duplicates and empties.
    Returns list of validated, formatted phone numbers.
    """
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


# ═══════════════════════════════
# IMAGE PROCESSING & OCR
# ═══════════════════════════════

def fix_orientation(image: Image.Image) -> Image.Image:
    """Fix image orientation using EXIF data (phone camera rotations)."""
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    return image


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess for better OCR: grayscale, contrast, sharpen, resize."""
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    gray = image.convert("L")

    w, h = gray.size
    if w < 1000:
        scale = 1000 / w
        gray = gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = ImageEnhance.Sharpness(gray).enhance(2.0)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = gray.point(lambda x: 0 if x < 130 else 255, "1")
    return gray


def ocr_single(image: Image.Image, config: str = "--psm 6 --oem 3") -> str:
    """Run OCR on a single image with given config."""
    try:
        return pytesseract.image_to_string(image, config=config).strip()
    except Exception:
        return ""


def extract_text_ocr(image: Image.Image) -> str:
    """
    Run OCR trying multiple orientations (0°, 90°, 180°, 270°) and PSM modes.
    Returns the best quality text. Handles upside-down/rotated cards.
    """
    image = fix_orientation(image)

    rotations = [0, 180, 90, 270]
    configs = [
        "--psm 6 --oem 3",
        "--psm 3 --oem 3",
        "--psm 4 --oem 3",
    ]

    best_text = ""
    best_score = 0

    for angle in rotations:
        rotated = image.rotate(angle, expand=True) if angle != 0 else image

        preprocessed = preprocess_image(rotated)
        for config in configs:
            text = ocr_single(preprocessed, config)
            score = _text_quality_score(text)
            if score > best_score:
                best_score = score
                best_text = text

        # Also try on original RGB
        try:
            original_rgb = rotated.convert("RGB") if rotated.mode != "RGB" else rotated
            text = ocr_single(original_rgb, "--psm 6 --oem 3")
            score = _text_quality_score(text)
            if score > best_score:
                best_score = score
                best_text = text
        except Exception:
            pass

        if best_score > 50:
            break

    return best_text


def _text_quality_score(text: str) -> int:
    """Score OCR quality. Higher = better orientation."""
    if not text:
        return 0

    score = min(len(text.strip()), 100)

    if re.search(r'[a-zA-Z]{3,}', text):
        score += 20
    if re.search(r'\d{5,}', text):
        score += 10
    if re.search(r'@', text):
        score += 15
    if re.search(r'www\.|\\.com|\\.in|\\.org', text, re.IGNORECASE):
        score += 10

    garbage = len(re.findall(r'[^a-zA-Z0-9\s@.+\-(),/:&]', text))
    score -= garbage

    return score


# ═══════════════════════════════
# REGEX EXTRACTION
# ═══════════════════════════════

def extract_regex(text: str) -> dict:
    """Extract phone numbers, emails, websites using regex."""
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
    """Use LLM to parse card text into structured fields."""
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


# ═══════════════════════════════
# MAIN PARSER
# ═══════════════════════════════

def parse_card(image: Image.Image) -> dict:
    """
    Main entry: PIL Image → parsed dict with validated phone numbers.
    """
    # Step 1: OCR with auto-rotation
    ocr_text = extract_text_ocr(image)

    if not ocr_text:
        return {"error": "Could not read any text. Please try a clearer image."}

    # Step 2: Regex extraction
    regex_data = extract_regex(ocr_text)

    # Step 3: LLM parsing
    llm_data = parse_with_llm(ocr_text)

    if "error" in llm_data:
        # Fallback to regex
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

    # Step 4: Merge phone numbers from LLM + regex
    llm_phones = llm_data.get("phone_numbers", [])
    if isinstance(llm_phones, str):
        llm_phones = [llm_phones] if llm_phones else []

    # Combine all raw phones, then validate/normalize
    all_raw_phones = [p for p in llm_phones if p] + regex_data["phones"]
    validated_phones = validate_phone_numbers(all_raw_phones)

    return {
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

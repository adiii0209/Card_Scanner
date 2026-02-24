"""
app.py — FastAPI Application for Business Card Scanner
Routes: Home, Scan, Save, Dashboard, Card Details, Delete, Search
"""

import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from card_parser import parse_card
from database import (
    save_card, get_all_cards, get_card_by_id,
    get_card_image, update_card, delete_card,
    search_cards, get_stats
)

# ─── FastAPI Init ───
app = FastAPI(title="CardVault — Business Card Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ─── HOME — Scan Page ───
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ─── SCAN — Process Card Image ───
@app.post("/scan")
async def scan_card_route(file: UploadFile = File(...)):
    try:
        # Read file bytes
        file_bytes = await file.read()

        # Open image with PIL
        image = Image.open(io.BytesIO(file_bytes))

        # Parse the card
        result = parse_card(image)

        if "error" in result and not result.get("person_name"):
            return JSONResponse(content={"error": result["error"]}, status_code=400)

        # Encode image as base64 for preview
        img_b64 = base64.b64encode(file_bytes).decode("utf-8")

        # Determine mime type
        ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else "png"
        mime_map = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "bmp": "image/bmp",
            "webp": "image/webp", "tiff": "image/tiff",
            "gif": "image/gif"
        }
        mime = mime_map.get(ext, "image/png")

        result["image_data"] = f"data:{mime};base64,{img_b64}"
        result["image_filename"] = file.filename or "card.png"

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ─── SAVE — Save Card to Database ───
@app.post("/save")
async def save_card_route(request: Request):
    try:
        data = await request.json()

        # Extract image data if present
        image_bytes = None
        image_filename = data.pop("image_filename", "")
        image_data = data.pop("image_data", "")

        if image_data and "base64," in image_data:
            b64_str = image_data.split("base64,")[1]
            image_bytes = base64.b64decode(b64_str)

        # Remove non-db fields
        data.pop("ocr_text", None)
        data.pop("parse_warning", None)

        card_id = save_card(data, image_bytes, image_filename)

        return JSONResponse(content={"success": True, "card_id": card_id})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ─── DASHBOARD — View All Saved Cards ───
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    cards = get_all_cards()
    stats = get_stats()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "cards": cards,
        "stats": stats
    })


# ─── CARD DETAIL ───
@app.get("/card/{card_id}")
async def card_detail(card_id: int):
    card = get_card_by_id(card_id)
    if not card:
        return JSONResponse(content={"error": "Card not found"}, status_code=404)
    return JSONResponse(content=card)


# ─── CARD IMAGE ───
@app.get("/card/{card_id}/image")
async def card_image(card_id: int):
    img_bytes, filename = get_card_image(card_id)
    if not img_bytes:
        return JSONResponse(content={"error": "No image"}, status_code=404)

    ext = filename.rsplit(".", 1)[-1].lower() if filename else "png"
    mime_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "bmp": "image/bmp",
        "webp": "image/webp", "tiff": "image/tiff",
    }
    mime = mime_map.get(ext, "image/png")

    return Response(content=img_bytes, media_type=mime)


# ─── UPDATE CARD ───
@app.put("/card/{card_id}")
async def update_card_route(card_id: int, request: Request):
    try:
        data = await request.json()
        success = update_card(card_id, data)
        if success:
            return JSONResponse(content={"success": True})
        return JSONResponse(content={"error": "Card not found"}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ─── DELETE CARD ───
@app.delete("/card/{card_id}")
async def delete_card_route(card_id: int):
    success = delete_card(card_id)
    if success:
        return JSONResponse(content={"success": True})
    return JSONResponse(content={"error": "Card not found"}, status_code=404)


# ─── SEARCH ───
@app.get("/search")
async def search_cards_route(q: str = ""):
    if not q.strip():
        cards = get_all_cards()
    else:
        cards = search_cards(q)
    return JSONResponse(content={"cards": cards})


# ─── STATS ───
@app.get("/stats")
async def stats_route():
    return JSONResponse(content=get_stats())


# ─── Run ───
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
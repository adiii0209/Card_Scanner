"""
database.py — SQLite Database Layer for Card Scanner
Lightweight, zero-config database for storing scanned visiting cards.
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards.db")


def get_connection():
    """Get a database connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT DEFAULT '',
            company_name TEXT DEFAULT '',
            designation TEXT DEFAULT '',
            department TEXT DEFAULT '',
            phone_numbers TEXT DEFAULT '[]',
            email TEXT DEFAULT '',
            secondary_email TEXT DEFAULT '',
            website TEXT DEFAULT '',
            office_address TEXT DEFAULT '',
            city TEXT DEFAULT '',
            state TEXT DEFAULT '',
            country TEXT DEFAULT '',
            pincode TEXT DEFAULT '',
            social_media TEXT DEFAULT '',
            fax TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            ocr_text TEXT DEFAULT '',
            card_image BLOB DEFAULT NULL,
            image_filename TEXT DEFAULT '',
            created_at TEXT DEFAULT '',
            updated_at TEXT DEFAULT ''
        )
    """)

    conn.commit()
    conn.close()


def save_card(card_data: dict, image_bytes: bytes = None, image_filename: str = "") -> int:
    """
    Save a scanned card to the database.
    Returns the ID of the saved card.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    phone_numbers = card_data.get("phone_numbers", [])
    if isinstance(phone_numbers, list):
        phone_numbers = json.dumps(phone_numbers)
    elif isinstance(phone_numbers, str):
        phone_numbers = json.dumps([phone_numbers]) if phone_numbers else "[]"

    social_media = card_data.get("social_media", "")
    if isinstance(social_media, (dict, list)):
        social_media = json.dumps(social_media)

    cursor.execute("""
        INSERT INTO cards (
            person_name, company_name, designation, department,
            phone_numbers, email, secondary_email, website,
            office_address, city, state, country, pincode,
            social_media, fax, notes, ocr_text,
            card_image, image_filename, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        card_data.get("person_name", ""),
        card_data.get("company_name", ""),
        card_data.get("designation", ""),
        card_data.get("department", ""),
        phone_numbers,
        card_data.get("email", ""),
        card_data.get("secondary_email", ""),
        card_data.get("website", ""),
        card_data.get("office_address", ""),
        card_data.get("city", ""),
        card_data.get("state", ""),
        card_data.get("country", ""),
        card_data.get("pincode", ""),
        social_media,
        card_data.get("fax", ""),
        card_data.get("notes", ""),
        card_data.get("ocr_text", ""),
        image_bytes,
        image_filename,
        now,
        now
    ))

    card_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return card_id


def get_all_cards() -> list:
    """Get all saved cards, ordered by most recent first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, person_name, company_name, designation, department,
               phone_numbers, email, secondary_email, website,
               office_address, city, state, country, pincode,
               social_media, fax, notes, ocr_text,
               image_filename, created_at, updated_at
        FROM cards ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    cards = []
    for row in rows:
        card = dict(row)
        # Parse JSON fields
        try:
            card["phone_numbers"] = json.loads(card.get("phone_numbers", "[]"))
        except (json.JSONDecodeError, TypeError):
            card["phone_numbers"] = []
        try:
            sm = card.get("social_media", "")
            if sm and sm.startswith(("{", "[")):
                card["social_media"] = json.loads(sm)
        except (json.JSONDecodeError, TypeError):
            pass
        cards.append(card)

    return cards


def get_card_by_id(card_id: int) -> dict:
    """Get a single card by its ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, person_name, company_name, designation, department,
               phone_numbers, email, secondary_email, website,
               office_address, city, state, country, pincode,
               social_media, fax, notes, ocr_text,
               image_filename, created_at, updated_at
        FROM cards WHERE id = ?
    """, (card_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    card = dict(row)
    try:
        card["phone_numbers"] = json.loads(card.get("phone_numbers", "[]"))
    except (json.JSONDecodeError, TypeError):
        card["phone_numbers"] = []
    try:
        sm = card.get("social_media", "")
        if sm and sm.startswith(("{", "[")):
            card["social_media"] = json.loads(sm)
    except (json.JSONDecodeError, TypeError):
        pass

    return card


def get_card_image(card_id: int) -> tuple:
    """Get the card image bytes and filename."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT card_image, image_filename FROM cards WHERE id = ?", (card_id,))
    row = cursor.fetchone()
    conn.close()

    if row and row["card_image"]:
        return row["card_image"], row["image_filename"]
    return None, None


def update_card(card_id: int, card_data: dict) -> bool:
    """Update an existing card."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    phone_numbers = card_data.get("phone_numbers", [])
    if isinstance(phone_numbers, list):
        phone_numbers = json.dumps(phone_numbers)

    social_media = card_data.get("social_media", "")
    if isinstance(social_media, (dict, list)):
        social_media = json.dumps(social_media)

    cursor.execute("""
        UPDATE cards SET
            person_name = ?, company_name = ?, designation = ?, department = ?,
            phone_numbers = ?, email = ?, secondary_email = ?, website = ?,
            office_address = ?, city = ?, state = ?, country = ?, pincode = ?,
            social_media = ?, fax = ?, notes = ?, updated_at = ?
        WHERE id = ?
    """, (
        card_data.get("person_name", ""),
        card_data.get("company_name", ""),
        card_data.get("designation", ""),
        card_data.get("department", ""),
        phone_numbers,
        card_data.get("email", ""),
        card_data.get("secondary_email", ""),
        card_data.get("website", ""),
        card_data.get("office_address", ""),
        card_data.get("city", ""),
        card_data.get("state", ""),
        card_data.get("country", ""),
        card_data.get("pincode", ""),
        social_media,
        card_data.get("fax", ""),
        card_data.get("notes", ""),
        now,
        card_id
    ))

    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def delete_card(card_id: int) -> bool:
    """Delete a card from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cards WHERE id = ?", (card_id,))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def search_cards(query: str) -> list:
    """Search cards by name, company, email, or phone."""
    conn = get_connection()
    cursor = conn.cursor()
    search = f"%{query}%"
    cursor.execute("""
        SELECT id, person_name, company_name, designation, department,
               phone_numbers, email, secondary_email, website,
               office_address, city, state, country, pincode,
               social_media, fax, notes, ocr_text,
               image_filename, created_at, updated_at
        FROM cards
        WHERE person_name LIKE ? OR company_name LIKE ?
              OR email LIKE ? OR phone_numbers LIKE ?
              OR designation LIKE ? OR office_address LIKE ?
        ORDER BY id DESC
    """, (search, search, search, search, search, search))
    rows = cursor.fetchall()
    conn.close()

    cards = []
    for row in rows:
        card = dict(row)
        try:
            card["phone_numbers"] = json.loads(card.get("phone_numbers", "[]"))
        except (json.JSONDecodeError, TypeError):
            card["phone_numbers"] = []
        cards.append(card)

    return cards


def get_stats() -> dict:
    """Get database statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM cards")
    total = cursor.fetchone()["total"]
    conn.close()
    return {"total_cards": total}


# Initialize the database on import
init_db()

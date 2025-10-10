import base64, json
from typing import List, Tuple, Optional
import pandas as pd

# Minimal Pydantic-like validation is kept inside the Streamlit app via Pandas schema
# We inject the `OpenAI` client from the app. The tool uses strict tool-calling.

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "record_invoice",
        "description": "Return a single normalized record extracted from an invoice image.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "order_number": {"type": ["integer", "null"]},
                "type_of_cost": {"type": ["string", "null"]},
                "date": {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$"},
                "amount": {"type": ["number", "null"]},
                "currency": {"type": ["string", "null"], "maxLength": 8},
                "vendor_name": {"type": ["string", "null"]},
                "confidence_note": {"type": ["string", "null"]},
            },
            "required": ["order_number", "type_of_cost", "date", "amount", "currency", "vendor_name", "confidence_note"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = (
    "You transcribe and normalize multilingual invoices precisely.\n"
    "- Read ALL visible text in any language.\n"
    "- Do NOT guess; use null and explain in confidence_note.\n"
    "- date: ISO YYYY-MM-DD; prefer invoice issue date.\n"
    "- amount: grand total with taxes; normalize thousands/decimal.\n"
    "- currency: ISO code if present; else symbol.\n"
    "- vendor_name: legal merchant name.\n"
)

COLS = [
    "Order number", "Type of cost", "Date", "Amount", "Currency", "Vendor name", "Path of image", "Confidence note"
]


def _b64(name: str, raw: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(raw).decode('utf-8')}"


def _normalize_currency(cur: Optional[str]) -> Optional[str]:
    if not cur:
        return None
    m = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₩": "KRW", "₫": "VND"}
    cur = cur.strip().upper()
    return m.get(cur, cur[:8])


def _normalize_amount(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(".", "").replace(",", ".")
    elif s.count(".") > 1:
        parts = s.split(".")
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(s)
    except Exception:
        return None


def extract_records_from_images(client, files: List[Tuple[str, bytes]], model: str = "gpt-4o"):
    rows, notes = [], []
    for name, raw in files:
        image_url = {"url": _b64(name, raw), "detail": "high"}
        # filename hint like: taxi_12.jpg → ("taxi", 12)
        type_hint, num_hint = None, None
        import re, os
        base = os.path.splitext(name)[0]
        m = re.match(r"([A-Za-z\- ]+)[_\- ](\d+)$", base)
        if m:
            type_hint = m.group(1).strip().lower()
            num_hint = int(m.group(2))

        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            parallel_tool_calls=False,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Extract the normalized fields. Filename hints — type_of_cost: {type_hint or 'null'}, order_number: {num_hint or 'null'}."},
                        {"type": "image_url", "image_url": image_url},
                    ],
                },
            ],
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "record_invoice"}},
        )

        tool_calls = resp.choices[0].message.tool_calls or []
        payload = {}
        if tool_calls:
            payload = json.loads(tool_calls[0].function.arguments)

        # fallbacks + normalization
        payload.setdefault("order_number", num_hint)
        payload.setdefault("type_of_cost", type_hint)
        payload["currency"] = _normalize_currency(payload.get("currency"))
        payload["amount"] = _normalize_amount(payload.get("amount"))

        row = [
            payload.get("order_number"),
            payload.get("type_of_cost"),
            payload.get("date"),
            payload.get("amount"),
            payload.get("currency"),
            payload.get("vendor_name"),
            name,
            payload.get("confidence_note"),
        ]
        rows.append(row)
        if payload.get("confidence_note"):
            notes.append(f"{name}: {payload['confidence_note']}")

    df = pd.DataFrame(rows, columns=COLS)
    return df, notes
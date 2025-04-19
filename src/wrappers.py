import os, time, hmac, hashlib, requests, json
from dotenv import load_dotenv

load_dotenv()
KEY, SECRET = os.getenv("ROOSTOO_KEY"), os.getenv("ROOSTOO_SECRET")

BASE = "https://mock-api.roostoo.com/v3"


def _sign(payload: str) -> str:
    return hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

class RoostooError(RuntimeError):
    """Raised when the exchange rejects the request or returns non‑200."""

def place_order(pair: str, side: str, otype: str,
                quantity: str, price: float | None = None) -> dict:
    """Send a signed order and return the JSON.  
       Raises **RoostooError** with detailed message on any failure."""
    body = {
        "pair": pair,
        "side": side,
        "type": otype,
        "quantity": quantity,
        "timestamp": int(time.time() * 1000),
    }
    if otype.upper() == "LIMIT":
        if price is None:
            raise ValueError("price required for LIMIT orders")
        body["price"] = price

    # Canonical key order for signature
    payload = "&".join(f"{k}={body[k]}" for k in sorted(body))
    hdr = {
        "Content-Type": "application/x-www-form-urlencoded",
        "RST-API-KEY": KEY,
        "MSG-SIGNATURE": _sign(payload),
    }

    r = requests.post(f"{BASE}/place_order", data=payload, headers=hdr, timeout=10)

    # Try to decode JSON; fall back to plain text
    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text.strip()}

    # HTTP‑level error (451, 4xx, 5xx…)
    if not r.ok:
        raise RoostooError(
            f"HTTP {r.status_code} {r.reason} — {data}",
        )

    # Exchange‑level reject (Success:false or ErrMsg not empty)
    if isinstance(data, dict) and (not data.get("Success", True) or data.get("ErrMsg")):
        raise RoostooError(f"Exchange error: {data.get('ErrMsg', data)}")

    return data

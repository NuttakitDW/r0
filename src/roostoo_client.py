import hashlib, hmac, time, os, requests
from dotenv import load_dotenv

load_dotenv()
KEY, SECRET = os.getenv("ROOSTOO_KEY"), os.getenv("ROOSTOO_SECRET")
BASE = "https://mock-api.roostoo.com/v3"

def _sign(payload: str) -> str:
    return hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

def place_order(pair: str, side: str, otype: str, quantity: str, price: float | None = None):
    """Submit an order; return parsed JSON."""
    body = {
        "pair": pair,
        "side": side,
        "type": otype,
        "quantity": quantity,
        "timestamp": str(int(time.time() * 1000)),
    }
    if otype.upper() == "LIMIT":
        if price is None:
            raise ValueError("price is mandatory for LIMIT orders")
        body["price"] = price

    # Build payload string for HMAC in the canonical order
    payload = "&".join(f"{k}={body[k]}" for k in sorted(body))
    headers = {
        "RCL_TopLevelCheck": _sign(payload),
        "RST-API-KEY": KEY,
    }
    r = requests.post(f"{BASE}/place_order", json=body, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

import os, time, hmac, hashlib, requests, urllib.parse
from dotenv import load_dotenv
from typing import Any, Dict, Optional

load_dotenv()
KEY, SECRET = os.getenv("ROOSTOO_KEY"), os.getenv("ROOSTOO_SECRET")

BASE = "https://mock-api.roostoo.com/v3"


def _sign(payload: str) -> str:
    return hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

class RoostooError(RuntimeError):
    """Raised when the exchange rejects the request or returns non‑200."""
    
def get_server_time() -> int:
    """GET /v3/serverTime – returns epoch‑ms."""
    url = f"{BASE}/serverTime"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return int(r.json()["ServerTime"])

def get_exchange_info() -> dict:
    """GET /v3/exchangeInfo – returns exchange information."""
    url = f"{BASE}/exchangeInfo"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return r.json()

def get_ticker(pair: str) -> dict:
    """
    GET /v3/ticker?pair=…&timestamp=…
    Returns last price, bid/ask, volume for `pair`.
    """
    ts   = int(time.time() * 1000)                       # current epoch‑ms
    pair_q = urllib.parse.quote_plus(pair)               # encode 'BTC/USD' → 'BTC%2FUSD'
    url  = f"{BASE}/ticker?pair={pair_q}&timestamp={ts}"

    r = requests.get(url, timeout=5)

    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text.strip()}

    if not r.ok:
        raise RoostooError(f"HTTP {r.status_code} {r.reason} — {data}")

    if isinstance(data, dict) and (not data.get("Success", True) or data.get("ErrMsg")):
        raise RoostooError(f"Exchange error: {data.get('ErrMsg', data)}")

    return data

def get_balance() -> dict:
    """GET /v3/balance?timestamp=… – returns account balance."""
    ts = int(time.time() * 1000)                # current epoch‑ms

    payload = f"timestamp={ts}"                 # 1. canonical string
    sig     = _sign(payload)                    # 2. HMAC

    url = f"{BASE}/balance?{payload}"           # 3. include timestamp in URL
    hdr = {
        "RST-API-KEY": KEY,
        "MSG-SIGNATURE": sig,
    }

    r = requests.get(url, headers=hdr, timeout=5)

    # --- robust error handling like other helpers ---
    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text.strip()}

    if not r.ok:
        raise RoostooError(f"HTTP {r.status_code} {r.reason} — {data}")
    if isinstance(data, dict) and data.get("ErrMsg"):
        raise RoostooError(f"Exchange error: {data['ErrMsg']}")

    return data

def get_pending_count() -> dict:
    """
    GET /v3/pending_count?timestamp=…
    Returns the count of pending orders.
    """
    timestamp = int(time.time() * 1000)         # Generate current epoch‑ms
    payload = f"timestamp={timestamp}"          # 1. canonical string
    sig     = _sign(payload)                    # 2. HMAC

    url = f"{BASE}/pending_count?{payload}"     # 3. include timestamp in URL
    hdr = {
        "RST-API-KEY": KEY,
        "MSG-SIGNATURE": sig,
    }

    r = requests.get(url, headers=hdr, timeout=5)

    # --- robust error handling like other helpers ---
    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text.strip()}

    if not r.ok:
        raise RoostooError(f"HTTP {r.status_code} {r.reason} — {data}")
    if isinstance(data, dict) and data.get("ErrMsg"):
        raise RoostooError(f"Exchange error: {data['ErrMsg']}")

    return data


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

    side  = side.upper()
    otype = otype.upper() 
    
    if otype.upper() == "LIMIT":
        if price is None:
            raise ValueError("price required for LIMIT orders")
        body["price"] = price
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    if otype not in ("MARKET", "LIMIT"):
        raise ValueError("type must be MARKET or LIMIT")
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

def query_order(
    *,
    order_id: Optional[str] = None,
    pair: Optional[str] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    pending_only: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Query a single order by `order_id` **or** filter orders by pair/offset/…

    Raises
    ------
    RoostooError
        – on HTTP/network issues
        – when the exchange returns {"Success": false} or a non‑empty ErrMsg
    """

    ts = int(time.time() * 1000)

    # ── 1. Build form/body dict ───────────────────────────────────────
    body: Dict[str, str] = {"timestamp": str(ts)}

    if order_id:
        body["order_id"] = order_id
    else:
        if pair:
            body["pair"] = pair
        if offset is not None:
            body["offset"] = str(offset)
        if limit is not None:
            body["limit"] = str(limit)
        if pending_only is not None:
            body["pending_only"] = "TRUE" if pending_only else "FALSE"

    # ► FIX #1 – proper URL‑encoding instead of manual join
    payload = urllib.parse.urlencode(body, safe="/")

    sig = _sign(payload)

    url = f"{BASE}/query_order"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "RST-API-KEY": KEY,
        "MSG-SIGNATURE": sig,
    }

    # ── 2. POST the request ───────────────────────────────────────────
    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=10)
        resp.raise_for_status()                     # ► FIX #2 – HTTP guard
    except requests.RequestException as e:
        raise RoostooError(f"Network/HTTP error: {e}") from e

    # ── 3. Parse JSON safely ──────────────────────────────────────────
    try:
        data: Dict[str, Any] = resp.json()
    except ValueError:                             # not JSON?
        raise RoostooError(f"Non‑JSON response: {resp.text.strip()}")

    # ── 4. Exchange‑level error handling ──────────────────────────────
    if not data.get("Success", True) or data.get("ErrMsg"):
        raise RoostooError(f"Exchange error: {data.get('ErrMsg', data)}")

    return data

# ── Cancel order (signed) ──────────────────────────────────────────
def cancel_order(
    *,
    order_id: str | None = None,
    pair: str | None = None,
) -> dict:

    if order_id and pair:
        raise ValueError("Provide only one of order_id OR pair, not both.")

    ts   = int(time.time() * 1000)
    body: dict[str, str] = {"timestamp": str(ts)}

    if order_id:
        body["order_id"] = str(order_id)
    elif pair:
        body["pair"] = pair

    payload = "&".join(f"{k}={body[k]}" for k in sorted(body))
    sig     = _sign(payload)

    url = f"{BASE}/cancel_order"
    hdr = {
        "Content-Type": "application/x-www-form-urlencoded",
        "RST-API-KEY": KEY,
        "MSG-SIGNATURE": sig,
    }

    r = requests.post(url, data=payload, headers=hdr, timeout=10)

    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text.strip()}

    if not r.ok:
        raise RoostooError(f"HTTP {r.status_code} {r.reason} — {data}")
    if isinstance(data, dict) and (not data.get("Success", True) or data.get("ErrMsg")):
        raise RoostooError(f"Exchange error: {data.get('ErrMsg', data)}")

    return data

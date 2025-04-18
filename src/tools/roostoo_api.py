# src/tools/roostoo_api.py
import os, time, hmac, hashlib, requests, pandas as pd
from langchain.tools import tool
from src.config import ROOSTOO_KEY, ROOSTOO_SECRET

BASE = "https://mock-api.roostoo.com/v2"

def _sign(payload: str) -> str:
    return hmac.new(ROOSTOO_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

# ── 1. GET KLINES ────────────────────────────────────────────────────────────
@tool(args_schema={"symbol": str, "interval": str, "limit": int})
def getKlines(symbol: str, interval: str = "1m", limit: int = 500):
    """Return recent candlesticks for `symbol` as a pandas DataFrame."""
    qs = f"symbol={symbol}&interval={interval}&limit={limit}&timestamp={int(time.time()*1000)}"
    headers = {"RST-API-KEY": ROOSTOO_KEY, "MSG-SIGNATURE": _sign(qs)}
    r = requests.get(f"{BASE}/market/klines?{qs}", headers=headers, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["ts", "o", "h", "l", "c", "v"]).astype(float)
    return df

# ── 2. PLACE ORDER ──────────────────────────────────────────────────────────
@tool(args_schema={
    "symbol": str, "side": str, "qty": float, "price": float
})
def placeOrder(symbol: str, side: str, qty: float, price: float):
    """Submit a mock trade to Roostoo and return the order response JSON."""
    body = {
        "symbol": symbol.upper(), "side": side.upper(),
        "qty": qty, "price": price,
        "timestamp": int(time.time()*1000)
    }
    payload = "&".join(f"{k}={v}" for k, v in body.items())
    headers = {"RST-API-KEY": ROOSTOO_KEY, "MSG-SIGNATURE": _sign(payload)}
    r = requests.post(f"{BASE}/order/new", json=body, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

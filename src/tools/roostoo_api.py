"""
Roostoo API wrappers + analyzeMarket tool (JSON‑safe).
"""
import hashlib, hmac, time
from typing import Dict, Any
import pandas as pd
import requests
from langchain.tools import tool
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands

from src.config import ROOSTOO_KEY, ROOSTOO_SECRET
from .ta_signals import _last_cross

BASE_URL = "https://mock-api.roostoo.com/v2"

def _sign(payload: str) -> str:
    return hmac.new(ROOSTOO_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

def _headers(payload: str) -> Dict[str, str]:
    return {"RST-API-KEY": ROOSTOO_KEY, "MSG-SIGNATURE": _sign(payload)}

# -----------------------------------------------------------------
@tool
def getKlines(symbol: str, interval: str = "1m", limit: int = 500) -> list:
    """Return recent klines (OHLCV rows) as JSON list."""
    qs = f"symbol={symbol.upper()}&interval={interval}&limit={limit}&timestamp={int(time.time()*1000)}"
    r = requests.get(f"{BASE_URL}/market/klines?{qs}", headers=_headers(qs), timeout=10)
    r.raise_for_status()
    return r.json()

@tool
def placeOrder(symbol: str, side: str, qty: float, price: float) -> dict:
    """Place a mock order and return Roostoo response JSON."""
    body: Dict[str, Any] = {
        "symbol": symbol.upper(), "side": side.upper(),
        "qty": qty, "price": price,
        "timestamp": int(time.time()*1000)
    }
    payload = "&".join(f"{k}={v}" for k, v in body.items())
    r = requests.post(f"{BASE_URL}/order/new", json=body, headers=_headers(payload), timeout=10)
    r.raise_for_status()
    return r.json()

# -----------------------------------------------------------------
@tool
def analyzeMarket(symbol: str, interval: str = "1m", limit: int = 500) -> dict:
    """
    Fetch klines + compute MA‑cross, RSI, Bollinger signals.
    Returns JSON-safe dict.
    """
    rows = getKlines(symbol, interval, limit)
    df = pd.DataFrame(rows, columns=["ts","o","h","l","c","v"]).astype(float)

    fast = SMAIndicator(df["c"], 9).sma_indicator()
    slow = SMAIndicator(df["c"], 21).sma_indicator()
    ma_cross = _last_cross(fast, slow)

    rsi_val = RSIIndicator(df["c"], 14).rsi().iloc[-1]
    rsi_sig = "oversold" if rsi_val < 30 else "overbought" if rsi_val > 70 else None

    bb = BollingerBands(df["c"], 20, 2)
    price = df["c"].iloc[-1]
    bb_sig = "lower_break" if price < bb.bollinger_lband().iloc[-1] else \
             "upper_break" if price > bb.bollinger_hband().iloc[-1] else None

    return {"price": round(price, 4), "ma_cross": ma_cross, "rsi": rsi_sig, "bollinger": bb_sig}

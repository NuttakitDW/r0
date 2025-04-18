# src/tools/ta_signals.py
import pandas as pd
from ta.trend    import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from langchain.tools import tool

# ── HELPER ---------------------------------------------------------------
def _last_cross(series_fast, series_slow):
    """Return 'bull', 'bear', or None depending on the latest crossover."""
    prev_fast,  prev_slow  = series_fast.iloc[-2], series_slow.iloc[-2]
    curr_fast,  curr_slow  = series_fast.iloc[-1], series_slow.iloc[-1]
    if prev_fast < prev_slow and curr_fast > curr_slow:
        return "bull"
    if prev_fast > prev_slow and curr_fast < curr_slow:
        return "bear"
    return None

# ── TOOL ----------------------------------------------------------------
@tool(args_schema={
    "df":    pd.DataFrame,
    "rules": dict
})
def computeSignals(df: pd.DataFrame, rules: dict):
    """
    Compute MA‑cross, RSI, and Bollinger signals from klines DataFrame.
    Returns dict like:
      { "ma_cross": "bull", "rsi": "oversold", "bollinger": null }
    """
    sig = {}

    # 1) MA‑cross
    ma_cfg = rules["signals"]["ma_cross"]
    fast   = SMAIndicator(df["c"], window=ma_cfg["fast"]).sma_indicator()
    slow   = SMAIndicator(df["c"], window=ma_cfg["slow"]).sma_indicator()
    sig["ma_cross"] = _last_cross(fast, slow)

    # 2) RSI
    rsi_cfg = rules["signals"]["rsi"]
    rsi_val = RSIIndicator(df["c"], window=rsi_cfg["period"]).rsi().iloc[-1]
    if rsi_val < rsi_cfg["oversold"]:
        sig["rsi"] = "oversold"
    elif rsi_val > rsi_cfg["overbought"]:
        sig["rsi"] = "overbought"
    else:
        sig["rsi"] = None

    # 3) Bollinger
    bb_cfg = rules["signals"]["bollinger"]
    bb = BollingerBands(df["c"], window=bb_cfg["period"], window_dev=bb_cfg["dev"])
    price = df["c"].iloc[-1]
    if price < bb.bollinger_lband().iloc[-1]:
        sig["bollinger"] = "lower_break"
    elif price > bb.bollinger_hband().iloc[-1]:
        sig["bollinger"] = "upper_break"
    else:
        sig["bollinger"] = None

    return sig

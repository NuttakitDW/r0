# src/tools.py
"""
Roostoo “tool belt” for the LangGraph agent
------------------------------------------

* Each function is wrapped with @tool so the LLM can see its JSON schema
  via OpenAI / function‑calling.
* TOOL_MAP  : convenient name → Tool lookup
* TOOLS     : list of all Tool objects (pass to ChatOpenAI(functions=...))
* tool_runner() : executes the tool selected by the LLM and returns its JSON

Usage in your graph
-------------------
    from src.tools import TOOLS, tool_runner
    llm = ChatOpenAI(model="gpt-4o-mini", functions=TOOLS, temperature=0)
"""

from __future__ import annotations
from typing import Dict, Any


import src.wrappers as w
from langchain.tools import tool, StructuredTool

# ──────────────────────── PUBLIC (no‑auth) TOOLS ──────────────────────────

@tool
def getServerTime() -> int:
    """Return the exchange's server clock in epoch‑milliseconds."""
    return w.get_server_time()


@tool
def getExchangeInfo() -> dict:
    """Return trading‑pair metadata such as precision and min size."""
    return w.get_exchange_info()


@tool
def getTicker(pair: str) -> dict:
    """Return last price, bid/ask and 24 h stats for a symbol (e.g. "BTC/USD")."""
    return w.get_ticker(pair)


# ─────────────────────── ACCOUNT / ORDER TOOLS (signed) ───────────────────

@tool
def getBalance() -> dict:
    """Return wallet balances for all assets."""
    return w.get_balance()


@tool
def getPendingCount() -> dict:
    """Return the number of currently pending orders."""
    return w.get_pending_count()


# ────────────────────── placeOrder tool (patched) ────────────────────
@tool
def placeOrder(
    pair: str,
    side: str,                             # BUY or SELL
    quantity: str,
    # Accept *either* `type` or `otype` from the LLM ------------------
    type: str | None = None,               # MARKET or LIMIT (preferred)
    otype: str | None = None,              # legacy fallback
    price: float | None = None,            # required for LIMIT
) -> dict:
    """
    Place a MARKET or LIMIT order.

    • `side` and `type` are case‑insensitive; they will be upper‑cased
      before the request is sent.  
    • You may pass EITHER `type` (recommended) OR legacy `otype`.
    • For LIMIT orders you **must** supply `price`.
    """
    # ── normalise inputs ────────────────────────────────────────────
    side_uc  = side.upper()
    t        = (type or otype or "").upper()          # allow both keys

    if side_uc not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    if t not in {"MARKET", "LIMIT"}:
        raise ValueError("type must be MARKET or LIMIT")

    # ── forward to wrapper (wrapper expects otype) ──────────────────
    return w.place_order(pair, side_uc, t, quantity, price)



@tool
def queryOrder(
    order_id: str | None = None,
    pair: str | None = None,
    offset: int | None = None,
    limit: int | None = None,
    pending_only: bool | None = None,
) -> dict:
    """
    Query past or pending orders.
    • Provide order_id OR filters (pair / pending_only / limit).
    """
    return w.query_order(order_id=order_id, pair=pair,
                         offset=offset, limit=limit,
                         pending_only=pending_only)


@tool
def cancelOrder(
    order_id: str | None = None,
    pair: str | None = None,
) -> dict:
    """
    Cancel one, many, or all pending orders.
    • Provide order_id OR pair, or omit both to cancel everything.
    """
    return w.cancel_order(order_id=order_id, pair=pair)


# ────────────────────────── TOOL REGISTRY & DISPATCH ──────────────────────

TOOL_MAP: Dict[str, Any] = {
    t.name: t
    for t in [
        getServerTime,
        getExchangeInfo,
        getTicker,
        getBalance,
        getPendingCount,
        placeOrder,
        queryOrder,
        cancelOrder,
    ]
}

TOOLS = list(TOOL_MAP.values())      # handy for ChatOpenAI(functions=...)

def tool_runner(action_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the JSON function call produced by the LLM.

    action_json = {"tool": "<name>", "args": {...}}
    """
    name = action_json.get("tool")
    args = action_json.get("args", {})

    tool = TOOL_MAP.get(name)
    if tool is None:
        raise ValueError(f"Unknown tool: {name}")

    # StructuredTool: .invoke(...) (or just tool(**args)) handles kwargs
    return tool.invoke(args)             # <— changed line

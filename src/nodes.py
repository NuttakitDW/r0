# src/nodes.py
"""
Brain nodes for R0, the Roostoo‑trading agent.

• think_node – calls GPT‑4o mini with the tool schemas; returns either
  {"action": {...}} or {"result": "..."}.
• act_node   – executes the selected tool via tool_runner and stores the
  wrapper’s JSON response.
"""

from __future__ import annotations
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function

from datetime import datetime

from src.memory import save_memory, retrieve_memory
from src.tools import TOOLS, tool_runner          # tool belt & dispatcher
from src.agent_state import State                 # TypedDict schema

from functools import wraps

import json


# ── 1. SYSTEM PROMPT ──────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are **R0**, an autonomous trading assistant for the Roostoo mock‑exchange.\n\n"

    "💼 **Things you are allowed to do**\n"
    "• Place trades with *exactly* these fields → {pair, side, type, quantity[, price]}  → `placeOrder`\n"
    "  – `side` must be `BUY` or `SELL` (UPPER‑CASE)\n"
    "  – `type` must be `MARKET` or `LIMIT`  (UPPER‑CASE)\n"
    "  – `price` is required only for `LIMIT` orders\n"
    "• Look up balances  → `getBalance`\n"
    "• Look up market price / ticker  → `getTicker`\n"
    "• Get trading rules  → `getExchangeInfo`\n"
    "• Check / cancel / count orders  → `queryOrder`, `cancelOrder`, `getPendingCount`\n\n"

    "📐 **USD‑to‑coin conversion rule**\n"
    "If the user gives an amount in *USD* instead of a coin quantity:\n"
    "1. Call **`getTicker(pair=\"BTC/USD\")`** (or the requested pair) and read `LastPrice`.\n"
    "2. Compute `quantity = usd_amount / LastPrice` and round **to 6 decimals**.\n"
    "3. Use that `quantity` when you call `placeOrder`.\n\n"

    "🚦 **Workflow guidance**\n"
    "• Plan step‑by‑step.  Call as many tools as needed until the task is complete.\n"
    "• After every tool you will see its JSON.  Decide whether another tool call is needed.\n"
    "• When no more tools are required, *summarise the final JSON(s) in plain English*.\n"
    "• If a tool errors (e.g., insufficient balance), explain the problem instead of crashing.\n\n"
    "When you call placeOrder, pass the keys pair, side, type, quantity and (for LIMIT) price.\n"

    "❌ If the user requests anything outside these capabilities, politely refuse.\n"
)


FUNCTION_SCHEMAS = [convert_to_openai_function(t) for t in TOOLS]

# ── 2. INITIALISE LLM WITH TOOL SCHEMAS ───────────────────────────────
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    max_tokens=4096,
    model_kwargs={"functions": FUNCTION_SCHEMAS}, 
)

def log_node(label: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(state: State, *args, **kwargs):
            # identical format at every hop
            print(f"  ↳ {label:<6}", state.get("action"))
            return fn(state, *args, **kwargs)
        return wrapper
    return decorator

# ── 3. THINK NODE  ────────────────────────────────────────────────────
@log_node("think")
def think_node(state: State) -> dict:
    """
    Decide next action based on:
      • recalled memories (if any)
      • previous tool result (if any)
      • newest user prompt
    """
    messages = [SystemMessage(content=SYSTEM_MSG)]

    # ── 1 ▸ inject recalled snippets (oldest→newest) ──────────────
    for chunk in state.get("recalled", []):
        messages.append(AIMessage(content=chunk))

    # ── 2 ▸ include prior tool result, if one exists ─────────────
    if state.get("result") is not None:
        messages.append(AIMessage(content=str(state["result"])))

    # ── 3 ▸ finally add the new user prompt ──────────────────────
    messages.append(HumanMessage(content=state["text"]))

    # ── 4 ▸ call the model with tool schemas bound ───────────────
    resp = llm.invoke(messages)
    fc   = resp.additional_kwargs.get("function_call")

    return {"action": fc} if fc else {"result": resp.content.strip()}

# ── 4. ACT NODE  ──────────────────────────────────────────────────────
@log_node("act")
def act_node(state: State) -> Dict[str, Any]:
    action = state.get("action")
    if not action:
        return {}

    # 1) arguments may be a JSON string; decode to dict
    raw_args = action["arguments"]
    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

    # 2) run the chosen tool with real kwargs
    result = tool_runner({"tool": action["name"], "args": args})
    state["action"] = None

    return {"result": result, "action": None}

# ── 4. MEMORY NODE  ──────────────────────────────────────────────────────
@log_node("memory")
def memory_node(state: State) -> dict:
    text   = state["text"]
    result = state.get("result")         # already a string/int

    if result is not None:               # nothing to save if tool failed
        save_memory(str(result))         # ① store answer only

    recalls = retrieve_memory(text, k=4)
    out = {
        "result": result,          # may be JSON or plain text
        "recalled": recalls,
        "loop_count": state.get("loop_count", 0) + 1
    }
    if result is not None:
        out["result"] = result
    return out
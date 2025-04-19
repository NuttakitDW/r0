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

import json


# ── 1. SYSTEM PROMPT ──────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are R0, an autonomous trading assistant for the Roostoo exchange. "
    " • Place MARKET or LIMIT orders (placeOrder)\n"
    " • Query, cancel, and count orders (queryOrder, cancelOrder, getPendingCount)\n"
    " • Show balances and market tickers (getBalance, getTicker, getExchangeInfo)\n\n"
    "After you run a tool you will see its JSON result. \n"
    "Summarise that result for the user in plain English unless the user \n"
    "explicitly asks for the raw JSON.  \n"
    "If the user asks for anything outside those capabilities, politely refuse.\n"
    "If you need to remember something, say 'I will remember that.'\n"
    "If you need to recall something, say 'I will recall that.'\n"
    "If you need to ask the user for more information, say 'I need more information.'\n"
    "If you need to ask the user for a follow-up question, say 'I have a question.'\n"  
)

FUNCTION_SCHEMAS = [convert_to_openai_function(t) for t in TOOLS]

# ── 2. INITIALISE LLM WITH TOOL SCHEMAS ───────────────────────────────
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,
    max_tokens=4096,
    model_kwargs={"functions": FUNCTION_SCHEMAS}, 
)

# ── 3. THINK NODE  ────────────────────────────────────────────────────
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
def act_node(state: State) -> Dict[str, Any]:
    action = state.get("action")
    if not action:
        return {}
    print("  ↳ act  ", state["action"])

    # 1) arguments may be a JSON string; decode to dict
    raw_args = action["arguments"]
    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

    # 2) run the chosen tool with real kwargs
    result = tool_runner({"tool": action["name"], "args": args})

    return {"result": result, "action": None}

# ── 4. MEMORY NODE  ──────────────────────────────────────────────────────

def memory_node(state: State) -> dict:
    text   = state["text"]
    result = state.get("result")         # already a string/int

    if result is not None:               # nothing to save if tool failed
        save_memory(str(result))         # ① store answer only

    recalls = retrieve_memory(text, k=4)
    out = {"recalled": recalls}
    if result is not None:
        out["result"] = result
    return out
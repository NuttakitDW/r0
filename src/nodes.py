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
    "You are **R0**, an autonomous trading assistant for the Roostoo "
    "exchange. Your allowed actions are strictly limited to:\n"
    " • Place MARKET or LIMIT orders (placeOrder)\n"
    " • Query, cancel, and count orders (queryOrder, cancelOrder, getPendingCount)\n"
    " • Show balances and market tickers (getBalance, getTicker, getExchangeInfo)\n\n"
    "If you see “recalled memories” in the conversation, use them to answer the user’s question accurately.\n"
    "If the user asks for anything outside those capabilities, politely refuse.\n"
)

FUNCTION_SCHEMAS = [convert_to_openai_function(t) for t in TOOLS]

# ── 2. INITIALISE LLM WITH TOOL SCHEMAS ───────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    model_kwargs={"functions": FUNCTION_SCHEMAS}, 
)

# ── 3. THINK NODE  ────────────────────────────────────────────────────
def think_node(state: State) -> Dict[str, Any]:
    print("➜ think", state.get("action"))
    """
    Decide next action based on the latest user prompt *and* the
    previous tool result (if any). Returns either:

        {"action": {...}}  – a JSON function call selected by the LLM
        {"result": "..."}  – final natural‑language answer
    """
    # Build message list
    messages = [SystemMessage(content=SYSTEM_MSG)]
    
    for chunk in state.get("recalled", []):
        messages.append(AIMessage(content=chunk))

    # Provide last tool result (if any) so the LLM can chain reasoning
    if state.get("result") is not None:
        messages.append(AIMessage(content=str(state["result"])))

    # Latest user input
    messages.append(HumanMessage(content=state["text"]))

    # Call the model
    resp = llm.invoke(messages)

    fc = resp.additional_kwargs.get("function_call")
    if fc:
        # LLM chose one of the registered tools
        return {"action": fc}

    # No tool call => conversation finished
    return {"result": resp.content}


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
    """Write to Pinecone and return recalls (always include the key)."""
    text   = state["text"]
    result = state.get("result")

    # ① save current turn
    blob = (
        f"[{datetime.utcnow().isoformat(timespec='seconds')}]\n"
        f"USER: {text}\nASSISTANT: {result}"
    )
    save_memory(blob)

    # ② pull top‑k similar
    recalls = retrieve_memory(text, k=4)

    # ③ always include 'recalled'; forward result only if present
    out = {"recalled": recalls}
    if result is not None:
        out["result"] = result
    return out

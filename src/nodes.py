# src/nodes.py  ─────────────────────────────────────────────────────────
"""
Brain nodes for R0, the Roostoo‑trading agent.

• think_node – talks to GPT‑4o mini with tool schemas; returns either
  {"action": {...}} or {"result": "..."}.
• act_node   – executes the selected tool and stores the JSON response.
"""
from __future__ import annotations
from typing import Dict, Any
from functools import wraps
import json
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function

from src.memory import save_memory, retrieve_memory
from src.tools import TOOLS, tool_runner
from src.agent_state import State

class RoostooError(RuntimeError):
    """Raised when the exchange rejects the request or returns non‑200."""

# ── 1. SYSTEM PROMPT & TOOL SCHEMAS ───────────────────────────────────
SYSTEM_MSG = (
    "You are **R0**, an autonomous trading assistant for the Roostoo mock‑exchange.\n\n"
    # … (prompt text unchanged, clipped for brevity) …
    "❌ If the user requests anything outside these capabilities, politely refuse.\n"
)
FUNCTION_SCHEMAS = [convert_to_openai_function(t) for t in TOOLS]

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    max_tokens=4096,
    model_kwargs={"functions": FUNCTION_SCHEMAS},
)

# ── 2. SIMPLE LOGGING DECORATOR (prints *after* the node runs) ─────────
def log_node(label: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(state: State, *args, **kwargs):
            new_state = fn(state, *args, **kwargs)
            print(f"  ↳ {label:<6}", new_state.get("action"))
            return new_state
        return wrapper
    return decorator

# ── 3. THINK NODE ─────────────────────────────────────────────────────
@log_node("think")
def think_node(state: State) -> Dict[str, Any]:
    """
    Decide the next step.  If we already executed an action in the previous
    loop and the LLM wants to call the *same* tool again, suppress the repeat
    and just return its textual response instead.
    """
    messages = [SystemMessage(content=SYSTEM_MSG)]

    # 1 ▸ recalled memories (oldest → newest)
    for chunk in state.get("recalled", []):
        messages.append(AIMessage(content=chunk))

    # 2 ▸ previous tool result (if any)
    if state.get("result") is not None:
        messages.append(AIMessage(content=str(state["result"])))
        
    if state.get("error"):
        messages.append(AIMessage(content=f"⚠️ ERROR: {state['error']}"))

    # 3 ▸ newest user prompt
    messages.append(HumanMessage(content=state["text"]))

    # 4 ▸ call the model
    resp = llm.invoke(messages)
    fc = resp.additional_kwargs.get("function_call")

    last_act = state.get("last_action")           # may be None

    # ‑‑‑ suppress identical repeat calls ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
    if (
        fc
        and last_act
        and fc.get("name") == last_act.get("name")
        and fc.get("arguments") == last_act.get("arguments")
    ):
        # treat this as the LLM's natural language answer
        return {"action": None, "result": resp.content.strip()}

    return {"action": fc} if fc else {"result": resp.content.strip()}

# ── 4. ACT NODE ───────────────────────────────────────────────────────
@log_node("act")
def act_node(state: State) -> Dict[str, Any]:
    action = state.get("action")
    if not action:
        return {"action": None}

    # 1 ▸ normalise args
    raw_args = action["arguments"]
    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

    # 2 ▸ run the tool, catching exchange errors
    try:
        result = tool_runner({"tool": action["name"], "args": args})
        error  = None
    except RoostooError as exc:
        result = None
        error  = str(exc)

    # 3 ▸ return **both** result *or* error, plus bookkeeping
    return {
        "result": result,           # dict | None
        "error":  error,            # str  | None
        "action": None,
        "last_action": action,
    }

# ── 5. MEMORY NODE ────────────────────────────────────────────────────
@log_node("memory")
def memory_node(state: State) -> Dict[str, Any]:
    text   = state["text"]
    result = state.get("result")

    # ① store only successful results
    if result is not None:
        save_memory(str(result))
    if state.get("error"):
        save_memory(state["error"])

    recalls = retrieve_memory(text, k=4)

    return {
        "result":   result,
        "recalled": recalls,
        "loop_count": state.get("loop_count", 0) + 1,
        "last_action": state.get("last_action"),  # carry forward
        "action": None,                           # stay explicit
    }

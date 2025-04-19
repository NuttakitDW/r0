# src/nodes.py
"""
Brain nodes for R0

• think_node – LLM reasoning: decide on next tool call (function‑call) or
  return a final answer.
• act_node   – executes the chosen tool via tool_runner and stores result.

These nodes are wired together in agent_graph.py.
"""

from __future__ import annotations
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from src.tools import TOOLS, tool_runner           # tool belt
from src.agent_state import State                  # TypedDict schema


# ── 1. SYSTEM PROMPT ───────────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are **R0**, an autonomous trading assistant for the Roostoo "
    "exchange. Your allowed actions are strictly limited to:\n"
    " • Place MARKET or LIMIT orders (placeOrder)\n"
    " • Query, cancel, or count orders\n"
    " • Show balances and market tickers\n\n"
    "If the user asks for anything outside those capabilities, politely refuse."
)

# ── 2. INITIALISE LLM WITH TOOL SCHEMAS ────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    functions=TOOLS,      # OpenAI will see each tool's JSON schema
)

# ── 3. THINK NODE  (LLM REASONING) ─────────────────────────────────────────
def think_node(state: State) -> Dict[str, Any]:
    """
    Decide on the next action based on the latest user prompt *and* the
    previous tool result (if any). Returns either:
      • {"action": {...}}  – a JSON function call, or
      • {"result": "text"} – a final natural‑language reply
    """
    # Build the message list (include system prompt every turn)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
    ]

    # Optionally include the last tool result for context
    if "result" in state and state["result"] is not None:
        messages.append(
            {"role": "assistant", "content": str(state["result"])}
        )

    # Finally, the new user prompt
    messages.append(HumanMessage(content=state["text"]))

    # Call the LLM
    resp = llm.invoke(messages)

    # If the model picked a tool, resp.additional_kwargs has "function_call"
    fc = resp.additional_kwargs.get("function_call")
    if fc:
        return {"action": fc}               # hand over to act_node

    # No tool call → conversation is over
    return {"result": resp.content}


# ── 4. ACT NODE  (TOOL EXECUTION) ─────────────────────────────────────────
def act_node(state: State) -> Dict[str, Any]:
    """
    Executes the previously selected tool (`state["action"]`) and returns
    its JSON response.
    """
    action = state.get("action")
    if not action:
        # Nothing to do; safeguard if graph mis‑wired
        return {}

    # Run the tool (this may raise RoostooError, propagated upward)
    result = tool_runner({"tool": action["name"], "args": action["arguments"]})

    # Store result; clear action so think_node knows when to stop
    return {"result": result, "action": None}

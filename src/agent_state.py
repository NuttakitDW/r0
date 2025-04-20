# src/agent_state.py
from __future__ import annotations

from typing import (
    TypedDict,
    NotRequired,      # Python ≥3.11
    Optional,
    Any,
    Dict,
    List,
)

# ── Helper type for the tool call structure ───────────────────────────
class ToolCall(TypedDict):
    name: str                  # e.g. "getBalance"
    arguments: Dict[str, Any]  # raw JSON‑serialisable kwargs

# ── Core agent state passed between LangGraph nodes ───────────────────
class State(TypedDict, total=False):
    # ── user input & memory context ────────────────────────────────
    text: str                               # latest user prompt
    recalled: NotRequired[List[str]]        # memory snippets (oldest→newest)

    # ── tool‑execution workflow ────────────────────────────────────
    action: Optional[ToolCall]              # tool queued for *next* act pass
    last_action: NotRequired[ToolCall]      # tool that was just executed
    result: Optional[Any]                   # tool JSON or final summary

    # ── flow‑control guards ────────────────────────────────────────
    loop_count: int                         # safety breaker (default 0)
    error:  NotRequired[str]  


# ── tiny convenience helper ----------------------------------------------------
def make_state(text: str) -> State:
    """
    Initialise a fresh State dict with sensible defaults.
    """
    return {
        "text": text,
        "action": None,
        "loop_count": 0,
    }

# src/agent_state.py
"""
TypedDict that describes the transient state passed between LangGraph
nodes during one agent run.

Keys can be added later (e.g., loop_count, error, summary) by editing
this file only—nodes and graph will pick them up automatically.
"""

from typing import TypedDict, List, Dict, Any, Optional


class State(TypedDict, total=False):
    # ── prompt & chat ──────────────────────────────────────────────
    text: str
    history: List[Dict[str, Any]]

    # ── tool execution loop ────────────────────────────────────────
    action: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any] | str]

    # ── memory ­────────────────────────────────────────────────────
    recalled: List[str]                 # <‑‑ NEW: always a list (may be empty)

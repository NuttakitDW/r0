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
    text: str                                 # latest user prompt
    history: List[Dict[str, Any]]             # optional chat or trace log

    # ── tool execution loop ────────────────────────────────────────
    action: Optional[Dict[str, Any]]          # JSON tool call chosen by LLM
    result: Optional[Dict[str, Any] | str]    # wrapper response OR final answer

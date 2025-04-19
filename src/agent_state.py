# src/agent_state.py

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional   # ← make sure this line exists

class State(TypedDict, total=False):
    text: str
    action: dict | None
    result: dict | str | None
    recalled: list[str]
    loop_count: int           # <‑‑ add guard

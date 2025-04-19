"""
agent_graph.py
Builds the LangGraph that flows:
    classify ➜ extract ➜ summarize ➜ END
and exposes `analyze(text)` for quick use.
"""

from langgraph.graph import StateGraph, END
from src.agent_state import State
from src.nodes import (
    classification_node,
    entity_extraction_node,
    summarize_node,
)

# ── 1.  Build the graph skeleton -------------------------------------
wf = StateGraph(State)

wf.add_node("classify",  classification_node)
wf.add_node("extract",   entity_extraction_node)
wf.add_node("summarize", summarize_node)

# ── 2.  Connect nodes -------------------------------------------------
wf.set_entry_point("classify")
wf.add_edge("classify",  "extract")
wf.add_edge("extract",   "summarize")
wf.add_edge("summarize", END)

# ── 3.  Compile to a runnable app ------------------------------------
app = wf.compile()

# Convenience wrapper --------------------------------------------------
def analyze(text: str) -> State:
    """
    Run the full flow on `text` and return the final state dict.
    Usage:
        from src.agent_graph import analyze
        result = analyze("Some Medium article ...")
    """
    return app.invoke({"text": text})

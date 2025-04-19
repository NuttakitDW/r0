# src/agent_graph.py
"""
LangGraph wiring for R0
-----------------------

Flow:
    think  ──►  act  ──►  think (loop)…  ──►  END
"""

from langgraph.graph import StateGraph, END

from src.agent_state import State
from src.nodes import think_node, act_node


# ── 1. Build the graph ─────────────────────────────────────────────
wf = StateGraph(State)

wf.add_node("think", think_node)
wf.add_node("act",   act_node)

wf.set_entry_point("think")

# After every act, jump back to think so the LLM can decide next step
wf.add_edge("act", "think")

# Branch out of the loop when think_node returns no action
wf.add_conditional_edges(
    "think",
    {
        # go to "act" while there's still a JSON tool call to execute
        "act": lambda s: s.get("action") is not None,

        # otherwise finish the workflow
        END:   lambda s: s.get("action") is None,
    },
)

# ── 2. Compile to an executable app ────────────────────────────────
app = wf.compile()

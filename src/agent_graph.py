from langgraph.graph import StateGraph, END
from src.agent_state import State
from src.nodes import think_node, act_node

wf = StateGraph(State)

wf.add_node("think", think_node)
wf.add_node("act",   act_node)

wf.set_entry_point("think")

# ── Conditional branching from THINK ───────────────────────────────
# If the LLM produced a tool‑call, run ACT; otherwise we're done.
wf.add_conditional_edges(
    "think",
    lambda s: s.get("action") is None,   # True → stop
    {
        True:  END,                      # no action  → END
        False: "act",                    # have action → run tool once
    },
)

# ── NO edge from act → think (tool runs only once per user turn) ────

app = wf.compile()

from langgraph.graph import StateGraph, END
from src.agent_state import State
from src.nodes import think_node, act_node, memory_node   # updated

wf = StateGraph(State)

wf.add_node("think", think_node)
wf.add_node("act",   act_node)
wf.add_node("memory", memory_node) 

wf.set_entry_point("think")
wf.add_edge("think",  "act")
wf.add_edge("act",    "memory")
wf.add_edge("memory", END)

# ── Conditional branching from THINK ───────────────────────────────
# If the LLM produced a tool‑call, run ACT; otherwise we're done.
wf.add_conditional_edges(           # skip act if no tool
    "think",
    lambda s: s.get("action") is None,
    {True: END, False: "act"},
)

# ── NO edge from act → think (tool runs only once per user turn) ────

app = wf.compile()

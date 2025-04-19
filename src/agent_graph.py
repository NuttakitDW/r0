# src/agent_graph.py  ───────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from src.agent_state import State
from src.nodes import think_node, act_node, memory_node

wf = StateGraph(State)

# nodes
wf.add_node("think",  think_node)
wf.add_node("act",    act_node)
wf.add_node("memory", memory_node)

# entry
wf.set_entry_point("think")

# core edges
wf.add_edge("act",    "memory")   # always go through memory
wf.add_edge("memory", "think")    # think again with new context

# single decision out of THINK
wf.add_conditional_edges(         # decide next hop *from think*
    "think",
    lambda s: (s.get("action") is not None)        # if a tool is queued
                and s.get("loop_count", 0) < 4,    #   and safety cap OK
    {True: "act", False: END},
)

# guard the whole graph too (failsafe)

app = wf.compile()

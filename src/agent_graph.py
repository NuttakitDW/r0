# src/agent_graph.py
# ──────────────────────────────────────────────────────────────────────
"""
Execution graph for R0, the Roostoo trading agent.

Flow:
        ┌────────────┐
        │  think     │  — decide what to do next
        └────┬───────┘
             │   (queues   state["action"] ?)
   action queued? │
             │
    yes      ▼                no / safety cap
        ┌────────────┐                       (END)
        │   act      │  — execute the tool
        └────┬───────┘
             │
             ▼
        ┌────────────┐
        │  memory    │  — store / recall
        └────┬───────┘
             │
             └──────────────► back to think
"""

from langgraph.graph import StateGraph, END

from src.agent_state import State
from src.nodes import think_node, act_node, memory_node

# ── build the state machine ───────────────────────────────────────────
wf = StateGraph(State)

wf.add_node("think",  think_node)
wf.add_node("act",    act_node)
wf.add_node("memory", memory_node)

# entry point
wf.set_entry_point("think")

# after executing a tool we always store / recall memory
wf.add_edge("act", "memory")

# after memory we think again with the new context
wf.add_edge("memory", "think")

# ── single conditional branch out of THINK ────────────────────────────
SAFETY_CAP = 10            # absolute max loops, adjust as you like

def need_to_act(state: State) -> bool:
    """
    Return True if a tool is queued *and* we are still below the safety cap.
    """
    return bool(state.get("action")) and state.get("loop_count", 0) < SAFETY_CAP

wf.add_conditional_edges(
    "think",
    need_to_act,
    {True: "act", False: END},
)

# ── compile to a runnable app ─────────────────────────────────────────
app = wf.compile()

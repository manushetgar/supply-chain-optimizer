"""LangGraph orchestration for the supply chain mitigation workflow.

Builds a stateful graph:

    analyze_root_cause -> propose_mitigation -> [INTERRUPT] -> execute_erp_write

Execution pauses before ``execute_erp_write`` via ``interrupt_before``, so a
human planner can review the SHAP-derived reasoning and the proposed action.
State is persisted across the interrupt by a checkpointer, allowing the graph
to resume on the same thread after human approval is injected.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import analyze_root_cause, execute_erp_write, propose_mitigation
from src.agent.state import SupplyChainState


def build_graph():
    """Compile the supply chain mitigation graph with a human-in-the-loop gate.

    Returns:
        A compiled LangGraph app with an in-memory checkpointer and an
        interrupt configured before the ERP write-back node.
    """
    workflow = StateGraph(SupplyChainState)

    workflow.add_node("analyze_root_cause", analyze_root_cause)
    workflow.add_node("propose_mitigation", propose_mitigation)
    workflow.add_node("execute_erp_write", execute_erp_write)

    workflow.add_edge(START, "analyze_root_cause")
    workflow.add_edge("analyze_root_cause", "propose_mitigation")
    workflow.add_edge("propose_mitigation", "execute_erp_write")
    workflow.add_edge("execute_erp_write", END)

    # MemorySaver persists thread state in-memory (SQLite-style semantics),
    # enabling pause/resume across the human-in-the-loop interrupt.
    checkpointer = MemorySaver()

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute_erp_write"],
    )

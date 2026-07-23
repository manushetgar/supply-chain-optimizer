"""Phase 3 tests: agent graph pauses at HITL interrupt and resumes to completion."""

from __future__ import annotations

from typing import Dict

from src.agent.graph import build_graph
from src.agent.state import SupplyChainState


def _initial_state() -> SupplyChainState:
    """Build a representative ML payload as graph input."""
    return SupplyChainState(
        prediction_data={"prediction": 0.92},
        shap_explanation={
            "vendor_lead_time_deviation": 0.41,
            "safety_stock_penetration_ratio": 0.28,
            "days_of_cover": -0.19,
        },
        audit_log=[],
    )


def test_graph_pauses_at_human_in_the_loop_interrupt() -> None:
    """Graph must halt before execute_erp_write with a proposed action ready."""
    app = build_graph()
    config: Dict = {"configurable": {"thread_id": "test-thread-1"}}

    app.invoke(_initial_state(), config=config)
    snapshot = app.get_state(config)

    # Execution is paused; the next node is the guarded ERP write.
    assert snapshot.next == ("execute_erp_write",)
    # Reasoning and a structured proposal exist before any execution.
    assert snapshot.values["root_cause_summary"]
    assert snapshot.values["proposed_action"]["action"] in {
        "expedite_purchase_order",
        "transfer_plant_stock",
    }
    # No terminal decision yet.
    assert not snapshot.values.get("final_decision")


def test_graph_resumes_after_human_approval() -> None:
    """Injecting approval into the paused state must execute the write-back."""
    app = build_graph()
    config: Dict = {"configurable": {"thread_id": "test-thread-2"}}

    app.invoke(_initial_state(), config=config)

    # Human planner approves: inject feedback into the persisted snapshot.
    app.update_state(config, {"human_feedback": {"decision": "approve"}})

    # Resume from the interrupt (None continues the paused thread).
    result = app.invoke(None, config=config)

    assert result["final_decision"]["status"] == "EXECUTED"
    assert "SIMULATED-SAP-WRITEBACK" in result["final_decision"]["erp_document"]


def test_graph_respects_human_rejection() -> None:
    """A rejection must prevent ERP execution."""
    app = build_graph()
    config: Dict = {"configurable": {"thread_id": "test-thread-3"}}

    app.invoke(_initial_state(), config=config)
    app.update_state(config, {"human_feedback": {"decision": "reject"}})
    result = app.invoke(None, config=config)

    assert result["final_decision"]["status"] == "REJECTED"


def test_lead_time_driver_selects_expedite_action() -> None:
    """Dominant lead-time SHAP driver must yield an expedite-PO proposal."""
    app = build_graph()
    config: Dict = {"configurable": {"thread_id": "test-thread-4"}}
    state = SupplyChainState(
        prediction_data={"prediction": 0.88},
        shap_explanation={"vendor_lead_time_deviation": 0.9, "days_of_cover": -0.1},
        audit_log=[],
    )
    app.invoke(state, config=config)
    snapshot = app.get_state(config)
    assert snapshot.values["proposed_action"]["action"] == "expedite_purchase_order"

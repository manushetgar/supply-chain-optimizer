"""Stateless node functions for the supply chain mitigation graph.

Each node is a pure function: it reads the incoming :class:`SupplyChainState`
and returns a partial dict of updates. LangGraph merges these updates into the
persisted state. No node mutates external globals.
"""

from __future__ import annotations

from typing import Dict

from src.agent.llm import get_llm
from src.agent.state import SupplyChainState
from src.agent.tools import MitigationDecision, MitigationType


def analyze_root_cause(state: SupplyChainState) -> Dict:
    """Generate a natural-language root-cause narrative from SHAP values.

    Args:
        state: Current graph state containing ``prediction_data`` and
            ``shap_explanation``.

    Returns:
        Partial state update with ``root_cause_summary`` and an audit entry.
    """
    llm = get_llm()
    prediction = float(state["prediction_data"].get("prediction", 0.0))
    shap = state.get("shap_explanation", {})
    summary = llm.summarize_risk(prediction, shap)

    audit = list(state.get("audit_log", []))
    audit.append("analyze_root_cause: generated risk narrative")
    return {"root_cause_summary": summary, "audit_log": audit}


def propose_mitigation(state: SupplyChainState) -> Dict:
    """Propose a structured mitigation action via simulated tool-calling.

    The decision policy is deterministic and driven by which SHAP feature
    dominates: lead-time-driven risk -> expedite PO; stock-depletion-driven
    risk -> inter-plant transfer.

    Args:
        state: Current graph state with SHAP explanation and prediction.

    Returns:
        Partial state update with a JSON-serializable ``proposed_action``.
    """
    shap = state.get("shap_explanation", {})
    prediction = float(state["prediction_data"].get("prediction", 0.0))

    # Identify the dominant risk driver by absolute SHAP contribution.
    dominant = max(shap.items(), key=lambda kv: abs(kv[1]), default=("", 0.0))[0]

    lead_time_signals = {"vendor_lead_time_deviation", "actual_lead_time", "lead_time_delay"}
    if dominant in lead_time_signals:
        decision = MitigationDecision(
            action=MitigationType.EXPEDITE_PURCHASE_ORDER,
            parameters={"po_number": "4500001234", "requested_days_earlier": 5},
            rationale=(
                "Lead-time deviation is the dominant driver; expediting the "
                "open PO restores cover before the buffer is breached."
            ),
            confidence=min(0.99, 0.5 + prediction / 2),
        )
    else:
        decision = MitigationDecision(
            action=MitigationType.TRANSFER_PLANT_STOCK,
            parameters={
                "source_plant": "PL02",
                "target_plant": "PL00",
                "quantity": 120.0,
            },
            rationale=(
                "Stock depletion is the dominant driver; an inter-plant "
                "transfer bridges the shortage faster than procurement."
            ),
            confidence=min(0.99, 0.5 + prediction / 2),
        )

    audit = list(state.get("audit_log", []))
    audit.append(f"propose_mitigation: {decision.action.value}")
    return {"proposed_action": decision.model_dump(mode="json"), "audit_log": audit}


def execute_erp_write(state: SupplyChainState) -> Dict:
    """Simulate the approved ERP write-back (guarded by human-in-the-loop).

    This node only runs after the graph resumes from its interrupt, i.e. after
    a human planner has approved the action. It simulates a REST call to SAP.

    Args:
        state: Current graph state including ``human_feedback`` and the
            ``proposed_action``.

    Returns:
        Partial state update with the terminal ``final_decision``.
    """
    feedback = state.get("human_feedback") or {}
    decision = feedback.get("decision", "reject")
    audit = list(state.get("audit_log", []))

    if decision == "approve":
        action = state.get("proposed_action", {})
        final = {
            "status": "EXECUTED",
            "erp_document": "SIMULATED-SAP-WRITEBACK-0001",
            "executed_action": action.get("action"),
            "message": "Simulated REST write-back to SAP S/4HANA succeeded.",
        }
        audit.append("execute_erp_write: action executed against ERP (simulated)")
    else:
        final = {
            "status": "REJECTED",
            "message": "Human planner rejected the proposed mitigation.",
        }
        audit.append("execute_erp_write: action rejected by human planner")

    return {"final_decision": final, "audit_log": audit}

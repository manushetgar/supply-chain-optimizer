"""State schema for the LangGraph supply chain mitigation agent.

The :class:`SupplyChainState` TypedDict is the single persisted memory object
that flows through every node of the graph. Nodes are pure functions that read
from and return partial updates to this state.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


class SupplyChainState(TypedDict, total=False):
    """Typed state carried through the agentic decision workflow.

    Attributes:
        prediction_data: Raw ML payload (probability + context).
        shap_explanation: Top contributing features and their SHAP scores.
        root_cause_summary: LLM-generated natural-language risk narrative.
        proposed_action: The mitigation action proposed by the agent.
        human_feedback: Human planner's decision ("approve"/"reject") + notes.
        final_decision: Terminal outcome after (simulated) ERP write-back.
    """

    prediction_data: Dict[str, float]
    shap_explanation: Dict[str, float]
    root_cause_summary: str
    proposed_action: Dict[str, object]
    human_feedback: Optional[Dict[str, str]]
    final_decision: Optional[Dict[str, object]]
    audit_log: List[str]

"""Pydantic tool schemas enforcing parsable, structured agent decisions.

Using Pydantic models as tool schemas guarantees the agent's proposed
mitigation is strictly typed and JSON-serializable, which is essential when the
output is consumed by downstream automation (the simulated ERP write-back).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MitigationType(str, Enum):
    """Enumerated set of supported supply chain mitigation actions."""

    EXPEDITE_PURCHASE_ORDER = "expedite_purchase_order"
    TRANSFER_PLANT_STOCK = "transfer_plant_stock"
    REROUTE_INBOUND_LOGISTICS = "reroute_inbound_logistics"
    NO_ACTION = "no_action"


class ExpeditePurchaseOrder(BaseModel):
    """Schema for expediting an open purchase order."""

    action: MitigationType = Field(default=MitigationType.EXPEDITE_PURCHASE_ORDER)
    po_number: str = Field(description="Purchasing document number to expedite.")
    requested_days_earlier: int = Field(
        ge=1, description="Days to pull the delivery date forward."
    )
    rationale: str = Field(description="Why this action mitigates the risk.")


class TransferPlantStock(BaseModel):
    """Schema for an inter-plant stock transfer."""

    action: MitigationType = Field(default=MitigationType.TRANSFER_PLANT_STOCK)
    source_plant: str = Field(description="Plant supplying surplus stock.")
    target_plant: str = Field(description="Plant receiving the transfer.")
    quantity: float = Field(gt=0, description="Quantity to transfer.")
    rationale: str = Field(description="Why this action mitigates the risk.")


class MitigationDecision(BaseModel):
    """Top-level structured decision emitted by the propose_mitigation node."""

    action: MitigationType
    parameters: dict = Field(default_factory=dict)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)

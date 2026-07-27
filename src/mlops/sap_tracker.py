"""Mock of the SAP Cloud SDK for AI (Core) tracking integration.

In a productive SAP AI Core workflow execution, metrics and tags are persisted
to the SAP AI Launchpad via the ``ai_core_sdk`` (``sap-ai-sdk-core``) tracking
module. Because SAP credentials and the productive runtime are unavailable in a
local/offline context, this module provides a faithful mock that mirrors the
exact method signatures used in production, with the real calls documented
inline.
"""

from __future__ import annotations

from typing import Dict, List


class MockTracking:
    """Local stand-in for ``ai_core_sdk.tracking.Tracking``.

    Mirrors the ``modify`` signature used to log metrics and custom tags back
    to an SAP AI Core execution so the integration pattern is demonstrable
    without a live SAP backend.
    """

    def __init__(self) -> None:
        self.logged_metrics: List[Dict] = []
        self.logged_tags: List[Dict] = []

    def modify(
        self,
        *,
        tags: List[Dict[str, str]] | None = None,
        metrics: List[Dict[str, object]] | None = None,
    ) -> None:
        """Persist metrics and tags to the (mocked) SAP AI Core execution.

        In production this call would look like::

            from ai_core_sdk.tracking import Tracking
            from ai_core_sdk.models import Metric, MetricTag

            Tracking().modify(
                tags=[MetricTag(name="model_type", value="focal_mlp")],
                metrics=[
                    Metric(name="PR-AUC", value=0.81, step=0, timestamp=...),
                    Metric(name="F2", value=0.74, step=0, timestamp=...),
                ],
            )

        Args:
            tags: List of ``{"name": ..., "value": ...}`` custom tags.
            metrics: List of ``{"name": ..., "value": ...}`` metric records.
        """
        if tags:
            self.logged_tags.extend(tags)
        if metrics:
            self.logged_metrics.extend(metrics)


def log_to_sap_ai_core(
    metrics: Dict[str, float], tags: Dict[str, str]
) -> MockTracking:
    """Simulate persisting run metrics/tags to SAP AI Core.

    Args:
        metrics: Mapping of metric name to value (e.g. PR-AUC, F2).
        tags: Mapping of custom tag name to value.

    Returns:
        The :class:`MockTracking` instance holding the recorded payloads.
    """
    tracker = MockTracking()
    tracker.modify(
        tags=[{"name": k, "value": v} for k, v in tags.items()],
        metrics=[{"name": k, "value": float(v)} for k, v in metrics.items()],
    )
    return tracker

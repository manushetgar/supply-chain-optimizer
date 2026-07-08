"""Offline-first LLM abstraction for the supply chain agent.

To keep the entire pipeline runnable without network access or API keys, the
default provider is a deterministic ``MockLLM`` that produces plausible,
template-driven natural language and structured tool selections from the SHAP
payload. The interface mirrors a minimal chat-completion contract so a real
backend (e.g. the SAP Cloud SDK for AI generative module, ``sap-ai-sdk-gen``,
or an Anthropic client) can be dropped in later.
"""

from __future__ import annotations

from typing import Dict, Protocol

from src.config import settings


class LLMClient(Protocol):
    """Minimal LLM contract used by the agent nodes."""

    def summarize_risk(self, prediction: float, shap: Dict[str, float]) -> str:
        """Return a natural-language risk narrative for a planner."""
        ...


class MockLLM:
    """Deterministic, offline stand-in for a production LLM.

    Generates a readable root-cause narrative by mapping known SHAP feature
    names to human phrasing. Fully deterministic for reproducible tests.
    """

    _FEATURE_PHRASES: Dict[str, str] = {
        "vendor_lead_time_deviation": "the vendor is delivering later than planned",
        "safety_stock_penetration_ratio": "stock has fallen into the safety buffer",
        "days_of_cover": "remaining days of inventory cover are low",
        "rolling_7d_consumption": "recent 7-day consumption has spiked",
        "rolling_30d_consumption": "30-day consumption is trending upward",
        "unrestricted_stock": "unrestricted stock on hand is depleted",
        "actual_lead_time": "the actual replenishment lead time is elevated",
        "lead_time_delay": "the vendor lead time is delayed",
        "low_safety_stock": "the safety stock level is critically low",
    }

    def summarize_risk(self, prediction: float, shap: Dict[str, float]) -> str:
        """Compose a planner-facing narrative from prediction + SHAP values.

        Args:
            prediction: Predicted stock-out probability in [0, 1].
            shap: Mapping of feature name -> SHAP contribution.

        Returns:
            A concise, human-readable risk explanation.
        """
        ranked = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)
        drivers = []
        for name, score in ranked[:3]:
            phrase = self._FEATURE_PHRASES.get(name, f"'{name}' is anomalous")
            direction = "increasing" if score > 0 else "reducing"
            drivers.append(f"{phrase} ({direction} risk, SHAP={score:+.3f})")

        driver_text = "; ".join(drivers) if drivers else "no dominant driver identified"
        return (
            f"Stock-out risk is {prediction:.0%}. Primary drivers: {driver_text}. "
            "Recommend proactive mitigation before the buffer is exhausted."
        )


def get_llm() -> LLMClient:
    """Factory returning the configured LLM client.

    Returns:
        A :class:`MockLLM` when ``LLM_PROVIDER=mock`` (the offline default).

    Raises:
        NotImplementedError: If a non-mock provider is requested. Real
            providers (e.g. ``sap-ai-sdk-gen``) can be wired in here.
    """
    if settings.llm_provider == "mock":
        return MockLLM()
    raise NotImplementedError(
        f"LLM provider {settings.llm_provider!r} not wired; use 'mock' for offline runs."
    )

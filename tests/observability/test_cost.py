"""Tests for the pluggable cost-extractor fallback shim.

``ModelResponse.cost()`` normally calls ``genai_prices.calc_price()``;
when genai-prices doesn't know the model it raises ``LookupError``.
``install_cost_fallback_patch`` extends that behaviour so registered
``CostExtractor`` strategies can supply a real billed cost — preserves
the pydantic-ai ``operation.cost`` span attribute path without
copy-pasting a custom decorator per app.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

# Import framework so the patch is installed.
import pydantic_ai_stateflow  # noqa: F401
from pydantic_ai_stateflow.observability.cost import (
    OpenRouterCostExtractor,
    OpenRouterUpstreamCostExtractor,
    ProviderDetailsCostExtractor,
    clear_cost_extractors,
    configure_cost_extractors,
    register_cost_extractor,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Each test starts with a clean extractor registry."""
    clear_cost_extractors()
    yield
    clear_cost_extractors()


def _stub_response_with_provider_details(details: dict) -> object:
    """ModelResponse stub that the cost() patch can introspect.

    We mimic the minimal shape: ``provider_details`` dict + a
    ``cost`` method bound to the real patched implementation. Using a
    real ``ModelResponse`` would require building text/usage fields too;
    a stub keeps the test focused on the fallback path.
    """
    from pydantic_ai.messages import ModelResponse

    msg = ModelResponse(parts=[])
    msg.provider_details = details  # type: ignore[attr-defined]
    msg.model_name = "fake-model"
    return msg


def test_openrouter_extractor_reads_provider_details_cost() -> None:
    register_cost_extractor(OpenRouterCostExtractor())
    msg = _stub_response_with_provider_details({"cost": 0.00045})

    result = msg.cost()

    assert result.total_price == Decimal("0.00045")


def test_returns_none_when_no_extractor_matches() -> None:
    register_cost_extractor(OpenRouterCostExtractor())
    msg = _stub_response_with_provider_details({})

    with pytest.raises(LookupError):
        msg.cost()


def test_upstream_inference_extractor_reads_different_field() -> None:
    register_cost_extractor(OpenRouterUpstreamCostExtractor())
    msg = _stub_response_with_provider_details(
        {"upstream_inference_cost": 0.00012},
    )

    result = msg.cost()

    assert result.total_price == Decimal("0.00012")


def test_extractors_chain_first_match_wins() -> None:
    """Extractors are tried in registration order; first non-None wins."""
    register_cost_extractor(OpenRouterUpstreamCostExtractor())
    register_cost_extractor(OpenRouterCostExtractor())
    msg = _stub_response_with_provider_details({"cost": 0.5})

    result = msg.cost()

    # ``upstream_inference_cost`` is missing → first extractor returns
    # None → second extractor (``cost``) wins.
    assert result.total_price == Decimal("0.5")


def test_buggy_extractor_does_not_break_handler() -> None:
    """An extractor that raises is logged + skipped, not re-raised."""

    class Buggy:
        name = "buggy"

        def extract(self, response):  # noqa: ANN001
            raise RuntimeError("kaboom")

    register_cost_extractor(Buggy())
    register_cost_extractor(OpenRouterCostExtractor())
    msg = _stub_response_with_provider_details({"cost": 1.0})

    # Buggy raises → skipped → next extractor handles it.
    assert msg.cost().total_price == Decimal("1.0")


def test_configure_with_none_registers_openrouter_default() -> None:
    """``configure_cost_extractors(None)`` registers the OpenRouter
    extractor as the framework default."""
    configure_cost_extractors(None)
    msg = _stub_response_with_provider_details({"cost": 0.001})

    assert msg.cost().total_price == Decimal("0.001")


def test_configure_with_empty_list_registers_nothing() -> None:
    """Passing ``[]`` explicitly disables the framework default."""
    configure_cost_extractors([])
    msg = _stub_response_with_provider_details({"cost": 0.999})

    with pytest.raises(LookupError):
        msg.cost()


def test_provider_details_extractor_custom_field() -> None:
    """Apps can extend by passing a custom field name."""
    register_cost_extractor(
        ProviderDetailsCostExtractor(field="usd_cost", name="custom"),
    )
    msg = _stub_response_with_provider_details({"usd_cost": 0.42})

    assert msg.cost().total_price == Decimal("0.42")

"""Pluggable cost-extraction fallback for ``ModelResponse.cost()``.

## Problem

``pydantic-ai``'s ``InstrumentedModel`` populates the ``operation.cost``
span attribute by calling ``ModelResponse.cost()``, which delegates to
`genai-prices <https://github.com/pydantic/genai-prices>`_. That library
ships a static catalogue of ``(provider, model)`` pairs; anything not
listed (e.g. ``qwen/qwen3.6-plus`` on OpenRouter at the time of writing)
raises ``LookupError`` and pydantic-ai silently swallows it. Result: no
cost on any span for those agents, even when the upstream provider
returned the real billed cost in the response.

## Design

A ``CostExtractor`` is a strategy that knows how to pull a real billed
cost off a ``ModelResponse`` — typically from ``provider_details`` or a
provider-specific field. The framework ships a default extractor for
``provider_details['cost']`` (OpenRouter + several other adapters
already populate it there). Apps register additional extractors via
``register_cost_extractor(...)`` (typically called once at boot, before
``ObservabilityConfig().install()``).

``install_cost_fallback_patch()`` monkey-patches ``ModelResponse.cost``
once: on ``LookupError`` from the original ``cost()``, it walks the
registered extractors and returns a duck-typed result that satisfies
``InstrumentedModel`` (which only reads ``.total_price``). When
genai-prices later adds an entry for the model, the original lookup
succeeds first and the patch becomes a no-op.

## Why monkey-patch and not a wrapper Model class

``ModelResponse.cost`` is a method on a pydantic model that
``InstrumentedModel`` invokes directly via attribute lookup on the
response object. Subclassing ``ModelResponse`` would require every
adapter (OpenRouter, OpenAI, Anthropic, ...) to instantiate the
subclass — defeats the point. A targeted patch on the method handles
every adapter uniformly.
"""

from __future__ import annotations

import functools
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic_ai_stateflow.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelResponse


_log = get_logger(__name__)


@runtime_checkable
class CostExtractor(Protocol):
    """Strategy for pulling a real billed cost off a ``ModelResponse``.

    ``extract`` returns ``None`` when the strategy doesn't apply to this
    response (so the patch can fall through to the next extractor and
    eventually re-raise the original ``LookupError`` if nothing matches).
    """

    name: str
    """Human-readable extractor id — surfaced in DEBUG logs so misconfig
    is easy to diagnose."""

    def extract(self, response: ModelResponse) -> Decimal | None: ...


class ProviderDetailsCostExtractor:
    """Default extractor: read ``provider_details[field]`` as a Decimal.

    Works out-of-the-box for any pydantic-ai adapter that surfaces a
    real cost on ``ModelResponse.provider_details`` — OpenRouter does
    this for ``cost`` (when the request is sent with
    ``openrouter_usage={'include': True}``) and
    ``upstream_inference_cost``.

    Args:
      field: which ``provider_details`` key to read. Defaults to
        ``"cost"``. Pass a different key (e.g.
        ``"upstream_inference_cost"``) to chain multiple extractors
        that pick up different cost facets.
    """

    def __init__(self, *, field: str = "cost", name: str | None = None) -> None:
        self.field = field
        self.name = name or f"provider_details.{field}"

    def extract(self, response: ModelResponse) -> Decimal | None:
        pd = getattr(response, "provider_details", None) or {}
        raw = pd.get(self.field)
        if raw is None:
            return None
        try:
            return Decimal(str(raw))
        except Exception:  # noqa: BLE001
            return None


class OpenRouterCostExtractor(ProviderDetailsCostExtractor):
    """OpenRouter-aware cost extractor.

    Reads the actual billed cost OpenRouter reports back in
    ``ModelResponse.provider_details['cost']``. **Requires** the request
    to opt in via ``openrouter_usage={'include': True}`` on
    ``OpenRouterModelSettings`` — without it OpenRouter omits cost from
    the response and this extractor returns ``None``.

    Shipped as the framework default for ``configure_cost_extractors``
    so an OpenRouter-backed app sees real cost on Logfire/OTel spans
    out of the box. Pair it with extractors for other providers when
    apps fan out to multiple upstreams.
    """

    def __init__(self) -> None:
        super().__init__(field="cost", name="openrouter")


class OpenRouterUpstreamCostExtractor(ProviderDetailsCostExtractor):
    """Cost paid to the *downstream* provider OpenRouter routed to.

    OpenRouter exposes this in
    ``provider_details['upstream_inference_cost']`` separately from
    the user-billed ``cost``. Useful when reconciling OpenRouter's
    margin against the underlying provider's billing.
    """

    def __init__(self) -> None:
        super().__init__(
            field="upstream_inference_cost",
            name="openrouter.upstream_inference_cost",
        )


# Module-level state — the monkey-patch reads from this list so apps
# can add extractors at any time via ``register_cost_extractor``
# (paired with the ``ObservabilityConfig`` install step at boot).
_extractors: list[CostExtractor] = []
_patched = False


def register_cost_extractor(extractor: CostExtractor) -> None:
    """Append an extractor. Idempotent by ``name``."""
    if any(e.name == extractor.name for e in _extractors):
        return
    _extractors.append(extractor)
    _log.debug("registered cost extractor: %s", extractor.name)


def clear_cost_extractors() -> None:
    """Reset the extractor registry. Test helper."""
    _extractors.clear()


def install_cost_fallback_patch() -> None:
    """Monkey-patch ``ModelResponse.cost`` to fall back to extractors.

    Idempotent. ImportError-safe — if ``pydantic_ai.messages`` isn't
    importable (slim test env without LLM deps), the function is a
    no-op.

    The patched method:
      1. Runs the original ``cost()`` — returns its result on success.
      2. On ``LookupError`` (genai-prices doesn't know the model),
         walks the registered extractors. First non-``None`` wins.
      3. Returns a ``SimpleNamespace`` with ``.total_price`` set —
         enough for ``InstrumentedModel.record_metrics`` and
         pydantic-ai's ``_instrumentation.py`` cost-attr emission,
         neither of which depends on the genai-prices dataclass shape.
      4. Re-raises ``LookupError`` if no extractor matched, preserving
         the original "cost dropped" semantics.
    """
    global _patched  # noqa: PLW0603
    if _patched:
        return

    try:
        from pydantic_ai import messages as _msgs  # noqa: PLC0415
    except ImportError:
        _patched = True
        return

    original = _msgs.ModelResponse.cost

    @functools.wraps(original)
    def cost_with_fallback(self: Any) -> Any:
        try:
            return original(self)
        except LookupError:
            for extractor in _extractors:
                try:
                    value = extractor.extract(self)
                except Exception as exc:  # noqa: BLE001
                    _log.debug(
                        "cost extractor %s raised: %s", extractor.name, exc,
                    )
                    continue
                if value is None:
                    continue
                _log.debug(
                    "cost extracted via %s = %s", extractor.name, value,
                )
                # InstrumentedModel only reads ``.total_price``;
                # SimpleNamespace satisfies the duck-typed contract
                # without depending on genai-prices' internal dataclass.
                return SimpleNamespace(
                    total_price=value,
                    input_price=Decimal(0),
                    output_price=Decimal(0),
                )
            raise

    _msgs.ModelResponse.cost = cost_with_fallback  # type: ignore[method-assign]
    _patched = True
    _log.debug("ModelResponse.cost fallback patch installed")


def configure_cost_extractors(
    extractors: Sequence[CostExtractor] | None,
) -> None:
    """Convenience: install the patch and register extractors.

    Equivalent to calling ``install_cost_fallback_patch()`` followed by
    ``register_cost_extractor(e)`` for each ``e in extractors``. When
    ``extractors`` is ``None``, the default
    (``ProviderDetailsCostExtractor()``) is registered.
    """
    install_cost_fallback_patch()
    resolved: Sequence[CostExtractor] = (
        extractors if extractors is not None
        else (OpenRouterCostExtractor(),)
    )
    for e in resolved:
        register_cost_extractor(e)


__all__ = [
    "CostExtractor",
    "OpenRouterCostExtractor",
    "OpenRouterUpstreamCostExtractor",
    "ProviderDetailsCostExtractor",
    "clear_cost_extractors",
    "configure_cost_extractors",
    "install_cost_fallback_patch",
    "register_cost_extractor",
]

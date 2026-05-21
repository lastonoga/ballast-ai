"""``ObservabilityConfig`` — replacement for the deleted
``ObservabilityProvider`` class.

A plain dataclass holding the knobs that used to be Provider ctor args.
``.install()`` configures Logfire + instrumentation once per process.
Idempotent: calling twice with the same config is a no-op; calling
with a different config raises.

SKELETON — `.install()` raises NotImplementedError. Real impl lands
in SP1 T5 (or earlier if needed).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservabilityConfig:
    """Defaults for framework observability."""

    service_name: str = "pydantic-ai-stateflow"
    environment: str = "dev"
    instrument_pydantic_ai: bool = True
    instrument_httpx: bool = True
    instrument_fastapi: bool = True

    def install(self) -> None:
        """Configure Logfire + instrumentation. Idempotent per process.

        SKELETON: raises NotImplementedError until SP1 T5.
        """
        raise NotImplementedError("ObservabilityConfig.install() lands in SP1 T5")


__all__ = ["ObservabilityConfig"]

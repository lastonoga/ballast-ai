"""Optional ``CoALAUnit`` adapter for ``DriftEngine``.

Apps using the CoALA subsystem may prefer to express a drift sidecar
as a ``CoALAUnit`` and wire it through ``as_capability(unit)`` /
``as_workflow(unit)`` rather than via the dedicated capability /
workflow surfaces. This factory provides that sugar.

This is OPTIONAL — the canonical surfaces (``GoalDriftDetector``
capability, ``with_drift_monitor`` decorator) remain the primary
public API. Most apps will not need this.
"""
from __future__ import annotations

from ballast.coala import CoALABase
from ballast.drift._core import DriftEngine
from ballast.drift._protocols import DriftContext
from ballast.drift._verdict import DriftVerdictBase


class _GoalDriftUnit(CoALABase[
    DriftContext,        # InT  — drift context snapshot
    DriftContext,        # ObsT — identity observation
    DriftVerdictBase,    # ContextT — verdict from judge
    DriftVerdictBase,    # OutT — same verdict as output
]):
    """Wraps ``DriftEngine.maybe_check`` as a 4-phase CoALA unit.

    Phase mapping:
      observe — identity (input IS the drift context)
      retrieve — call engine.maybe_check; handlers fire here as side-effect
      act — return the verdict (no further action; handlers already ran)
      learn — no-op
    """

    def __init__(self, engine: DriftEngine) -> None:
        self._engine = engine

    async def retrieve(self, observation: DriftContext) -> DriftVerdictBase:
        # Synthetic signal for "always check" semantics — caller wired the
        # gating logic into the engine's strategy already.
        from ballast.drift._protocols import DriftCheckSignal
        signal = DriftCheckSignal(
            step_index=1, tool_calls=0, tokens_used=0, seconds_elapsed=0.0,
        )
        verdict = await self._engine.maybe_check(signal, observation)
        # If strategy short-circuited (None), return a synthetic
        # "on-track" verdict so act() has something to return.
        if verdict is None:
            from ballast.drift._verdict import DefaultDriftVerdict
            return DefaultDriftVerdict(
                should_interrupt=False, reason="not checked",
                score=1.0, category="on_track",
            )
        return verdict

    async def act(
        self, observation: DriftContext, context: DriftVerdictBase,
    ) -> DriftVerdictBase:
        return context


def goal_drift_as_unit(engine: DriftEngine) -> _GoalDriftUnit:
    """Wrap a ``DriftEngine`` as a CoALA unit.

    The unit can then be adapted via ``ballast.coala.as_capability(unit)``
    or ``ballast.coala.as_workflow(unit)`` — same engine, different
    runtime wiring.
    """
    return _GoalDriftUnit(engine)


__all__ = ["goal_drift_as_unit"]

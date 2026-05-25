"""Built-in DriftCheckStrategy implementations."""
from __future__ import annotations

from ballast.drift._protocols import DriftCheckSignal
from ballast.drift._strategies import (
    AfterEveryStep,
    Compose,
    EveryNSteps,
    EveryNToolCalls,
    OnBudgetThreshold,
    Periodic,
)


def _sig(step=1, tool_calls=0, tokens=0, secs=0.0) -> DriftCheckSignal:
    return DriftCheckSignal(
        step_index=step, tool_calls=tool_calls,
        tokens_used=tokens, seconds_elapsed=secs,
    )


def test_after_every_step_always_true() -> None:
    s = AfterEveryStep()
    assert s.should_check(_sig(step=1))
    assert s.should_check(_sig(step=10))


def test_every_n_tool_calls_fires_at_threshold() -> None:
    s = EveryNToolCalls(n=3)
    assert not s.should_check(_sig(tool_calls=0))
    assert not s.should_check(_sig(tool_calls=2))
    assert s.should_check(_sig(tool_calls=3))
    assert not s.should_check(_sig(tool_calls=4))
    assert not s.should_check(_sig(tool_calls=5))
    assert s.should_check(_sig(tool_calls=6))


def test_every_n_steps_fires_at_threshold() -> None:
    s = EveryNSteps(n=2)
    assert not s.should_check(_sig(step=1))
    assert s.should_check(_sig(step=2))
    assert not s.should_check(_sig(step=3))
    assert s.should_check(_sig(step=4))


def test_periodic_fires_after_interval() -> None:
    s = Periodic(seconds=10.0)
    assert not s.should_check(_sig(secs=5.0))
    assert s.should_check(_sig(secs=10.0))
    assert not s.should_check(_sig(secs=15.0))
    assert s.should_check(_sig(secs=20.0))


def test_on_budget_threshold_fires_once_above_fraction() -> None:
    # Cannot test without metadata access — strategies don't see metadata.
    # OnBudgetThreshold reads from a budget callable supplied at construct time.
    consumed = {"input": 0, "max": 100}
    def budget_fn() -> tuple[int, int]:
        return consumed["input"], consumed["max"]

    s = OnBudgetThreshold(fraction=0.5, budget_fn=budget_fn)

    consumed["input"] = 49
    assert not s.should_check(_sig())

    consumed["input"] = 60
    assert s.should_check(_sig())

    # Already fired — does not re-fire while still above threshold.
    consumed["input"] = 70
    assert not s.should_check(_sig())


def test_compose_is_or_of_components() -> None:
    fired = []

    class _Once:
        def should_check(self, _sig):
            if not fired:
                fired.append(True)
                return True
            return False

    class _Never:
        def should_check(self, _sig):
            return False

    s = Compose(_Once(), _Never())
    assert s.should_check(_sig())     # _Once fires
    assert not s.should_check(_sig()) # both quiet now

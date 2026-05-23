"""Shared helper-agent spawn validation for HITL flows.

Both ``HITLGate.ask_helper`` and ``DurableHITLWorkflow.open`` validate
the same two invariants before opening a helper thread:

  1. ``helper_agent.metadata_model`` is set — agents that don't declare
     a metadata schema can't be used as HITL helpers (no contract for
     what the helper's tools should read from ``Thread.metadata_``).
  2. ``context`` is an instance of that metadata_model — so the
     thread metadata round-trips back through the helper agent's
     ``build_deps`` validation cleanly.

The downstream thread-creation + opening-message + decision-workflow
shape is intentionally NOT shared — the durable path wraps those side
effects in ``@Durable.step`` for crash-recovery, the gate path does
them inline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ballast.runtime.agents import BallastAgent


def validate_helper_agent(
    helper_agent: type["BallastAgent"], context: "BaseModel",
) -> type:
    """Validate + return the helper agent's ``metadata_model``.

    Raises ``ValueError`` if the agent has no ``metadata_model`` set;
    ``TypeError`` if ``context`` isn't an instance of that model.
    """
    metadata_model = helper_agent.metadata_model
    if metadata_model is None:
        raise ValueError(
            f"{helper_agent.__name__}.metadata_model is None — cannot "
            "use it as a HITL helper agent. Set a metadata_model that "
            "validates the context shape.",
        )
    if not isinstance(context, metadata_model):
        raise TypeError(
            f"context must be an instance of "
            f"{helper_agent.__name__}.metadata_model "
            f"({metadata_model.__name__}), got {type(context).__name__}",
        )
    return metadata_model


__all__ = ["validate_helper_agent"]

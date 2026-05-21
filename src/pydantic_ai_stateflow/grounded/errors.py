from __future__ import annotations

from pydantic_ai_stateflow.errors import StateflowError


class GroundedError(StateflowError):
    """Base for all grounded-schema errors."""

    code = "STATEFLOW_GROUNDED"
    status_code = 500


class GroundedBuildError(GroundedError):
    """Raised at .run() time when the dynamic output model cannot be built
    (e.g., no entity instances in context for a required Ref field)."""

    code = "STATEFLOW_GROUNDED_BUILD"


class GroundedHydrationError(GroundedError):
    """Raised when hydration cannot resolve a Ref via the given repos."""

    code = "STATEFLOW_GROUNDED_HYDRATION"

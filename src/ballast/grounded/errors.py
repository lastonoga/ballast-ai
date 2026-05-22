from __future__ import annotations

from ballast.errors import BallastError


class GroundedError(BallastError):
    """Base for all grounded-schema errors."""

    code = "BALLAST_GROUNDED"
    status_code = 500


class GroundedBuildError(GroundedError):
    """Raised at .run() time when the dynamic output model cannot be built
    (e.g., no entity instances in context for a required Ref field)."""

    code = "BALLAST_GROUNDED_BUILD"


class GroundedHydrationError(GroundedError):
    """Raised when hydration cannot resolve a Ref via the given repos."""

    code = "BALLAST_GROUNDED_HYDRATION"

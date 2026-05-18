class GroundedError(Exception):
    """Base for all grounded-schema errors."""


class GroundedBuildError(GroundedError):
    """Raised at .run() time when the dynamic output model cannot be built
    (e.g., no entity instances in context for a required Ref field)."""


class GroundedHydrationError(GroundedError):
    """Raised when hydration cannot resolve a Ref via the given repos."""

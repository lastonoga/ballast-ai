"""Structured error hierarchy for the framework.

See ``docs/superpowers/specs/2026-05-22-sp2-settings-errors-design.md``
§B for the design. Every framework-raised exception inherits
``BallastError`` with a stable ``code`` and a class-level
``status_code`` used by the HTTP middleware to map to a response.
"""
from __future__ import annotations

import sys
from typing import Any, ClassVar


class BallastError(Exception):
    """Base for every framework-raised error.

    Subclasses override class-level attributes; instance args populate
    ``detail`` / ``hint`` / ``context``.

    Attributes:
      code: stable identifier, ``BALLAST_<DOMAIN>_<SPECIFIC>`` format.
        UPPER_SNAKE; never includes free-form text. Frontends + CLI
        switch on this string.
      status_code: HTTP status used by the error middleware when this
        class escapes a route handler. Class-level default; subclasses
        override.
      detail: human-readable one-liner. Required.
      hint: actionable suggestion. Optional.
      context: machine-readable structured info (workflow id, field
        name, retry count, etc.). Optional; default empty dict.
    """

    code: ClassVar[str] = "BALLAST_UNKNOWN"
    status_code: ClassVar[int] = 500

    def __init__(
        self,
        detail: str,
        *,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.detail = detail
        self.hint = hint
        self.context: dict[str, Any] = dict(context or {})
        super().__init__(detail)

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable representation. Used by middleware and logs."""
        return {
            "code": self.code,
            "detail": self.detail,
            "hint": self.hint,
            "context": self.context,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, detail={self.detail!r})"


# ---------- Configuration domain ----------


class ConfigurationError(BallastError):
    code = "BALLAST_CONFIG"


class SettingsValidationError(ConfigurationError):
    code = "BALLAST_CONFIG_SETTINGS_INVALID"


class MissingDependencyError(ConfigurationError):
    code = "BALLAST_CONFIG_DEPENDENCY_MISSING"


class ConfigurationInvariantViolation(ConfigurationError):
    """Replaces the old ``EngineInvariantViolation`` (deleted by SP1).

    Raised when a one-time bootstrap call (e.g. ``ObservabilityConfig
    .install``) detects an internal invariant has been broken — a sign of
    misconfiguration or accidental double-init."""

    code = "BALLAST_CONFIG_INVARIANT"


# ---------- Persistence domain ----------


class PersistenceError(BallastError):
    code = "BALLAST_PERSISTENCE"


class ThreadNotFound(PersistenceError):
    code = "BALLAST_PERSISTENCE_THREAD_NOT_FOUND"
    status_code = 404

    def __init__(
        self,
        detail: str | None = None,
        *,
        thread_id: str | None = None,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        if thread_id is not None and "thread_id" not in ctx:
            ctx["thread_id"] = str(thread_id)
        if detail is None:
            if thread_id is not None:
                detail = f"thread {thread_id} not found"
            else:
                detail = "thread not found"
        super().__init__(
            detail,
            hint=hint or (
                "Confirm the thread id matches an active thread; check that "
                "it wasn't soft-deleted."
            ),
            context=ctx,
        )
        self.thread_id = thread_id


class ThreadMetadataInvalid(PersistenceError):
    code = "BALLAST_PERSISTENCE_THREAD_METADATA_INVALID"
    status_code = 422


# ---------- Auth domain ----------


class AuthError(BallastError):
    code = "BALLAST_AUTH"
    status_code = 401


class AuthorizationDenied(AuthError):
    code = "BALLAST_AUTH_FORBIDDEN"
    status_code = 403


# ---------- Pattern domain ----------


class PatternError(BallastError):
    """Base for pattern-related errors.

    Catch this to handle any pattern failure regardless of subclass.
    """

    code = "BALLAST_PATTERN"


# ---------- Workflow / agent registry domain ----------


class WorkflowNotFound(BallastError):
    """Auto-generated workflow router could not resolve a kebab name."""

    code = "BALLAST_WORKFLOW_NOT_FOUND"
    status_code = 404


class AgentNotRegistered(BallastError):
    """Streaming / A2A route received an agent name not in the registry."""

    code = "BALLAST_AGENT_NOT_REGISTERED"
    status_code = 404


# ---------- Streaming / messaging domain ----------


class EmptyMessageBody(BallastError):
    """POST /chat received a user message with empty content."""

    code = "BALLAST_STREAMING_EMPTY_MESSAGE"
    status_code = 400


class CancelNotSupported(BallastError):
    """Cancel requested on an agent that is not durable (no workflow id)."""

    code = "BALLAST_STREAMING_CANCEL_NOT_SUPPORTED"
    status_code = 400


# ---------- Formatting helpers ----------


def _format_plain(exc: BallastError) -> str:
    lines = [f"[{exc.code}] {exc.detail}"]
    if exc.hint:
        lines.append(f"  hint: {exc.hint}")
    if exc.context:
        lines.append("  context:")
        for k, v in exc.context.items():
            lines.append(f"    {k}: {v!r}")
    return "\n".join(lines)


def _format_rich(exc: BallastError) -> str:
    from io import StringIO

    from rich.console import Console
    from rich.text import Text

    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        color_system="truecolor",
        width=120,
        highlight=False,
    )
    header = Text()
    header.append("✗ ", style="bold red")
    header.append(exc.code, style="bold red")
    console.print(header)
    console.print(Text(f"  {exc.detail}"))
    if exc.hint:
        console.print()
        hint_line = Text()
        hint_line.append("  hint  ", style="cyan")
        hint_line.append(exc.hint)
        console.print(hint_line)
    if exc.context:
        console.print()
        console.print(Text("  context", style="cyan"))
        key_width = max(len(str(k)) for k in exc.context) if exc.context else 0
        for k, v in exc.context.items():
            row = Text()
            row.append(f"    {str(k).ljust(key_width)}  ", style="dim")
            row.append(repr(v))
            console.print(row)
    return buf.getvalue().rstrip("\n")


def format_error(exc: BallastError, *, color: bool | None = None) -> str:
    """Pretty multi-line representation for stderr/logs.

    Returns ANSI-colored text when ``color=True`` (or auto-detected from
    a tty stderr). Plain text otherwise. Falls back to plain text when
    ``rich`` is unavailable.
    """
    use_color = color if color is not None else sys.stderr.isatty()
    if not use_color:
        return _format_plain(exc)
    try:
        return _format_rich(exc)
    except ImportError:
        return _format_plain(exc)


__all__ = [
    "AgentNotRegistered",
    "AuthError",
    "AuthorizationDenied",
    "CancelNotSupported",
    "ConfigurationError",
    "ConfigurationInvariantViolation",
    "EmptyMessageBody",
    "MissingDependencyError",
    "PatternError",
    "PersistenceError",
    "SettingsValidationError",
    "BallastError",
    "ThreadMetadataInvalid",
    "ThreadNotFound",
    "WorkflowNotFound",
    "format_error",
]

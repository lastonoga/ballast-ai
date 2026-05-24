"""Reflection pattern — writer → critic → refiner durable loop.

Public surface re-exported from focused submodules:

  - :class:`Reflection` — the pattern class
  - :class:`ReflectionExhausted` — raised on no convergence
  - :class:`ReflectionEvent`, :data:`reflection_progress`,
    :func:`default_chat_router` — typed progress signal + UI render
  - :data:`Writer` — type alias for the writer callable signature
  - :func:`to_critic_callable` — adapter from ``LLMJudge`` →
    ``Critique``-returning callable (exposed for advanced custom
    composition)
"""
from ballast.patterns.reflection._critic import (
    CriticCallable,
    to_critic_callable,
)
from ballast.patterns.reflection._errors import ReflectionExhausted
from ballast.patterns.reflection._events import (
    ReflectionEvent,
    default_chat_router,
    reflection_progress,
)
from ballast.patterns.reflection.pattern import Reflection, Writer

__all__ = [
    "CriticCallable",
    "Reflection",
    "ReflectionEvent",
    "ReflectionExhausted",
    "Writer",
    "default_chat_router",
    "reflection_progress",
    "to_critic_callable",
]

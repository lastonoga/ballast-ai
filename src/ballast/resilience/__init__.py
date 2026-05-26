"""Resilience primitives — Circuit Breaker, future Retry/Bulkhead/RateLimiter."""
from ballast.resilience.circuit_breaker import (
    CircuitBreaker, CircuitOpenError,
)

__all__ = ["CircuitBreaker", "CircuitOpenError"]

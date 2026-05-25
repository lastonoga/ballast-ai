"""Authentication context primitives.

Currently only the ``current_user_id`` ContextVar — read by repos and
channels to scope visibility / persist `user_id` stamps. Filled by API
middleware in production; tests use ``acting_as`` directly.
"""
from ballast.auth.context import acting_as, current_user_id

__all__ = ["acting_as", "current_user_id"]

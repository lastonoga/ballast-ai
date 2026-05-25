"""HITL channels — request delivery surfaces.

A channel is the unit of "how a human is asked" — UI card, helper
thread, Slack DM, custom user-written. ``HITLChannel`` is the Protocol;
``DBOSHITLChannel`` is the convenient base for the common case where
verdicts arrive on a DBOS topic.
"""
from ballast.patterns.hitl.channels._base import DBOSHITLChannel
from ballast.patterns.hitl.channels._protocol import (
    HITLChannel,
    InT,
    VerdictT,
)

__all__ = ["DBOSHITLChannel", "HITLChannel", "InT", "VerdictT"]

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
from ballast.patterns.hitl.channels.ui_card import (
    CardVerdict,
    UICardChannel,
    approval_card_decided,
    approval_card_requested,
    card_kind_registry,
    register_card_kind,
)

__all__ = [
    "CardVerdict",
    "DBOSHITLChannel",
    "HITLChannel",
    "InT",
    "UICardChannel",
    "VerdictT",
    "approval_card_decided",
    "approval_card_requested",
    "card_kind_registry",
    "register_card_kind",
]

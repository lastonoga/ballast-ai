"""Human-in-the-loop primitives.

The framework ships a single ``HITLChannel`` Protocol and a
``DBOSHITLChannel`` base. Concrete channels live in
``ballast.patterns.hitl.channels``: ``ThreadChannel`` (helper
sub-thread + agent) and ``UICardChannel`` (out-of-thread approval
card in a side panel).
"""
from ballast.patterns.hitl.channels import (
    CardVerdict,
    DBOSHITLChannel,
    HITLChannel,
    ThreadChannel,
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
    "ThreadChannel",
    "UICardChannel",
    "approval_card_decided",
    "approval_card_requested",
    "card_kind_registry",
    "register_card_kind",
]

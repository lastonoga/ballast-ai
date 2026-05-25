"""Goal Drift Detection — pluggable LLM-judge sidecar for agent runs."""
from ballast.drift._verdict import DefaultDriftVerdict, DriftVerdictBase

__all__ = ["DefaultDriftVerdict", "DriftVerdictBase"]

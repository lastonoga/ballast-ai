"""LLM-as-a-Judge — runtime quality gate.

Public surface re-exported from focused submodules:

  - :class:`LLMJudge`, :func:`set_default_judge_model` — core
  - :class:`JudgeVerdict`, :class:`PairwiseVerdict` — result shapes
  - :class:`JudgeFailed`, :class:`JudgeUnavailable` — exceptions
  - :class:`JudgeAfterRun` — capability wrapper
  - :func:`persist_verdict_as_thread_event` — UI-card persistence

Internal modules (``_models``, ``_errors``, ``_retry``, ``_pairwise``)
underscore-prefixed — apps should reach for the names re-exported
here, not the submodules.
"""
from ballast.capabilities.llm_judge._errors import (
    JudgeFailed,
    JudgeUnavailable,
)
from ballast.capabilities.llm_judge._models import (
    JudgeVerdict,
    PairwiseVerdict,
)
from ballast.capabilities.llm_judge.capability import JudgeAfterRun
from ballast.capabilities.llm_judge.judge import (
    LLMJudge,
    get_default_judge_model_settings,
    set_default_judge_model,
)
from ballast.capabilities.llm_judge.persistence import (
    persist_verdict_as_thread_event,
)

__all__ = [
    "JudgeAfterRun",
    "JudgeFailed",
    "JudgeUnavailable",
    "JudgeVerdict",
    "LLMJudge",
    "PairwiseVerdict",
    "get_default_judge_model_settings",
    "persist_verdict_as_thread_event",
    "set_default_judge_model",
]

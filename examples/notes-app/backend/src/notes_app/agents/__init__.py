"""Pydantic-ai + Stateflow agents for the notes-app.

One sub-module per agent (or per logically-grouped pair, e.g. the two
brainstorm agents). Singletons are constructed at module load.

This package also exposes ``agents`` — a ``Registry[StateflowAgent]``
keyed by ``agent.name`` — so the streaming route resolves
``thread.agent`` → instance via ``agents.get(thread.agent)``. Tests
override individual entries via ``agents.override(mock_agent)``
(returns the previous value so they can restore on teardown).
"""
from __future__ import annotations

from pydantic_ai_stateflow import Registry
from pydantic_ai_stateflow.runtime import StateflowAgent

from notes_app.agents.notes import notes_agent
from notes_app.agents.todo_approval import approval_agent

# App-owned dispatch registry — framework stays agnostic about agents.
agents: Registry[StateflowAgent] = Registry(notes_agent, approval_agent)


__all__ = ["agents"]

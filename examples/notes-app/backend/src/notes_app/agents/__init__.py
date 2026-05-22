"""Pydantic-ai + Stateflow agents for the notes-app.

One sub-module per agent (or per logically-grouped pair, e.g. the two
brainstorm agents). Singletons are constructed at module load so
``main.py`` can import them by name without a registry.
"""

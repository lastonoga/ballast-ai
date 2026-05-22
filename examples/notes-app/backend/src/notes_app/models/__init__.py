"""Pydantic data models for the notes-app.

One sub-module per domain concept (Django-style ``models/<entity>.py``).
Models are pure data — no I/O, no LLM coupling — so storage layers
(``notes_app.repositories``), agents (``notes_app.agents``), and
workflows (``notes_app.workflows``) can all import them without cycles.
"""

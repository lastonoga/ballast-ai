"""Storage layer — repositories that persist app domain models.

One sub-module per aggregate root (Django-style ``repositories/<entity>.py``).
Implementations are kept thin and substitutable so tests can swap an
in-memory impl in for the production one without touching agents,
workflows, or routes.
"""

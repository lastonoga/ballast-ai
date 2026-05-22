"""Durable workflows owned by the notes-app.

One sub-module per workflow class (or per logically-grouped factory).
Singletons live at module load so HTTP routes and agent tools can
spawn workflow runs by direct import.
"""

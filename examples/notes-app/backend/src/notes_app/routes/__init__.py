"""HTTP routes (FastAPI routers) owned by the notes-app.

Each sub-module owns one logical group of endpoints (notes CRUD,
workflow triggers, streaming + cancellation). ``main.py`` mounts
them via ``extra_routers``.
"""

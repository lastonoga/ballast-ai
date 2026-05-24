def test_persistence_public_api():
    """Persistence-layer Protocols are importable from top-level package."""
    from ballast.persistence import (
        EventLogRepository,
        ThreadRepository,
    )

    assert ThreadRepository is not None
    assert EventLogRepository is not None

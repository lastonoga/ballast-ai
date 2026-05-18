def test_persistence_public_api():
    """Persistence-layer Protocols are importable from top-level package."""
    from pydantic_ai_stateflow.persistence import (
        HITLRepository,
        OutboxRepository,
        SqlAlchemyUnitOfWork,
        ThreadRepository,
        UnitOfWork,
    )

    assert UnitOfWork is not None
    assert SqlAlchemyUnitOfWork is not None
    assert ThreadRepository is not None
    assert OutboxRepository is not None
    assert HITLRepository is not None

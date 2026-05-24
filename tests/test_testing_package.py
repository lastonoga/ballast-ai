from ballast.testing import InMemoryThreadRepository


def test_testing_exports_inmemory_repos():
    assert InMemoryThreadRepository is not None

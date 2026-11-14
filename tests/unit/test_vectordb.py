"""QdrantDBProvider regression tests (temporary local Qdrant)."""
import os


def test_delete_then_recreate_collection_is_really_empty(qdrant):
    """Regression: embedded Qdrant on Windows kept the SQLite file of a deleted collection open, so
    recreating a collection with the same name resurrected the old points and do_reset did nothing."""
    qdrant.create_collection("c", embedding_size=4)
    qdrant.insert_many("c", texts=[f"t{i}" for i in range(10)], vectors=[[1, 0, 0, 0]] * 10,
                       record_ids=list(range(10)))
    assert qdrant.client.count("c").count == 10

    qdrant.delete_collection("c")
    assert not qdrant.is_collection_existed("c")
    assert not os.path.isdir(os.path.join(qdrant.db_path, "collection", "c"))

    qdrant.create_collection("c", embedding_size=4)
    assert qdrant.client.count("c").count == 0


def test_create_collection_do_reset_clears_points(qdrant):
    qdrant.create_collection("r", embedding_size=4)
    qdrant.insert_many("r", texts=["a", "b"], vectors=[[1, 0, 0, 0]] * 2, record_ids=[0, 1])
    qdrant.create_collection("r", embedding_size=4, do_reset=True)
    assert qdrant.client.count("r").count == 0


def test_delete_only_removes_its_own_collection(qdrant):
    for name in ("keep", "drop"):
        qdrant.create_collection(name, embedding_size=4)
        qdrant.insert_many(name, texts=["x"], vectors=[[1, 0, 0, 0]], record_ids=[0])
    qdrant.delete_collection("drop")
    assert qdrant.is_collection_existed("keep") and qdrant.client.count("keep").count == 1


def test_delete_missing_collection_is_noop(qdrant):
    assert qdrant.delete_collection("never_created") is None

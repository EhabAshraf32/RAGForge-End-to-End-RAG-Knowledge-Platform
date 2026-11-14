"""Route-level regression tests for B2 / B7 / B8 with fake models (no MongoDB needed)."""
import asyncio
import json
from types import SimpleNamespace

from bson import ObjectId
from langchain_core.documents import Document

import routes.data as data_routes
import routes.nlp as nlp_routes
from controllers.ProcessController import ProcessController
from models.db_schemes import Asset, DataChunk, Project
from routes.schemes.data import ProcessRequest
from routes.schemes.nlp import PushRequest


def make_chunks(n, project_oid):
    return [DataChunk(_id=ObjectId(), chunk_text=f"chunk {i}", chunk_metadata={"page": i % 7}, chunk_order=i + 1,
                      chunk_project_id=project_oid, chunk_asset_id=ObjectId()) for i in range(n)]


def make_request(qdrant, embedding):
    return SimpleNamespace(app=SimpleNamespace(
        db_client=None, vectordb_client=qdrant, generation_client=None,
        embedding_client=embedding, template_parser=None))


def patch_push(monkeypatch, project, chunks):
    class FakeProjectModel:
        async def get_project_or_create_one(self, project_id): return project

    class FakeChunkModel:
        async def get_poject_chunks(self, project_id, page_no=1, page_size=50):
            return chunks[(page_no - 1) * page_size: page_no * page_size]

    async def pm(db_client): return FakeProjectModel()
    async def cm(db_client): return FakeChunkModel()
    monkeypatch.setattr(nlp_routes.ProjectModel, "create_instance", staticmethod(pm))
    monkeypatch.setattr(nlp_routes.ChunkModel, "create_instance", staticmethod(cm))


def test_b2_push_with_reset_keeps_all_pages(monkeypatch, qdrant, fake_embedding):
    project = Project(_id=ObjectId(), project_id="t1")
    req = make_request(qdrant, fake_embedding)

    patch_push(monkeypatch, project, make_chunks(120, project.id))
    r = asyncio.run(nlp_routes.index_project(req, "t1", PushRequest(do_reset=1)))
    assert r.status_code == 200
    assert qdrant.client.count("collection_t1").count == 120   # the bug kept only the last page (20)

    patch_push(monkeypatch, project, make_chunks(30, project.id))
    asyncio.run(nlp_routes.index_project(req, "t1", PushRequest(do_reset=1)))
    assert qdrant.client.count("collection_t1").count == 30    # no stale vectors


def patch_process(monkeypatch, project, file_contents, configs=None):
    configs = configs or {}
    assets = [Asset(_id=ObjectId(), asset_project_id=project.id, asset_type="file", asset_name=fid,
                    asset_config=configs.get(fid)) for fid in file_contents]
    state = {"deleted": False, "inserted": []}

    class FakeProjectModel:
        async def get_project_or_create_one(self, project_id): return project

    class FakeAssetModel:
        async def get_all_project_assets(self, asset_project_id, asset_type): return assets
        async def get_asset_record(self, asset_project_id, asset_name):
            return next((a for a in assets if a.asset_name == asset_name), None)
        async def update_asset_processing(self, asset_id, processing):
            state.setdefault("processing", {})[asset_id] = processing
            return 1

    class FakeChunkModel:
        async def delete_chunks_by_project_id(self, project_id):
            state["deleted"] = True
            return 0
        async def delete_chunks_by_asset_id(self, project_id, asset_id):
            return 0
        async def insert_many_chunks(self, chunks, batch_size=100):
            state["inserted"].extend(chunks)
            return len(chunks)

    class FakeProcessController(ProcessController):
        def __init__(self, project_id): pass
        def get_file_content(self, file_id): return file_contents[file_id]

    async def pm(db_client): return FakeProjectModel()
    async def am(db_client): return FakeAssetModel()
    async def cm(db_client): return FakeChunkModel()
    monkeypatch.setattr(data_routes.ProjectModel, "create_instance", staticmethod(pm))
    monkeypatch.setattr(data_routes.AssetModel, "create_instance", staticmethod(am))
    monkeypatch.setattr(data_routes.ChunkModel, "create_instance", staticmethod(cm))
    monkeypatch.setattr(data_routes, "ProcessController", FakeProcessController)
    return state


def test_b8_empty_file_is_skipped_not_fatal(monkeypatch, qdrant, fake_embedding):
    project = Project(_id=ObjectId(), project_id="t3")
    files = {
        "aaaaaaaaaaaa_scanned.pdf": [Document(page_content="", metadata={"page": 0})],
        "bbbbbbbbbbbb_missing.pdf": None,
        "cccccccccccc_lecture.pdf": [Document(page_content="int main() { return 0; }", metadata={"page": 0})],
    }
    state = patch_process(monkeypatch, project, files,
                          {"cccccccccccc_lecture.pdf": {"original_name": "Lecture 1.pdf", "doc_type": "lecture"}})
    r = asyncio.run(data_routes.process_endpoint(make_request(qdrant, fake_embedding), "t3", ProcessRequest()))
    body = json.loads(r.body)
    assert r.status_code == 200, body
    assert body["processed_files"] == 1 and len(state["inserted"]) == 1
    # Phase 1: asset_config reaches the stored chunk
    md = state["inserted"][0].chunk_metadata
    assert md["document_name"] == "Lecture 1.pdf" and md["doc_type"] == "lecture" and md["page_number"] == 1
    # per-file outcome recorded for the admin dashboard
    outcomes = sorted(p["status"] for p in state["processing"].values())
    assert outcomes == ["failed", "failed", "processed"]


def test_b8_all_files_empty_is_failure(monkeypatch, qdrant, fake_embedding):
    project = Project(_id=ObjectId(), project_id="t4")
    patch_process(monkeypatch, project, {"a.pdf": [Document(page_content="", metadata={})]})
    r = asyncio.run(data_routes.process_endpoint(make_request(qdrant, fake_embedding), "t4", ProcessRequest()))
    assert r.status_code == 400 and json.loads(r.body)["signal"] == "processing_failed"


def test_b7_process_reset_also_resets_vectors(monkeypatch, qdrant, fake_embedding):
    project = Project(_id=ObjectId(), project_id="t5")
    qdrant.create_collection("collection_t5", embedding_size=4)
    qdrant.insert_many("collection_t5", texts=["old"], vectors=[[1, 0, 0, 0]], record_ids=[0])

    files = {"a.pdf": [Document(page_content="int x = 1;", metadata={"page": 0})]}
    state = patch_process(monkeypatch, project, files)

    asyncio.run(data_routes.process_endpoint(make_request(qdrant, fake_embedding), "t5", ProcessRequest(do_reset=0)))
    assert qdrant.is_collection_existed("collection_t5")

    asyncio.run(data_routes.process_endpoint(make_request(qdrant, fake_embedding), "t5", ProcessRequest(do_reset=1)))
    assert state["deleted"] is True
    assert not qdrant.is_collection_existed("collection_t5")


def test_app_imports_and_routes_unchanged():
    import main
    # app.openapi() works across FastAPI versions (newer versions wrap included routers in app.routes)
    assert set(main.app.openapi()["paths"]) >= {
        "/api/v1/", "/api/v1/data/upload/{project_id}", "/api/v1/data/process/{project_id}",
        "/api/v1/nlp/index/push/{project_id}", "/api/v1/nlp/index/search/{project_id}",
        "/api/v1/nlp/index/answer/{project_id}", "/api/v1/nlp/index/info/{project_id}"}

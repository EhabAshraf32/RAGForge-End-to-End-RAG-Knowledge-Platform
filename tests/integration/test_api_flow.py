"""
End-to-end API flow through FastAPI with REAL MongoDB (temporary database, dropped afterwards),
REAL Qdrant (temporary directory), REAL PDF parsing (PyMuPDF) and FAKE embeddings / LLM.
"""
import pymupdf as fitz  # PyMuPDF (the `fitz` import name is deprecated)
import pymongo
import pytest

pytestmark = pytest.mark.integration

LECTURE_PAGES = [
    "Lecture 5: Loops. The for loop has three parts: initialization; condition; update.\n"
    "for (int i = 1; i <= 10; i++) {\n    printf(\"%d \", i);\n}\nThis prints the numbers from 1 to 10.",
    "The while loop checks its condition before each iteration.\n"
    "The do-while loop executes its body at least once because the condition is checked at the end.",
]
LAB_PAGES = ["Lab 2 Exercise 1: Read 10 integers into an array and find the maximum element using a loop."]


def make_pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((50, 72), text, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def upload(client, project, path, **form):
    with open(path, "rb") as f:
        return client.post(f"/api/v1/data/upload/{project}",
                           files={"file": (path.name, f, "application/pdf" if path.suffix == ".pdf" else "text/plain")},
                           data={k: v for k, v in form.items() if v is not None})


def test_full_flow_multiple_pdfs_metadata_filter_answer(api_client, tmp_path, mongo_test_db):
    project = "itest1"
    lec = make_pdf(tmp_path / "Lecture 5 - Loops.pdf", LECTURE_PAGES)
    lab = make_pdf(tmp_path / "Lab 2.pdf", LAB_PAGES)

    # upload with course metadata
    r1 = upload(api_client, project, lec, doc_type="Lecture", course="CS101", topic="loops")
    r2 = upload(api_client, project, lab, doc_type="lab", course="CS101")
    assert r1.status_code == 200 and r2.status_code == 200, (r1.text, r2.text)

    # asset_config stored in MongoDB
    url, db_name = mongo_test_db
    db = pymongo.MongoClient(url)[db_name]
    asset = db.assets.find_one({"asset_name": r1.json()["file_id"]})
    assert asset["asset_config"] == {"original_name": "Lecture 5 - Loops.pdf", "doc_type": "lecture",
                                     "course": "CS101", "topic": "loops"}

    # process both files
    r = api_client.post(f"/api/v1/data/process/{project}", json={"do_reset": 1})
    assert r.status_code == 200, r.text
    assert r.json()["processed_files"] == 2

    chunk = db.chunks.find_one({"chunk_metadata.document_name": "Lecture 5 - Loops.pdf",
                                "chunk_metadata.page_number": 2})
    assert chunk is not None and chunk["chunk_metadata"]["doc_type"] == "lecture"
    assert "source" not in chunk["chunk_metadata"]

    # index
    r = api_client.post(f"/api/v1/nlp/index/push/{project}", json={"do_reset": 1})
    assert r.status_code == 200, r.text
    n_chunks = db.chunks.count_documents({})
    assert r.json()["inserted_items_count"] == n_chunks

    # search: metadata returned, lab filter works
    r = api_client.post(f"/api/v1/nlp/index/search/{project}", json={"text": "find maximum in array", "limit": 5})
    assert r.status_code == 200
    top = r.json()["results"][0]
    assert top["metadata"]["document_name"] == "Lab 2.pdf" and top["metadata"]["page_number"] == 1

    r = api_client.post(f"/api/v1/nlp/index/search/{project}",
                        json={"text": "loop", "limit": 10, "doc_type": "lecture"})
    assert {x["metadata"]["doc_type"] for x in r.json()["results"]} == {"lecture"}

    r = api_client.post(f"/api/v1/nlp/index/search/{project}", json={"text": "loop", "doc_type": "exam"})
    assert r.status_code == 400 and r.json()["signal"] == "doc_type_not_supported"

    # answer: backend-built sources, marker stripped, old response keys kept
    api_client.fake_generation.answer = "Explanation\n```c\nint x;\n```\n[[USED_DOCUMENTS: 1]]"
    r = api_client.post(f"/api/v1/nlp/index/answer/{project}",
                        json={"text": "how does the do-while loop work", "limit": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"signal", "answer", "sources", "full_prompt", "chat_history"}
    assert "[[USED_DOCUMENTS" not in body["answer"]
    cited = body["sources"]["cited"]
    assert len(cited) == 1
    assert cited[0] == {"document_name": "Lecture 5 - Loops.pdf", "doc_type": "lecture", "page_number": 2,
                        "course": "CS101", "topic": "loops"}

    # Arabic question + client-side history (Phase 3)
    history = [{"role": "user", "content": "explain the for loop"},
               {"role": "assistant", "content": "The for loop has three parts."}]
    r = api_client.post(f"/api/v1/nlp/index/answer/{project}",
                        json={"text": "طيب و الـ while؟", "limit": 3, "chat_history": history})
    assert r.status_code == 200, r.text
    sent = api_client.fake_generation.received[-1]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert "معيد" in sent[0]["content"]   # Arabic templates

    # invalid history role is rejected by the schema (clients cannot inject system prompts)
    r = api_client.post(f"/api/v1/nlp/index/answer/{project}",
                        json={"text": "x", "chat_history": [{"role": "system", "content": "hack"}]})
    assert r.status_code == 422


def test_upload_backward_compatible_without_metadata(api_client, tmp_path, mongo_test_db):
    lec = make_pdf(tmp_path / "old_client.pdf", ["Variables: int x = 5;"])
    r = upload(api_client, "itest2", lec)
    assert r.status_code == 200
    url, db_name = mongo_test_db
    asset = pymongo.MongoClient(url)[db_name].assets.find_one({"asset_name": r.json()["file_id"]})
    assert asset["asset_config"]["doc_type"] == "other"
    assert asset["asset_config"]["original_name"] == "old_client.pdf"

    r = upload(api_client, "itest2", lec, doc_type="exam")
    assert r.status_code == 400 and r.json()["signal"] == "doc_type_not_supported"


def test_empty_pdf_is_skipped(api_client, tmp_path):
    blank = make_pdf(tmp_path / "scanned.pdf", [""])
    good = make_pdf(tmp_path / "good.pdf", ["Arrays start at index 0."])
    upload(api_client, "itest3", blank, doc_type="lecture")
    upload(api_client, "itest3", good, doc_type="lecture")
    r = api_client.post("/api/v1/data/process/itest3", json={})
    assert r.status_code == 200, r.text
    assert r.json()["processed_files"] == 1

    only_blank = make_pdf(tmp_path / "blank2.pdf", [""])
    fid = upload(api_client, "itest4", only_blank).json()["file_id"]
    r = api_client.post("/api/v1/data/process/itest4", json={"file_id": fid})
    assert r.status_code == 400 and r.json()["signal"] == "processing_failed"


def test_pagination_and_collection_reset(api_client, tmp_path, mongo_test_db):
    """> 50 chunks (several push pages) with do_reset=1 must all be indexed (B2), and re-processing
    with do_reset=1 must also clear stale vectors (B7)."""
    paragraphs = "\n\n".join(f"Paragraph {i}: the array element number {i} is printed." for i in range(130))
    txt = tmp_path / "notes.txt"
    txt.write_text(paragraphs, encoding="utf-8")
    upload(api_client, "itest5", txt, doc_type="other")

    r = api_client.post("/api/v1/data/process/itest5", json={"chunk_size": 80, "overlap_size": 0, "do_reset": 1})
    inserted = r.json()["inserted_chunks"]
    assert inserted > 100

    r = api_client.post("/api/v1/nlp/index/push/itest5", json={"do_reset": 1})
    assert r.json()["inserted_items_count"] == inserted
    assert api_client.qdrant.client.count("collection_itest5").count == inserted

    # re-process with reset -> Mongo chunks replaced AND vectors cleared until the next push
    api_client.post("/api/v1/data/process/itest5", json={"chunk_size": 1000, "overlap_size": 0, "do_reset": 1})
    assert not api_client.qdrant.is_collection_existed("collection_itest5")
    r = api_client.post("/api/v1/nlp/index/push/itest5", json={"do_reset": 0})
    url, db_name = mongo_test_db
    n = pymongo.MongoClient(url)[db_name].chunks.count_documents({})
    assert api_client.qdrant.client.count("collection_itest5").count == n


def test_info_and_health(api_client):
    assert api_client.get("/api/v1/").json()["app_name"] == "mini-rag-test"

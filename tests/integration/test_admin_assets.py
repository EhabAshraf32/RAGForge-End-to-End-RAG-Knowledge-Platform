"""Admin course-material management: listing with real status, re-processing, safe deletion.
Real MongoDB (temp DB) + temp Qdrant + temp files dir + fake embeddings."""
import os

import pymongo
import pytest

from integration.test_api_flow import make_pdf, upload

pytestmark = pytest.mark.integration
P = "adm1"


def assets(client, project=P):
    r = client.get(f"/api/v1/data/assets/{project}")
    assert r.status_code == 200, r.text
    return r.json()


def by_name(listing):
    return {a["document_name"]: a for a in listing["assets"]}


def test_status_lifecycle_and_summary(api_client, tmp_path):
    lec = make_pdf(tmp_path / "Lecture 1.pdf", ["Variables and data types: int, float, char.", "Operators."])
    lab = make_pdf(tmp_path / "Lab 1.pdf", ["Lab exercise: sum of even numbers."])
    blank = make_pdf(tmp_path / "Scanned.pdf", [""])
    for path, t in [(lec, "lecture"), (lab, "lab"), (blank, "lecture")]:
        assert upload(api_client, P, path, doc_type=t, course="CS101", topic="basics").status_code == 200

    listing = assets(api_client)
    s = listing["summary"]
    assert (s["total_documents"], s["lectures"], s["labs"]) == (3, 2, 1)
    assert s["awaiting_processing"] == 3 and s["indexed_documents"] == 0
    a = by_name(listing)["Lecture 1.pdf"]
    assert a["uploaded_at"] and a["course"] == "CS101" and a["topic"] == "basics"
    assert a["processing_status"] == "pending" and a["indexing_status"] == "not_indexed"

    api_client.post(f"/api/v1/data/process/{P}", json={})
    listing = assets(api_client)
    docs = by_name(listing)
    assert docs["Lecture 1.pdf"]["processing_status"] == "processed" and docs["Lecture 1.pdf"]["chunks"] == 2
    assert docs["Scanned.pdf"]["processing_status"] == "failed" and "text" in docs["Scanned.pdf"]["processing_error"]
    assert listing["summary"]["processing_errors"] == 1 and listing["summary"]["awaiting_indexing"] == 2

    api_client.post(f"/api/v1/nlp/index/push/{P}", json={"do_reset": 1})
    listing = assets(api_client)
    docs = by_name(listing)
    assert docs["Lecture 1.pdf"]["indexing_status"] == "indexed" and docs["Lab 1.pdf"]["indexed_points"] == 1
    assert listing["summary"]["indexed_documents"] == 2 and listing["summary"]["index_needs_rebuild"] is False


def test_reprocessing_a_file_replaces_its_chunks(api_client, tmp_path, mongo_test_db):
    lec = make_pdf(tmp_path / "L.pdf", ["Loops page one.", "Loops page two."])
    fid = upload(api_client, P, lec, doc_type="lecture").json()["file_id"]
    for _ in range(3):
        assert api_client.post(f"/api/v1/data/process/{P}", json={"file_id": fid}).status_code == 200
    url, db = mongo_test_db
    assert pymongo.MongoClient(url)[db].chunks.count_documents({}) == 2   # not 6


def test_delete_removes_only_that_document(api_client, tmp_path, isolated_files, mongo_test_db):
    keep = make_pdf(tmp_path / "Keep.pdf", ["Arrays start at index zero."])
    drop = make_pdf(tmp_path / "Drop.pdf", ["Pointers store addresses.", "Pointer arithmetic."])
    keep_id = upload(api_client, P, keep, doc_type="lecture").json()["file_id"]
    drop_id = upload(api_client, P, drop, doc_type="lab").json()["file_id"]
    # an unrelated project that must stay intact
    other = make_pdf(tmp_path / "Other.pdf", ["Other course."])
    upload(api_client, "otherproj", other, doc_type="lecture")
    for proj in (P, "otherproj"):
        api_client.post(f"/api/v1/data/process/{proj}", json={})
        api_client.post(f"/api/v1/nlp/index/push/{proj}", json={"do_reset": 1})

    r = api_client.delete(f"/api/v1/data/asset/{P}/{drop_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted_chunks"] == 2 and body["deleted_vectors"] == 2 and body["file_removed"]

    listing = assets(api_client)
    assert [a["file_id"] for a in listing["assets"]] == [keep_id]
    assert by_name(listing)["Keep.pdf"]["indexing_status"] == "indexed"
    assert not os.path.exists(isolated_files / P / drop_id)
    assert os.path.exists(isolated_files / P / keep_id)
    # other project untouched
    assert assets(api_client, "otherproj")["summary"]["indexed_documents"] == 1
    assert api_client.qdrant.client.count(f"collection_{P}").count == 1

    # deleting again -> 404, nothing else changes
    assert api_client.delete(f"/api/v1/data/asset/{P}/{drop_id}").status_code == 404
    assert api_client.delete(f"/api/v1/data/asset/{P}/doesnotexist.pdf").status_code == 404
    assert len(assets(api_client)["assets"]) == 1


def test_delete_before_indexing(api_client, tmp_path):
    pdf = make_pdf(tmp_path / "Fresh.pdf", ["x"])
    fid = upload(api_client, P, pdf, doc_type="other").json()["file_id"]
    r = api_client.delete(f"/api/v1/data/asset/{P}/{fid}")
    assert r.status_code == 200 and r.json()["deleted_vectors"] == 0
    assert assets(api_client)["summary"]["total_documents"] == 0

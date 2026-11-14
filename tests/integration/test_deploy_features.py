"""Hosted-deployment features with REAL MongoDB (temp DB): GridFS file persistence across a wiped disk,
server-side upload validation, and per-user rate-limit keys for trusted server-side frontends."""
import os
import shutil

import pymongo
import pytest

from conftest import TEST_STUDENT_CODE, pdf_bytes

pytestmark = pytest.mark.integration
P = "dep1"


def upload_bytes(client, name, data, **form):
    return client.post(f"/api/v1/data/upload/{P}", files={"file": (name, data, "application/pdf")},
                       data={"doc_type": "lecture", **form})


def test_files_survive_a_wiped_server_disk(api_client, isolated_files, mongo_test_db, monkeypatch):
    monkeypatch.setenv("STORE_FILES_IN_DB", "true")
    r = upload_bytes(api_client, "Lecture 1.pdf", pdf_bytes(["Variables store values.", "int x = 5;"]))
    assert r.status_code == 200, r.text
    fid = r.json()["file_id"]
    url, db = mongo_test_db
    files = pymongo.MongoClient(url)[db]["asset_files.files"]
    assert files.count_documents({"filename": f"{P}/{fid}"}) == 1

    # simulate a restart on an ephemeral disk (Render free): the local upload folder is gone
    shutil.rmtree(isolated_files / P)
    assert not (isolated_files / P / fid).exists()

    r = api_client.post(f"/api/v1/data/process/{P}", json={})
    assert r.status_code == 200 and r.json()["inserted_chunks"] == 2, r.text
    assert (isolated_files / P / fid).exists()          # restored from GridFS

    r = api_client.delete(f"/api/v1/data/asset/{P}/{fid}")
    assert r.status_code == 200
    assert files.count_documents({}) == 0               # database copy removed too


def test_without_db_store_nothing_goes_to_gridfs(api_client, mongo_test_db):
    assert upload_bytes(api_client, "x.pdf", pdf_bytes(["x"])).status_code == 200
    url, db = mongo_test_db
    assert "asset_files.files" not in pymongo.MongoClient(url)[db].list_collection_names()


def test_upload_rejects_fake_pdf_and_oversize(api_client, isolated_files, monkeypatch):
    r = upload_bytes(api_client, "fake.pdf", b"MZ\x90\x00 not a pdf")
    assert r.status_code == 400 and r.json()["signal"] == "file_type_not_supported"
    monkeypatch.setenv("FILE_MAX_SIZE", "1")
    r = upload_bytes(api_client, "big.pdf", b"%PDF-1.7\n" + b"0" * (2 * 1024 * 1024))
    assert r.status_code == 400 and r.json()["signal"] == "file_size_exceeded"
    # rejected files are not left on disk
    assert not any((isolated_files / P).glob("*")) if (isolated_files / P).exists() else True


def test_trusted_frontend_ip_gives_each_user_own_login_budget(api_client, monkeypatch):
    api_client.headers.pop("Authorization", None)
    monkeypatch.setenv("LOGIN_RATE_LIMIT", "2")
    monkeypatch.setenv("TRUSTED_FRONTEND_SECRET", "frontend-secret-xyz")

    def login(ip=None, secret="frontend-secret-xyz"):
        headers = {"X-Client-IP": ip, "X-Frontend-Secret": secret} if ip else {}
        return api_client.post("/api/v1/auth/student/login", json={"access_code": "wrong"}, headers=headers).status_code

    # two different students behind the same hosted frontend do not block each other
    assert [login("10.0.0.1"), login("10.0.0.1"), login("10.0.0.1")] == [401, 401, 429]
    assert login("10.0.0.2") == 401
    # a forged header without the right secret falls back to the real connection address
    assert login("10.0.0.3", secret="guess") == 401
    assert login("10.0.0.4", secret="guess") == 401
    assert login("10.0.0.5", secret="guess") == 429
    # correct credentials still work for a fresh user
    r = api_client.post("/api/v1/auth/student/login", json={"access_code": TEST_STUDENT_CODE},
                        headers={"X-Client-IP": "10.0.0.9", "X-Frontend-Secret": "frontend-secret-xyz"})
    assert r.status_code == 200

"""
Admin dashboard (streamlit_app/admin_app.py) driven with Streamlit AppTest against a REAL uvicorn backend
(temp MongoDB DB, temp Qdrant, fake embeddings / LLM).
"""
import os

import pytest

pytest.importorskip("streamlit", minversion="1.67", reason="frontend tests need streamlit>=1.67 (streamlit_app/requirements.txt)")
from streamlit.testing.v1 import AppTest

from conftest import STREAMLIT_DIR, TEST_ADMIN_PASSWORD, TEST_ADMIN_USERNAME, TEST_STUDENT_CODE, pdf_bytes
from conftest import app_texts as texts

pytestmark = pytest.mark.integration
ADMIN_APP = os.path.join(STREAMLIT_DIR, "admin_app.py")
PROJECT = "itest1"
TIMEOUT = 60


def logged_in_admin(monkeypatch):
    import settings
    monkeypatch.setattr(settings, "ADMIN_DEFAULT_PROJECT_ID", PROJECT)
    at = AppTest.from_file(ADMIN_APP, default_timeout=TIMEOUT)
    at.run()
    at.text_input[0].input(TEST_ADMIN_USERNAME)
    at.text_input[1].input(TEST_ADMIN_PASSWORD)
    at.button[0].click().run()
    assert not at.exception, at.exception
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_admin_login_required_and_wrong_password(live_backend):
    at = AppTest.from_file(ADMIN_APP, default_timeout=TIMEOUT).run()
    assert at.title[0].value.endswith("Course Admin")
    assert len(at.text_input) == 2 and not at.metric            # login form, no dashboard
    at.text_input[0].input(TEST_ADMIN_USERNAME)
    at.text_input[1].input("wrong-password")
    at.button[0].click().run()
    assert "Wrong username, password or access code" in texts(at)
    assert not at.metric


def test_student_token_is_rejected_by_admin_dashboard(live_backend):
    import requests
    token = requests.post(f"{live_backend.base_url}/auth/student/login",
                          json={"access_code": TEST_STUDENT_CODE}).json()["access_token"]
    at = AppTest.from_file(ADMIN_APP, default_timeout=TIMEOUT)
    at.session_state["admin_token"] = token     # e.g. a student pasting their token
    at.run()
    assert "not an admin account" in texts(at)
    assert not at.metric


def test_full_admin_workflow(live_backend, monkeypatch):
    at = logged_in_admin(monkeypatch)
    assert metric(at, "Documents") == "0"

    # --- upload two PDFs with metadata (multiple files)
    lecture = pdf_bytes(["Lecture 4: the switch statement selects one of many cases.", "break exits the switch."])
    lab = pdf_bytes(["Lab 3: write a calculator using switch."])
    at.file_uploader[0].set_value([("lec4.pdf", lecture, "application/pdf"),
                                   ("lab3.pdf", lab, "application/pdf")]).run()
    # per-file display names
    name_inputs = [t for t in at.text_input if t.label.startswith("Name for file")]
    assert [t.value for t in name_inputs] == ["lec4.pdf", "lab3.pdf"]
    name_inputs[0].input("Lecture 4 - Switch.pdf")
    at.radio[0].set_value("Lecture")
    next(t for t in at.text_input if t.label == "Course").input("CS101")
    next(t for t in at.text_input if t.label == "Topic").input("switch")
    next(b for b in at.button if b.label == "Upload").click().run()
    out = texts(at)
    assert "Uploaded: Lecture 4 - Switch.pdf" in out and "Uploaded: lab3.pdf" in out, out
    assert metric(at, "Documents") == "2" and metric(at, "Lectures") == "2"
    assert metric(at, "Awaiting processing") == "2"

    # --- process (backend extracts text + chunks)
    next(b for b in at.button if b.label == "Process documents").click().run()
    out = texts(at)
    assert "Lecture 4 - Switch.pdf: 2 chunks" in out and "lab3.pdf: 1 chunks" in out, out
    assert metric(at, "Awaiting processing") == "0" and metric(at, "Awaiting indexing") == "2"

    # --- index (backend embeds + stores vectors)
    next(b for b in at.button if b.label == "Build search index").click().run()
    assert "Search index rebuilt by the backend: 3 chunks indexed." in texts(at)
    assert metric(at, "Indexed") == "2"

    # --- documents table shows real status + metadata
    df = at.dataframe[0].value
    row = df[df["Document"] == "Lecture 4 - Switch.pdf"].iloc[0]
    assert row["Type"] == "Lecture" and row["Course"] == "CS101" and row["Topic"] == "switch"
    assert "processed" in row["Processing"] and "indexed" in row["Indexing"] and row["Uploaded"]

    # --- delete one document (needs confirmation)
    sel = at.selectbox[0]
    target = next(o for o in sel.options if o.startswith("lab3.pdf"))
    sel.set_value(target).run()
    delete_btn = next(b for b in at.button if b.label == "Delete document")
    assert delete_btn.disabled
    at.checkbox[0].check().run()
    next(b for b in at.button if b.label == "Delete document").click().run()
    assert "Deleted lab3.pdf: 1 chunks, 1 vectors removed." in texts(at)
    assert metric(at, "Documents") == "1"
    assert list(at.dataframe[0].value["Document"]) == ["Lecture 4 - Switch.pdf"]


def test_upload_validation_rejects_non_pdf_and_oversize(live_backend, monkeypatch):
    import settings
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)
    at = logged_in_admin(monkeypatch)
    at.file_uploader[0].set_value([("fake.pdf", b"not a pdf at all", "application/pdf"),
                                   ("big.pdf", b"%PDF-" + b"0" * (2 * 1024 * 1024), "application/pdf")]).run()
    next(b for b in at.button if b.label == "Upload").click().run()
    out = texts(at)
    assert "Not uploaded: fake.pdf - not a valid PDF file" in out
    assert "Not uploaded: big.pdf - larger than 1 MB" in out
    assert metric(at, "Documents") == "0"


def test_processing_error_is_reported(live_backend, monkeypatch):
    at = logged_in_admin(monkeypatch)
    at.file_uploader[0].set_value([("scan.pdf", pdf_bytes([""]), "application/pdf")]).run()
    next(b for b in at.button if b.label == "Upload").click().run()
    next(b for b in at.button if b.label == "Process documents").click().run()
    assert "scan.pdf: No text could be extracted" in texts(at)
    assert metric(at, "Processing errors") == "1"
    assert "no extractable text" in at.dataframe[0].value.iloc[0]["Error"]


def test_expired_admin_session_returns_to_login(live_backend):
    from conftest import make_token
    at = AppTest.from_file(ADMIN_APP, default_timeout=TIMEOUT)
    at.session_state["admin_token"] = make_token("admin", expires_minutes=-1)
    at.run()
    assert "Your session expired" in texts(at)
    assert not at.metric


def test_validate_pdf_unit():
    import sys
    sys.path.insert(0, STREAMLIT_DIR)
    from admin_actions import validate_pdf
    assert validate_pdf("a.pdf", b"%PDF-1.7 ...", 10) is None
    assert validate_pdf("a.txt", b"%PDF-", 10) == "only PDF files are accepted"
    assert validate_pdf("a.pdf", b"", 10) == "the file is empty"
    assert validate_pdf("a.pdf", b"PK\x03\x04", 10) == "not a valid PDF file"

"""
Student chat (streamlit_app/student_app.py) driven with Streamlit AppTest against a REAL uvicorn backend
(temp MongoDB DB, temp Qdrant, fake embeddings / LLM).
"""
import os

import pytest

pytest.importorskip("streamlit", minversion="1.67", reason="frontend tests need streamlit>=1.67 (streamlit_app/requirements.txt)")
import requests
from streamlit.testing.v1 import AppTest

from conftest import STREAMLIT_DIR, TEST_STUDENT_CODE, TEST_STUDENT_PROJECT, auth_headers, make_token
from conftest import app_texts as texts, pdf_bytes

pytestmark = pytest.mark.integration
STUDENT_APP = os.path.join(STREAMLIT_DIR, "student_app.py")
TIMEOUT = 60

ANSWER = ("A **for loop** repeats a block.\n\n```c\nfor (int i = 1; i <= 3; i++)\n    printf(\"%d \", i);\n```\n\n"
          "Output:\n\n```text\n1 2 3\n```\n[[USED_DOCUMENTS: 1]]")


@pytest.fixture
def course(live_backend):
    """The TA has uploaded, processed and indexed one lecture (via the real admin API)."""
    admin = auth_headers(make_token("admin"))
    base = live_backend.base_url
    r = requests.post(f"{base}/data/upload/{TEST_STUDENT_PROJECT}", headers=admin,
                      files={"file": ("Lecture 5 - Loops.pdf",
                                      pdf_bytes(["Lecture 5: the for loop repeats a block a known number of times."]),
                                      "application/pdf")},
                      data={"doc_type": "lecture", "course": "CS101", "topic": "loops"})
    assert r.status_code == 200, r.text
    assert requests.post(f"{base}/data/process/{TEST_STUDENT_PROJECT}", headers=admin, json={}).status_code == 200
    assert requests.post(f"{base}/nlp/index/push/{TEST_STUDENT_PROJECT}", headers=admin,
                         json={"do_reset": 1}).status_code == 200
    live_backend.fake_generation.answer = ANSWER
    return live_backend


def logged_in_student(name="Mona"):
    at = AppTest.from_file(STUDENT_APP, default_timeout=TIMEOUT).run()
    at.text_input[0].input(TEST_STUDENT_CODE)
    at.text_input[1].input(name)
    at.button[0].click().run()
    assert not at.exception, at.exception
    return at


def chat_markdown(at):
    return [m.value for msg in at.chat_message for m in msg.markdown]


def test_login_wrong_code_and_success(live_backend):
    at = AppTest.from_file(STUDENT_APP, default_timeout=TIMEOUT).run()
    assert not at.chat_input
    at.text_input[0].input("wrong")
    at.button[0].click().run()
    assert "Wrong username, password or access code" in texts(at)
    at = logged_in_student()
    assert at.chat_input and "Mona" in texts(at)
    assert "admin_token" not in at.session_state  # students never hold admin credentials


def test_english_question_code_output_and_sources(course):
    at = logged_in_student()
    at.chat_input[0].set_value("How does a for loop work?").run()
    assert not at.exception, at.exception
    md = chat_markdown(at)
    assert md[0] == "How does a for loop work?"
    answer = md[1]
    assert "```c\nfor (int i = 1; i <= 3; i++)" in answer and "```text\n1 2 3\n```" in answer
    assert "[[USED_DOCUMENTS" not in answer
    # sources rendered from backend metadata only
    sources = at.chat_message[1].expander[0]
    assert sources.label.startswith("📚 Sources from the course material (1)")
    src_text = sources.markdown[0].value
    assert src_text == "- **Lecture 5 - Loops.pdf** · Lecture · page 1 · CS101 · topic: loops"


def test_arabic_question_and_follow_up_history(course):
    at = logged_in_student()
    at.chat_input[0].set_value("اشرح الـ for loop").run()
    sent = course.fake_generation.received[-1]
    assert "معيد" in sent[0]["content"]                    # backend used the Arabic templates
    assert chat_markdown(at)[0] == "اشرح الـ for loop"     # rendered as typed

    at.chat_input[0].set_value("طيب وايه الفرق بينها وبين while؟").run()
    sent = course.fake_generation.received[-1]
    roles = [m["role"] for m in sent]
    assert roles == ["system", "user", "assistant", "user"]   # previous turn sent as chat_history
    assert sent[1]["content"] == "اشرح الـ for loop"
    assert len(at.chat_message) == 4


def test_no_cited_sources_message(course):
    course.fake_generation.answer = "This specific information is not available in the provided course material.\n" \
                                    "[[USED_DOCUMENTS: none]]"
    at = logged_in_student()
    at.chat_input[0].set_value("How does malloc work?").run()
    assert "No course material was cited for this answer." in texts(at)
    assert not at.chat_message[1].expander


def test_new_conversation_clears_history(course):
    at = logged_in_student()
    at.chat_input[0].set_value("for loop?").run()
    assert len(at.chat_message) == 2
    next(b for b in at.button if "New conversation" in b.label).click().run()
    assert len(at.chat_message) == 0 and at.session_state["messages"] == []


def test_example_prompt_button(course):
    at = logged_in_student()
    next(b for b in at.button if b.label == "🔁 Loops").click().run()
    assert chat_markdown(at)[0].startswith("What is the difference between while and do-while")


def test_rate_limit_error_is_shown_and_history_kept_consistent(course, monkeypatch):
    monkeypatch.setenv("STUDENT_ANSWER_RATE_LIMIT_PER_MINUTE", "1")
    at = logged_in_student()
    at.chat_input[0].set_value("first").run()
    at.chat_input[0].set_value("second").run()
    assert "Too many requests" in texts(at)
    assert [m["content"] for m in at.session_state["messages"]][-1] != "second"  # unanswered question dropped
    assert len(at.session_state["messages"]) == 2


def test_backend_unreachable(course, monkeypatch):
    at = logged_in_student()
    import settings
    monkeypatch.setattr(settings, "API_BASE_URL", "http://127.0.0.1:9/api/v1")
    at.chat_input[0].set_value("anything").run()
    assert "not reachable" in texts(at)


def test_expired_student_session_returns_to_login(live_backend):
    at = AppTest.from_file(STUDENT_APP, default_timeout=TIMEOUT)
    at.session_state["student_token"] = make_token("student", expires_minutes=-1)
    at.run()
    assert "Your session expired" in texts(at)
    assert not at.chat_input


def test_admin_token_cannot_be_used_as_student_session(live_backend):
    at = AppTest.from_file(STUDENT_APP, default_timeout=TIMEOUT)
    at.session_state["student_token"] = make_token("admin")
    at.run()
    assert "Please use the student access code" in texts(at)


def test_student_token_from_app_cannot_call_admin_endpoints(course):
    at = logged_in_student()
    token = at.session_state["student_token"]
    base = course.base_url
    h = auth_headers(token)
    assert requests.get(f"{base}/data/assets/{TEST_STUDENT_PROJECT}", headers=h).status_code == 403
    assert requests.post(f"{base}/nlp/index/push/{TEST_STUDENT_PROJECT}", headers=h, json={}).status_code == 403
    assert requests.post(f"{base}/data/upload/{TEST_STUDENT_PROJECT}", headers=h,
                         files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}).status_code == 403

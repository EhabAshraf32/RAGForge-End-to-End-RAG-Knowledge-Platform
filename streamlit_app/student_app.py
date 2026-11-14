"""
Student chat - an online TA support session for the Structured Programming course (currently taught with C).

    streamlit run streamlit_app/student_app.py --server.port 8501

Answers, grounding and sources come from the backend (/nlp/index/answer). The conversation is kept
client-side in st.session_state and sent as chat_history; nothing is stored on the server.
"""
import streamlit as st

import settings
from api_client import ApiError, BackendClient, friendly
from identity import backend

st.set_page_config(page_title="Structured Programming TA", page_icon="💬", layout="centered")

# Arabic / English in the same chat: every paragraph takes its direction from its own text,
# code blocks always stay left-to-right.
st.markdown("""
<style>
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] h1, [data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3, [data-testid="stChatMessageContent"] td,
[data-testid="stChatMessageContent"] th { unicode-bidi: plaintext; text-align: start; }
[data-testid="stChatMessageContent"] pre, [data-testid="stChatMessageContent"] code {
  direction: ltr; text-align: left; unicode-bidi: isolate; }
</style>
""", unsafe_allow_html=True)

HISTORY_TURNS_SENT = 6  # the backend also caps this (RAG_HISTORY_MAX_MESSAGES)
SCOPES = {"All course material": None, "Lectures only": "lecture", "Labs only": "lab"}
EXAMPLES = [
    ("🧭 Algorithm", "Write an algorithm and describe its flowchart to find the largest of three numbers."),
    ("🔁 Loops", "What is the difference between while and do-while? Show an example."),
    ("🧮 Output", "What is the output of this code?\n```c\nint x = 7 / 2;\nprintf(\"%d\", x);\n```"),
    ("🇪🇬 عربي", "اشرح الـ if else مع مثال"),
]
TYPE_LABELS = {"lecture": "Lecture", "lab": "Lab", "other": "Other"}


# ---------------------------------------------------------------------------- session / auth
def logout(message: str = None):
    for key in ("student_token", "student", "messages"):
        st.session_state.pop(key, None)
    if message:
        st.session_state["login_message"] = message


def client():
    return backend(st.session_state.get("student_token"))


def ensure_student_session():
    """Validate the stored token with the backend on every run."""
    if "student_token" not in st.session_state:
        return False
    try:
        me = client().me()
    except ApiError as e:
        logout(friendly(e))
        return False
    except Exception as e:
        st.error(friendly(e))
        st.stop()
    if me.get("role") != "student" or not me.get("project_id"):
        logout("Please use the student access code.")
        return False
    st.session_state["student"] = me
    return True


def login_page():
    st.title("💬 Structured Programming TA")
    st.caption(f"{settings.COURSE_TITLE} - online support session")
    if st.session_state.get("login_message"):
        st.warning(st.session_state.pop("login_message"))
    st.markdown("Ask about the course in **English or Arabic**: algorithms, flowcharts, concepts, code tracing, "
                "expected output, debugging and the lab exercises.")
    with st.form("student_login"):
        code = st.text_input("Course access code", type="password", help="Your TA shares this code.")
        name = st.text_input("Your first name (optional)", max_chars=40)
        submitted = st.form_submit_button("Start session", type="primary")
    if submitted:
        try:
            body = backend().student_login(code, name)
            st.session_state["student_token"] = body["access_token"]
            st.session_state["messages"] = []
            st.rerun()
        except Exception as e:
            st.error(friendly(e))


# ---------------------------------------------------------------------------- rendering
def format_source(src: dict) -> str:
    """Only metadata actually returned by the backend is shown."""
    parts = [f"**{src.get('document_name') or 'Unknown document'}**"]
    if src.get("doc_type"):
        parts.append(TYPE_LABELS.get(src["doc_type"], src["doc_type"]))
    if src.get("page_number") is not None:
        parts.append(f"page {src['page_number']}")
    if src.get("course"):
        parts.append(src["course"])
    if src.get("topic"):
        parts.append(f"topic: {src['topic']}")
    return " · ".join(parts)


def render_sources(sources: dict):
    if sources is None:
        return
    cited = sources.get("cited") or []
    if cited:
        with st.expander(f"📚 Sources from the course material ({len(cited)})"):
            for src in cited:
                st.markdown(f"- {format_source(src)}")
    else:
        st.caption("📚 No course material was cited for this answer.")


def render_message(msg: dict):
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🧑‍🏫"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources"))


# ---------------------------------------------------------------------------- chat
def ask(question: str, scope: str):
    messages = st.session_state.setdefault("messages", [])
    # backend schema: max 20000 chars per turn (and it keeps only its own last N turns)
    history = [{"role": m["role"], "content": m["content"][:20000]} for m in messages][-HISTORY_TURNS_SENT:]
    messages.append({"role": "user", "content": question})
    render_message(messages[-1])

    with st.chat_message("assistant", avatar="🧑‍🏫"):
        with st.spinner("The TA is preparing an answer..."):
            try:
                body = client().answer(st.session_state["student"]["project_id"], question,
                                       chat_history=history, doc_type=SCOPES[scope])
            except ApiError as e:
                messages.pop()  # keep the history consistent: the question was not answered
                if e.status_code == 401:
                    logout(friendly(e))
                    st.rerun()
                st.session_state["chat_error"] = friendly(e)
                st.rerun()
            except Exception as e:
                messages.pop()
                st.session_state["chat_error"] = friendly(e)
                st.rerun()
    messages.append({"role": "assistant", "content": body["answer"], "sources": body.get("sources")})
    st.rerun()


def chat_page():
    me = st.session_state["student"]
    with st.sidebar:
        st.markdown(f"### 👋 {me.get('name') or 'Welcome'}")
        scope = st.radio("Search in", list(SCOPES), index=0)
        if st.button("🆕 New conversation"):
            st.session_state["messages"] = []
            st.rerun()
        if st.button("Log out"):
            logout()
            st.rerun()
        st.caption("Answers are generated by an AI teaching assistant from the course materials. "
                   "Check important details with your TA.")

    st.title("💬 Structured Programming TA")
    messages = st.session_state.setdefault("messages", [])

    if not messages:
        st.markdown("Hi! I'm your TA for **Structured Programming** (we practise it with C). Ask me about "
                    "algorithms, flowcharts, conditions, loops, arrays or functions - or ask me to trace code, "
                    "predict output or find a bug. English or Arabic.")
        cols = st.columns(len(EXAMPLES))
        for col, (label, prompt) in zip(cols, EXAMPLES):
            if col.button(label, width="stretch"):
                st.session_state["pending_question"] = prompt
                st.rerun()

    for msg in messages:
        render_message(msg)

    if st.session_state.get("chat_error"):
        st.error(st.session_state.pop("chat_error"))

    question = st.chat_input("Ask about the course... / اسأل عن المادة...", max_chars=8000)
    question = question or st.session_state.pop("pending_question", None)
    if question and question.strip():
        ask(question.strip(), scope)


if ensure_student_session():
    chat_page()
else:
    login_page()

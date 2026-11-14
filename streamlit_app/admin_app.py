"""
TA / Admin dashboard - manage the course materials students learn from.

    streamlit run streamlit_app/admin_app.py --server.port 8502

All data comes from the backend; all permissions are enforced by the backend (admin token).
"""
import pandas as pd
import streamlit as st

import settings
from admin_actions import DOC_TYPES, build_index, process_documents, upload_batch
from api_client import ApiError, BackendClient, friendly
from identity import backend

st.set_page_config(page_title="Course Admin - Structured Programming Assistant", page_icon="🛠️", layout="wide")

STATUS_LABELS = {
    "pending": "⏳ pending", "processed": "✅ processed", "failed": "❌ failed",
    "indexed": "✅ indexed", "not_indexed": "⏳ not indexed", "out_of_date": "⚠️ out of date", "stale": "⚠️ stale",
}


# ---------------------------------------------------------------------------- session / auth
def logout(message: str = None):
    for key in ("admin_token", "admin_user"):
        st.session_state.pop(key, None)
    if message:
        st.session_state["login_message"] = message


def current_client():
    return backend(st.session_state.get("admin_token"))


def ensure_admin_session():
    """Validate the stored token with the backend on every run (never trust the frontend alone)."""
    if "admin_token" not in st.session_state:
        return False
    try:
        me = current_client().me()
    except ApiError as e:
        logout(friendly(e))
        return False
    except Exception as e:
        st.error(friendly(e))
        st.stop()
    if me.get("role") != "admin":
        logout("This account is not an admin account.")
        return False
    st.session_state["admin_user"] = me
    return True


def login_page():
    st.title("🛠️ Course Admin")
    st.caption(f"{settings.COURSE_TITLE} - Teaching Assistant dashboard")
    if st.session_state.get("login_message"):
        st.warning(st.session_state.pop("login_message"))
    with st.form("admin_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")
    if submitted:
        try:
            body = backend().admin_login(username, password)
            st.session_state["admin_token"] = body["access_token"]
            st.rerun()
        except Exception as e:
            st.error(friendly(e))


# ---------------------------------------------------------------------------- dashboard
def flash(kind: str, text: str):
    st.session_state.setdefault("flash", []).append((kind, text))


def show_flash():
    for kind, text in st.session_state.pop("flash", []):
        getattr(st, kind)(text)


def pdf_name(name: str, fallback: str):
    name = (name or "").strip() or fallback
    return name if name.lower().endswith(".pdf") else name + ".pdf"


def handle_api_error(e: Exception):
    if isinstance(e, ApiError) and e.status_code == 401:
        logout(friendly(e))
        st.rerun()
    st.error(friendly(e))


def metrics(summary: dict):
    cols = st.columns(7)
    cols[0].metric("Documents", summary["total_documents"])
    cols[1].metric("Lectures", summary["lectures"])
    cols[2].metric("Labs", summary["labs"])
    cols[3].metric("Indexed", summary["indexed_documents"])
    cols[4].metric("Awaiting processing", summary["awaiting_processing"])
    cols[5].metric("Awaiting indexing", summary["awaiting_indexing"])
    cols[6].metric("Processing errors", summary["processing_errors"])
    if summary.get("index_needs_rebuild"):
        st.warning(f"The search index is out of sync with the processed documents "
                   f"({summary['total_vectors']} vectors vs {summary['total_chunks']} chunks). "
                   f"Use **Build search index** to rebuild it.")


def upload_tab(client: BackendClient, project_id: str):
    st.subheader("Upload course materials")
    st.caption(f"PDF only, up to {settings.MAX_UPLOAD_MB} MB per file. The metadata below is applied to every "
               f"selected file. Students never upload files.")
    files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, key="uploader")
    c1, c2, c3 = st.columns(3)
    doc_type_label = c1.radio("Document type", list(DOC_TYPES), horizontal=True)
    course = c2.text_input("Course", placeholder="e.g. CS101 Structured Programming")
    topic = c3.text_input("Topic", placeholder="e.g. Loops")

    names = []
    if files:
        st.markdown("**Document names shown to students** (defaults to the original file name)")
        for i, f in enumerate(files):
            names.append(st.text_input(f"Name for file {i + 1}", value=f.name, key=f"name_{i}_{f.name}"))

    if st.button("Upload", type="primary", disabled=not files):
        batch = [(pdf_name(name, f.name), f.getvalue()) for f, name in zip(files, names)]
        with st.spinner(f"Uploading {len(batch)} file(s)..."):
            results = upload_batch(client, project_id, batch, DOC_TYPES[doc_type_label], course, topic,
                                   settings.MAX_UPLOAD_MB)
        for r in results:
            if r["ok"]:
                flash("success", f"Uploaded: {r['name']}")
            else:
                flash("error", f"Not uploaded: {r['name']} - {r['error']}")
        if any(r["ok"] for r in results):
            flash("info", "Next step: **Process & index** tab.")
            st.session_state.pop("uploader", None)  # clear the selection
        st.rerun()


def process_tab(client: BackendClient, project_id: str, listing: dict):
    st.subheader("Process and index")
    st.markdown("1. **Process** extracts the text of each PDF and splits it into chunks.  \n"
                "2. **Build search index** embeds all chunks and stores them in the vector index "
                "(full rebuild, so the index always matches the processed documents).")
    assets = listing["assets"]
    to_process = [a for a in assets if a["processing_status"] in ("pending", "failed")]
    chunks = listing["summary"]["total_chunks"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{len(to_process)}** document(s) waiting for processing (pending or failed).")
        if st.button("Process documents", disabled=not to_process):
            with st.spinner("Processing on the server..."):
                results = process_documents(client, project_id, assets)
            for r in results:
                flash("success" if r["ok"] else "error",
                      f"{r['name']}: {str(r['chunks']) + ' chunks' if r['ok'] else r['error']}")
            st.rerun()
    with c2:
        st.markdown(f"**{chunks}** processed chunk(s) in total.")
        if st.button("Build search index", type="primary", disabled=chunks == 0):
            with st.spinner(f"Embedding {chunks} chunks on the server - this can take a few minutes..."):
                try:
                    count = build_index(client, project_id)
                    flash("success", f"Search index rebuilt by the backend: {count} chunks indexed.")
                    st.rerun()
                except Exception as e:
                    handle_api_error(e)
    if st.button("Refresh status"):
        st.rerun()


def documents_tab(client: BackendClient, project_id: str, listing: dict):
    st.subheader("Course materials")
    assets = listing["assets"]
    if not assets:
        st.info("No documents uploaded yet.")
        return
    table = pd.DataFrame([{
        "Document": a["document_name"],
        "Type": a["doc_type"].capitalize(),
        "Course": a["course"] or "",
        "Topic": a["topic"] or "",
        "Uploaded": (a["uploaded_at"] or "")[:16].replace("T", " "),
        "Processing": STATUS_LABELS.get(a["processing_status"], a["processing_status"]),
        "Chunks": a["chunks"],
        "Indexing": STATUS_LABELS.get(a["indexing_status"], a["indexing_status"]),
        "Error": a["processing_error"] or "",
    } for a in assets])
    st.dataframe(table, hide_index=True, width="stretch")

    with st.expander("Delete a document"):
        st.caption("Deletes the PDF, its chunks and its vectors. Other documents are not affected. "
                   "Rebuild the search index afterwards.")
        options = {f"{a['document_name']}  ({a['doc_type']}, {a['file_id']})": a for a in assets}
        choice = st.selectbox("Document", list(options))
        confirm = st.checkbox("I understand this cannot be undone")
        if st.button("Delete document", disabled=not confirm):
            asset = options[choice]
            try:
                body = client.delete_asset(project_id, asset["file_id"])
                flash("success", f"Deleted {asset['document_name']}: {body['deleted_chunks']} chunks, "
                                 f"{body['deleted_vectors']} vectors removed.")
            except Exception as e:
                handle_api_error(e)
            else:
                st.rerun()


def dashboard():
    client = current_client()
    with st.sidebar:
        st.markdown(f"**Signed in as** `{st.session_state['admin_user']['subject'].split(':', 1)[-1]}`")
        project_id = st.text_input("Course project ID", value=settings.ADMIN_DEFAULT_PROJECT_ID,
                                   help="Must match the backend STUDENT_PROJECT_ID so students can use it.")
        if st.button("Log out"):
            logout()
            st.rerun()
        st.caption(f"Backend: {settings.API_BASE_URL}")

    st.title("🛠️ Course Admin")
    if not project_id or not project_id.isalnum():
        st.info("Enter the course project ID (letters and digits only) in the sidebar.")
        return
    try:
        listing = client.list_assets(project_id)
    except Exception as e:
        handle_api_error(e)
        return

    show_flash()
    metrics(listing["summary"])
    tab_upload, tab_process, tab_docs = st.tabs(["Upload", "Process & index", "Documents"])
    with tab_upload:
        upload_tab(client, project_id)
    with tab_process:
        process_tab(client, project_id, listing)
    with tab_docs:
        documents_tab(client, project_id, listing)


if ensure_admin_session():
    dashboard()
else:
    login_page()

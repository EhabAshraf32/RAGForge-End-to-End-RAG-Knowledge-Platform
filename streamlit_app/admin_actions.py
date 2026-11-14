"""Admin workflow steps used by admin_app.py. Each step only calls the backend and reports what the
backend returned - no processing / indexing logic and no optimistic success messages."""
from api_client import ApiError, BackendClient, friendly

DOC_TYPES = {"Lecture": "lecture", "Lab": "lab", "Other": "other"}


def validate_pdf(name: str, content: bytes, max_mb: int):
    """Returns an error message, or None if the file looks like an acceptable PDF."""
    if not name.lower().endswith(".pdf"):
        return "only PDF files are accepted"
    if not content:
        return "the file is empty"
    if len(content) > max_mb * 1024 * 1024:
        return f"larger than {max_mb} MB"
    if not content.startswith(b"%PDF-"):
        return "not a valid PDF file"
    return None


def upload_batch(client: BackendClient, project_id: str, files: list, doc_type: str,
                 course: str = None, topic: str = None, max_mb: int = 10):
    """files: [(display_name, bytes)]. Validates each file, uploads the valid ones one by one."""
    results = []
    for name, content in files:
        error = validate_pdf(name, content, max_mb)
        if error:
            results.append({"name": name, "ok": False, "error": error})
            continue
        try:
            body = client.upload(project_id, name, content, doc_type, course or None, topic or None)
            results.append({"name": name, "ok": True, "file_id": body["file_id"]})
        except Exception as e:
            results.append({"name": name, "ok": False, "error": friendly(e)})
    return results


def process_documents(client: BackendClient, project_id: str, assets: list, include_failed: bool = True):
    """Process every document that is pending (and optionally retry failed ones), one file at a time,
    so one bad PDF does not hide the others. Re-processing a file replaces its chunks in the backend."""
    wanted = {"pending", "failed"} if include_failed else {"pending"}
    results = []
    for asset in assets:
        if asset["processing_status"] not in wanted:
            continue
        try:
            body = client.process(project_id, file_id=asset["file_id"])
            results.append({"name": asset["document_name"], "ok": True, "chunks": body.get("inserted_chunks")})
        except ApiError as e:
            results.append({"name": asset["document_name"], "ok": False, "error": friendly(e)})
        except Exception as e:
            results.append({"name": asset["document_name"], "ok": False, "error": friendly(e)})
    return results


def build_index(client: BackendClient, project_id: str):
    """Full rebuild of the project's vector index from the processed chunks (do_reset=1).
    A full rebuild keeps the index exactly in sync with MongoDB, also after deletions."""
    body = client.push(project_id, do_reset=1)
    return body.get("inserted_items_count", 0)

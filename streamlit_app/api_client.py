"""
Thin HTTP client for the MINI-RAG FastAPI backend.

The Streamlit apps contain NO RAG / processing logic and never access MongoDB or Qdrant:
every action is a call to the existing backend endpoints, and every permission is enforced there.
"""
import requests


class ApiError(Exception):
    def __init__(self, status_code: int, signal: str = None, message: str = None, retry_after: int = None):
        self.status_code = status_code
        self.signal = signal
        self.retry_after = retry_after
        super().__init__(message or signal or f"HTTP {status_code}")


FRIENDLY_MESSAGES = {
    "invalid_credentials": "Wrong username, password or access code.",
    "authentication_required": "Please log in.",
    "invalid_token": "Your session is not valid. Please log in again.",
    "token_expired": "Your session expired. Please log in again.",
    "forbidden": "You are not allowed to do this.",
    "rate_limit_exceeded": "Too many requests. Please wait a moment and try again.",
    "authentication_not_configured": "Login is not configured on the server yet. Contact the TA.",
    "file_type_not_supported": "File type not supported (PDF only).",
    "file_size_exceeded": "File is larger than the server limit.",
    "doc_type_not_supported": "Unknown document type.",
    "rag_answer_error": "No answer could be generated (the course index may be empty, or the AI service is "
                        "busy). Please try again in a moment.",
    "processing_failed": "No text could be extracted from the selected document(s).",
    "not_found_files": "There are no uploaded files in this project.",
    "no_file_found_with_this_id": "Document not found (it may have been deleted).",
    "insert_into_vectordb_error": "Indexing failed (see backend log).",
    "asset_delete_failed": "Deletion failed (see backend log).",
}


def friendly(error: Exception) -> str:
    if isinstance(error, ApiError):
        text = FRIENDLY_MESSAGES.get(error.signal) or str(error)
        if error.signal == "rate_limit_exceeded" and error.retry_after:
            text += f" (retry in about {error.retry_after} s)"
        return text
    if isinstance(error, requests.ConnectionError):
        return "The course server is not reachable right now."
    if isinstance(error, requests.Timeout):
        return "The server took too long to respond. Please try again."
    return f"Unexpected error: {error}"


class BackendClient:

    def __init__(self, base_url: str, token: str = None, timeout: float = 180,
                 client_ip: str = None, frontend_secret: str = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        # end-user identity for per-user rate limiting (only honoured by the backend with the secret)
        self.client_ip = client_ip
        self.frontend_secret = frontend_secret

    # ------------------------------------------------------------------ plumbing
    def _headers(self):
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        if self.frontend_secret and self.client_ip:
            headers["X-Frontend-Secret"] = self.frontend_secret
            headers["X-Client-IP"] = str(self.client_ip)
        return headers

    def _request(self, method: str, path: str, timeout: float = None, **kwargs):
        response = requests.request(method, f"{self.base_url}{path}", headers=self._headers(),
                                    timeout=timeout or self.timeout, **kwargs)
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400:
            detail = body.get("detail") if isinstance(body, dict) else None
            signal = body.get("signal") if isinstance(body, dict) else None
            if isinstance(detail, dict):
                signal = detail.get("signal", signal)
            elif isinstance(detail, list):  # FastAPI validation errors
                signal = signal or "validation_error"
            retry = response.headers.get("Retry-After")
            raise ApiError(response.status_code, signal,
                           message=None if signal else f"HTTP {response.status_code}",
                           retry_after=int(retry) if retry and retry.isdigit() else None)
        return body

    # ------------------------------------------------------------------ auth
    def admin_login(self, username: str, password: str):
        return self._request("POST", "/auth/admin/login", json={"username": username, "password": password})

    def student_login(self, access_code: str, display_name: str = None):
        return self._request("POST", "/auth/student/login",
                             json={"access_code": access_code, "display_name": display_name or None})

    def me(self):
        return self._request("GET", "/auth/me", timeout=15)

    # ------------------------------------------------------------------ admin
    def upload(self, project_id: str, filename: str, content: bytes, doc_type: str,
               course: str = None, topic: str = None):
        data = {"doc_type": doc_type}
        if course:
            data["course"] = course
        if topic:
            data["topic"] = topic
        return self._request("POST", f"/data/upload/{project_id}",
                             files={"file": (filename, content, "application/pdf")}, data=data)

    def process(self, project_id: str, file_id: str = None, do_reset: int = 0):
        body = {"do_reset": do_reset}
        if file_id:
            body["file_id"] = file_id
        return self._request("POST", f"/data/process/{project_id}", json=body, timeout=900)

    def push(self, project_id: str, do_reset: int = 1):
        # embedding every chunk on CPU can take minutes for a whole course
        return self._request("POST", f"/nlp/index/push/{project_id}", json={"do_reset": do_reset}, timeout=3600)

    def list_assets(self, project_id: str):
        return self._request("GET", f"/data/assets/{project_id}", timeout=60)

    def delete_asset(self, project_id: str, file_id: str):
        return self._request("DELETE", f"/data/asset/{project_id}/{file_id}", timeout=120)

    # ------------------------------------------------------------------ student
    def answer(self, project_id: str, text: str, chat_history: list = None, doc_type: str = None,
               limit: int = 5):
        body = {"text": text, "limit": limit, "chat_history": chat_history or []}
        if doc_type:
            body["doc_type"] = doc_type
        return self._request("POST", f"/nlp/index/answer/{project_id}", json=body, timeout=300)

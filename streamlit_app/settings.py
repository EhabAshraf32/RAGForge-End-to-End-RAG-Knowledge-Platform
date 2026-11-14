"""Frontend settings from environment variables, streamlit_app/.env, or Streamlit secrets
(Streamlit Community Cloud). The only secret is FRONTEND_SHARED_SECRET, which stays on the Streamlit
server (it is never sent to the browser): users type their own credentials, exchanged for tokens."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass


def _get(name: str, default: str = ""):
    value = os.environ.get(name)
    if value:
        return value
    try:  # Streamlit Community Cloud secrets
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


API_BASE_URL = _get("API_BASE_URL", "http://127.0.0.1:8000/api/v1")
# project the TA manages; must be the same as the backend's STUDENT_PROJECT_ID so students see it
ADMIN_DEFAULT_PROJECT_ID = _get("ADMIN_DEFAULT_PROJECT_ID", "")
# should match the backend FILE_MAX_SIZE (MB); the backend enforces it anyway
MAX_UPLOAD_MB = int(_get("MAX_UPLOAD_MB", "10"))
COURSE_TITLE = _get("COURSE_TITLE", "Structured Programming using C")
# must equal the backend TRUSTED_FRONTEND_SECRET; lets the backend rate-limit per end user instead of
# per Streamlit server (all users of a hosted Streamlit app reach the API from the same IP)
FRONTEND_SHARED_SECRET = _get("FRONTEND_SHARED_SECRET", "")

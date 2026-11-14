"""
Runs the existing FastAPI backend (src/main.py) INSIDE the Streamlit server process, on 127.0.0.1 only.

Used on hosts that offer no free backend service (API_BASE_URL = "embedded", e.g. Streamlit Community Cloud).
- Same code path as production: real uvicorn, real HTTP, same auth / roles / rate limits / RAG pipeline.
- Not reachable from the internet: it listens on localhost; browsers only talk to the Streamlit page.
- Secrets come from Streamlit secrets (server-side) and are never sent to the browser.
- Data lives in external services (MongoDB Atlas, Qdrant Cloud, Cloudflare), so restarts lose nothing.
"""
import os
import socket
import sys
import threading
import time

import streamlit as st

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _export_secrets_to_env():
    """Backend settings are read from environment variables; copy root-level Streamlit secrets."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)) and key not in os.environ:
                os.environ[key] = str(value).lower() if isinstance(value, bool) else str(value)
    except Exception:
        pass  # no secrets file (local runs use src/.env or the environment)


@st.cache_resource(show_spinner="Starting the course assistant...")
def start_embedded_backend() -> str:
    _export_secrets_to_env()
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    import uvicorn
    import main  # the existing FastAPI app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="embedded-backend", daemon=True)
    thread.start()
    for _ in range(600):  # startup connects to MongoDB / Qdrant; fail loudly instead of hanging
        if server.started:
            return f"http://127.0.0.1:{port}/api/v1"
        if not thread.is_alive():
            break
        time.sleep(0.1)
    raise RuntimeError("The embedded backend failed to start - check the app logs and the secrets.")

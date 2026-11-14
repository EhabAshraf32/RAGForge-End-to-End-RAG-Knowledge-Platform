"""Builds backend clients that carry the end user's identity for per-user rate limiting."""
import uuid

import streamlit as st

import settings
from api_client import BackendClient


def end_user_key():
    """The viewer's IP as seen by Streamlit (None on localhost) or, failing that, a random id per
    browser session. Used ONLY as a rate-limit key by the backend - never for authorization."""
    try:
        ip = st.context.ip_address
    except Exception:
        ip = None
    if ip:
        return ip
    if "_anon_session_id" not in st.session_state:
        st.session_state["_anon_session_id"] = f"session-{uuid.uuid4().hex}"
    return st.session_state["_anon_session_id"]


def backend(token: str = None) -> BackendClient:
    return BackendClient(settings.API_BASE_URL, token,
                         client_ip=end_user_key(), frontend_secret=settings.FRONTEND_SHARED_SECRET or None)

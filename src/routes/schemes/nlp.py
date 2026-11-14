from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class PushRequest(BaseModel):
    do_reset: Optional[int] = 0

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=20000)

class SearchRequest(BaseModel):
    # bounded to keep per-request cost predictable (prompt size / number of retrieved chunks)
    text: str = Field(..., min_length=1, max_length=8000)
    limit: Optional[int] = Field(5, ge=1, le=20)
    # optional: only retrieve "lecture" / "lab" / "other" chunks
    doc_type: Optional[str] = None
    # optional, /answer only: previous turns managed by the client (e.g. Streamlit st.session_state)
    chat_history: Optional[List[ChatTurn]] = Field(None, max_length=50)

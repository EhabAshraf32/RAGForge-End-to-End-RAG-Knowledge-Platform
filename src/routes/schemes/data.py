from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    file_id: str=None
    # keep chunk_size <= INPUT_DAFAULT_MAX_CHARACTERS (embedding input limit),
    # otherwise the tail of each chunk is not embedded
    chunk_size: Optional[int] = 1000
    overlap_size: Optional[int] = 200
    do_reset: Optional[int] = 0

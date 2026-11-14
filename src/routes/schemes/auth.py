from pydantic import BaseModel, Field
from typing import Optional

class AdminLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)

class StudentLoginRequest(BaseModel):
    access_code: str = Field(..., min_length=1, max_length=200)
    display_name: Optional[str] = Field(None, max_length=40)

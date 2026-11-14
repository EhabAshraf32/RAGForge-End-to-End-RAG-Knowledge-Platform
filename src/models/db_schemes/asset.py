from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectId
from datetime import datetime

class Asset(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    asset_project_id: ObjectId
    asset_type: str = Field(..., min_length=1)
    asset_name: str = Field(..., min_length=1)
    asset_size: int = Field(ge=0, default=None)
    # {"original_name", "doc_type", "course", "topic"} set at upload; None for assets uploaded before
    asset_config: Optional[dict] = Field(default=None)
    # upload time; set explicitly at upload (the old default was the function object itself and was
    # never stored because inserts use exclude_unset) -> None for assets uploaded before this fix
    asset_pushed_at: Optional[datetime] = Field(default=None)
    # last processing result written by /data/process: {"status", "chunks", "error", "processed_at"}
    asset_processing: Optional[dict] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_indexes(cls):

        return [
            {
                "key": [
                    ("asset_project_id", 1)
                ],
                "name": "asset_project_id_index_1",
                "unique": False
            },
            {
                "key": [
                    ("asset_project_id", 1),
                    ("asset_name", 1)
                ],
                "name": "asset_project_id_name_index_1",
                "unique": True
            },
        ]
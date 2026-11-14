from .BaseDataModel import BaseDataModel
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
import os


class FileStoreModel(BaseDataModel):
    """
    Uploaded course files kept in MongoDB GridFS (enabled with STORE_FILES_IN_DB=true).
    Used when the API server has an ephemeral disk (e.g. Render free): the local copy in
    assets/files/<project_id>/ is only a cache and is restored from GridFS when missing.
    """

    BUCKET_NAME = "asset_files"

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.bucket = AsyncIOMotorGridFSBucket(self.db_client, bucket_name=self.BUCKET_NAME)

    @classmethod
    async def create_instance(cls, db_client: object):
        return cls(db_client)

    @staticmethod
    def stored_name(project_id: str, file_id: str):
        return f"{project_id}/{file_id}"

    async def _find_ids(self, project_id: str, file_id: str):
        cursor = self.bucket.find({"filename": self.stored_name(project_id, file_id)})
        return [doc._id async for doc in cursor]

    async def save_file(self, project_id: str, file_id: str, file_path: str):
        with open(file_path, "rb") as f:
            return await self.bucket.upload_from_stream(
                self.stored_name(project_id, file_id), f,
                metadata={"project_id": project_id, "file_id": file_id},
            )

    async def restore_file(self, project_id: str, file_id: str, target_path: str):
        """Write the stored file to target_path. Returns False if it is not in the store."""
        ids = await self._find_ids(project_id, file_id)
        if not ids:
            return False
        tmp_path = target_path + ".part"
        with open(tmp_path, "wb") as f:
            await self.bucket.download_to_stream(ids[-1], f)
        os.replace(tmp_path, target_path)
        return True

    async def delete_file(self, project_id: str, file_id: str):
        ids = await self._find_ids(project_id, file_id)
        for _id in ids:
            await self.bucket.delete(_id)
        return len(ids)

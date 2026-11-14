from pathlib import Path
import gc
import os
import shutil
from models.db_schemes.data_chunk import RetrievedDocument
from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
import logging
from typing import List

class QdrantDBProvider(VectorDBInterface):

    # payload fields used in filters (doc_type filter, per-file count / delete)
    INDEXED_PAYLOAD_FIELDS = ("metadata.doc_type", "metadata.file_id")

    def __init__(self, db_path: str, distance_method: str, url: str = None, api_key: str = None):

        self.client = None
        self.db_path = db_path
        # url set -> Qdrant server / Qdrant Cloud; otherwise embedded storage in db_path
        self.url = url
        self.api_key = api_key
        self.distance_method = None

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger(__name__)

    def is_remote(self):
        return bool(self.url)

    def connect(self):
        if self.is_remote():
            self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=60)
            self.logger.info("Connected to remote Qdrant")
            return
        print("Current working directory:", Path.cwd())
        print("Qdrant path:", Path(self.db_path).resolve())
        self.client = QdrantClient(path=self.db_path)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)
    
    def list_all_collections(self) -> List:
        return self.client.get_collections()
    
    def get_collection_info(self, collection_name: str) -> dict:
        return self.client.get_collection(collection_name=collection_name)
    
    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            result = self.client.delete_collection(collection_name=collection_name)
            self._remove_local_collection_files(collection_name)
            return result

    def _remove_local_collection_files(self, collection_name: str):
        """
        Local (embedded) Qdrant drops the collection object without closing its SQLite file and then
        calls rmtree(ignore_errors=True). On Windows the still-open file cannot be removed, and the next
        create_collection with the same name silently reloads the OLD points (do_reset would not reset).
        Collect the orphaned connection and remove ONLY this collection's directory inside db_path.
        """
        if self.is_remote():  # a Qdrant server manages its own storage
            return
        gc.collect()
        base = os.path.realpath(os.path.join(self.db_path, "collection"))
        target = os.path.realpath(os.path.join(base, collection_name))
        if os.path.dirname(target) != base:  # never touch anything outside <db_path>/collection/
            self.logger.error(f"Refusing to remove unexpected path for collection {collection_name}")
            return
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
            if os.path.isdir(target):
                self.logger.error(f"Could not remove local files of deleted collection {collection_name}")
        
    def create_collection(self, collection_name: str, 
                                embedding_size: int,
                                do_reset: bool = False):
        if do_reset:
            _ = self.delete_collection(collection_name=collection_name)
        
        if not self.is_collection_existed(collection_name):
            _ = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method
                )
            )
            if self.is_remote():
                # Qdrant servers can require an index for filtered queries (local mode ignores indexes)
                for field in self.INDEXED_PAYLOAD_FIELDS:
                    self.client.create_payload_index(collection_name=collection_name, field_name=field,
                                                     field_schema=models.PayloadSchemaType.KEYWORD)

            return True
        
        return False
    
    # def insert_one(self, collection_name: str, text: str, vector: list,
    #                      metadata: dict = None, 
    #                      record_id: str = None):
        
    #     if not self.is_collection_existed(collection_name):
    #         self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
    #         return False
        
    #     try:
    #         _ = self.client.upload_records(
    #             collection_name=collection_name,
    #             records=[
    #                 models.Record(
    #                     id=[record_id],
    #                     vector=vector,
    #                     payload={
    #                         "text": text, "metadata": metadata
    #                     }
    #                 )
    #             ]
    #         )
    #     except Exception as e:
    #         self.logger.error(f"Error while inserting batch: {e}")
    #         return False

    #     return True
    
    # def insert_many(self, collection_name: str, texts: list, 
    #                       vectors: list, metadata: list = None, 
    #                       record_ids: list = None, batch_size: int = 50):
        
    #     if metadata is None:
    #         metadata = [None] * len(texts)

    #     if record_ids is None:
    #         record_ids = list(range(0, len(texts)))

    #     for i in range(0, len(texts), batch_size):
    #         batch_end = i + batch_size

    #         batch_texts = texts[i:batch_end]
    #         batch_vectors = vectors[i:batch_end]
    #         batch_metadata = metadata[i:batch_end]
    #         batch_record_ids=record_ids[i:batch_end]

    #         batch_records = [
    #             models.Record(
    #                 id=batch_record_ids[x],
    #                 vector=batch_vectors[x],
    #                 payload={
    #                     "text": batch_texts[x], "metadata": batch_metadata[x]
    #                 }
    #             )

    #             for x in range(len(batch_texts))
    #         ]

    #         try:
    #             _ = self.client.upload_records(
    #                 collection_name=collection_name,
    #                 records=batch_records,
    #             )
    #         except Exception as e:
    #             self.logger.error(f"Error while inserting batch: {e}")
    #             return False

    #     return True
        
    ##newest version used points method rather than upload method
    def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False
        
        try:
            _ = self.client.upload_points(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=record_id,
                        vector=vector,
                        payload={
                            "text": text, "metadata": metadata
                        }
                    )
                ]
            )
        except Exception as e:
            self.logger.error(f"Error while inserting batch: {e}")
            return False

        return True
    
    def insert_many(self, collection_name: str, texts: list, 
                          vectors: list, metadata: list = None, 
                          record_ids: list = None, batch_size: int = 50):
        
        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None:
            record_ids = list(range(0, len(texts)))

        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            batch_record_ids=record_ids[i:batch_end]

            batch_points = [
                models.PointStruct(
                    id=batch_record_ids[x],
                    vector=batch_vectors[x],
                    payload={
                        "text": batch_texts[x], "metadata": batch_metadata[x]
                    }
                )

                for x in range(len(batch_texts))
            ]

            try:
                _ = self.client.upload_points(
                    collection_name=collection_name,
                    points=batch_points,
                )
            except Exception as e:
                self.logger.error(f"Error while inserting batch: {e}")
                return False

        return True
        
    def get_collection_vector_size(self, collection_name: str) -> int:
        if not self.is_collection_existed(collection_name):
            return None
        info = self.get_collection_info(collection_name=collection_name)
        vectors = info.config.params.vectors
        return getattr(vectors, "size", None)

    def _metadata_filter(self, filter_metadata: dict):
        """Exact-match filter on payload metadata, e.g. {"doc_type": "lab"}."""
        if not filter_metadata:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value)
                )
                for key, value in filter_metadata.items()
            ]
        )

    def count_by_metadata(self, collection_name: str, filter_metadata: dict) -> int:
        if not self.is_collection_existed(collection_name):
            return 0
        return self.client.count(collection_name=collection_name,
                                 count_filter=self._metadata_filter(filter_metadata), exact=True).count

    def delete_by_metadata(self, collection_name: str, filter_metadata: dict) -> bool:
        """Delete only the points matching ALL given metadata values (refuses an empty filter)."""
        if not filter_metadata:
            self.logger.error("Refusing delete_by_metadata without a filter")
            return False
        if not self.is_collection_existed(collection_name):
            return True
        self.client.delete(collection_name=collection_name,
                           points_selector=models.FilterSelector(filter=self._metadata_filter(filter_metadata)))
        return True

    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5,
                               filter_metadata: dict = None):

        query_filter = self._metadata_filter(filter_metadata)

        result = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            query_filter=query_filter,
        )
        
        if not result.points:
            return None
            
        return [
            RetrievedDocument(**{
                "score": point.score,
                "text": point.payload["text"],
                "metadata": point.payload.get("metadata"),
            })
            for point in result.points
        ]
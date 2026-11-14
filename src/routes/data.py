from fastapi import FastAPI, APIRouter, Depends, UploadFile, status, Request, Form
from fastapi.responses import JSONResponse
from typing import Optional
from datetime import datetime
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController, NLPController
import aiofiles
from models import ResponseSignal, DocTypeEnum
import logging
from .schemes.data import ProcessRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from models.FileStoreModel import FileStoreModel
from models.db_schemes import DataChunk, Asset
from models.enums.AssetTypeEnum import AssetTypeEnum
from .dependencies import require_admin

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)

@data_router.post("/upload/{project_id}", dependencies=[Depends(require_admin)])
async def upload_data(request: Request, project_id: str, file: UploadFile,
                      doc_type: Optional[str] = Form(None),
                      course: Optional[str] = Form(None),
                      topic: Optional[str] = Form(None),
                      app_settings: Settings = Depends(get_settings)):

    # optional course metadata (backward compatible: old clients send only `file`)
    normalized_doc_type = DocTypeEnum.normalize(doc_type) if doc_type else DocTypeEnum.OTHER.value
    if normalized_doc_type is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.DOC_TYPE_NOT_SUPPORTED.value
            }
        )

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    # validate the file properties
    data_controller = DataController()

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
            }
        )

    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )
    max_bytes = app_settings.FILE_MAX_SIZE * data_controller.size_scale
    written = 0
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                written += len(chunk)
                if written > max_bytes:  # enforce the size limit even when the client sent no size
                    break
                await f.write(chunk)
    except Exception as e:

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )

    # content checks (the declared content type comes from the client and is not trusted)
    rejected_signal = None
    if written > max_bytes:
        rejected_signal = ResponseSignal.FILE_SIZE_EXCEEDED.value
    elif written == 0:
        rejected_signal = ResponseSignal.FILE_UPLOAD_FAILED.value
    elif file_id.lower().endswith(".pdf"):
        with open(file_path, "rb") as f:
            if f.read(5) != b"%PDF-":
                rejected_signal = ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value
    if rejected_signal:
        os.remove(file_path)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"signal": rejected_signal})

    # persistent copy (ephemeral server disks lose assets/files on restart)
    if app_settings.STORE_FILES_IN_DB:
        try:
            file_store = await FileStoreModel.create_instance(db_client=request.app.db_client)
            await file_store.save_file(project_id=project_id, file_id=file_id, file_path=file_path)
        except Exception as e:
            logger.error(f"Error while storing file in the database: {e}")
            os.remove(file_path)
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                                content={"signal": ResponseSignal.FILE_UPLOAD_FAILED.value})

    # store the assets into the database
    asset_model = await AssetModel.create_instance(
        db_client=request.app.db_client
    )

    asset_resource = Asset(
        asset_project_id=project.id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id,
        asset_size=os.path.getsize(file_path),
        asset_pushed_at=datetime.utcnow(),
        asset_config={
            "original_name": file.filename,
            "doc_type": normalized_doc_type,
            "course": course.strip() if course else None,
            "topic": topic.strip() if topic else None,
        }
    )

    asset_record = await asset_model.create_asset(asset=asset_resource)

    return JSONResponse(
            content={
                "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
                # "file_id": str(asset_record.id),
                "file_id": asset_record.asset_name

            }
        )

@data_router.post("/process/{project_id}", dependencies=[Depends(require_admin)])
async def process_endpoint(request: Request, project_id: str, process_request: ProcessRequest):

    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    do_reset = process_request.do_reset

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    asset_model = await AssetModel.create_instance(
            db_client=request.app.db_client
        )

    project_files_ids = {}
    if process_request.file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.id,
            asset_name=process_request.file_id
        )

        if asset_record is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.FILE_ID_ERROR.value,
                }
            )

        project_files_ids = {
            asset_record.id: asset_record.asset_name
        }
        project_files_configs = {
            asset_record.id: asset_record.asset_config
        }

    else:
        

        project_files = await asset_model.get_all_project_assets(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value,
        )

        project_files_ids = {
            record.id: record.asset_name
            for record in project_files
        }
        project_files_configs = {
            record.id: record.asset_config
            for record in project_files
        }

    if len(project_files_ids) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.NO_FILES_ERROR.value,
            }
        )
    
    process_controller = ProcessController(project_id=project_id)

    no_records = 0
    no_files = 0

    chunk_model = await ChunkModel.create_instance(
                        db_client=request.app.db_client
                    )

    if do_reset == 1:
        _ = await chunk_model.delete_chunks_by_project_id(
            project_id=project.id
        )

        # keep the vector index consistent with MongoDB: the vectors of the deleted
        # chunks would otherwise stay searchable until the next push with do_reset=1
        nlp_controller = NLPController(
            vectordb_client=request.app.vectordb_client,
            generation_client=request.app.generation_client,
            embedding_client=request.app.embedding_client,
            template_parser=request.app.template_parser,
        )
        _ = nlp_controller.reset_vector_db_collection(project=project)

    for asset_id, file_id in project_files_ids.items():

        # ephemeral server disk: bring the file back from the database copy if needed
        if get_settings().STORE_FILES_IN_DB and process_controller.get_file_path(file_id) is None:
            file_store = await FileStoreModel.create_instance(db_client=request.app.db_client)
            restored = await file_store.restore_file(
                project_id=project_id, file_id=file_id,
                target_path=os.path.join(process_controller.project_path, file_id))
            if not restored:
                logger.error(f"File not found locally nor in the database store: {file_id}")

        file_content = process_controller.get_file_content(file_id=file_id)

        if file_content is None:
            logger.error(f"Error while processing file: {file_id}")
            await asset_model.update_asset_processing(asset_id, {
                "status": "failed", "chunks": 0, "error": "file could not be loaded",
                "processed_at": datetime.utcnow().isoformat()})
            continue

        file_chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=file_id,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            asset_config=project_files_configs.get(asset_id),
        )

        if file_chunks is None or len(file_chunks) == 0:
            # e.g. a scanned PDF with no extractable text: skip it, keep processing the others
            logger.error(f"No chunks produced for file: {file_id}")
            await asset_model.update_asset_processing(asset_id, {
                "status": "failed", "chunks": 0, "error": "no extractable text (scanned or empty PDF?)",
                "processed_at": datetime.utcnow().isoformat()})
            continue

        # re-processing a file REPLACES its chunks (previously they were duplicated)
        _ = await chunk_model.delete_chunks_by_asset_id(project_id=project.id, asset_id=asset_id)

        file_chunks_records = [
            DataChunk(
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata,
                chunk_order=i+1,
                chunk_project_id=project.id,
                chunk_asset_id=asset_id
            )
            for i, chunk in enumerate(file_chunks)
        ]

        inserted = await chunk_model.insert_many_chunks(chunks=file_chunks_records)
        no_records += inserted
        no_files += 1
        await asset_model.update_asset_processing(asset_id, {
            "status": "processed", "chunks": inserted, "error": None,
            "processed_at": datetime.utcnow().isoformat()})

    if no_files == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROCESSING_FAILED.value
            }
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.PROCESSING_SUCCESS.value,
            "inserted_chunks": no_records,
            "processed_files": no_files
        }
    )

def _indexing_status(chunks: int, indexed_points: int):
    if chunks == 0:
        return "not_indexed" if indexed_points == 0 else "stale"
    if indexed_points == chunks:
        return "indexed"
    return "not_indexed" if indexed_points == 0 else "out_of_date"


@data_router.get("/assets/{project_id}", dependencies=[Depends(require_admin)])
async def list_assets(request: Request, project_id: str):
    """Course materials of a project with their processing / indexing status (admin only).
    Status is computed from the real stores: MongoDB chunks per asset and Qdrant points per file_id."""

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)

    assets = await asset_model.get_all_project_assets(asset_project_id=project.id,
                                                      asset_type=AssetTypeEnum.FILE.value)
    chunk_counts = await chunk_model.count_chunks_by_asset(project_id=project.id)

    vectordb_client = request.app.vectordb_client
    collection_name = NLPController(vectordb_client=vectordb_client, generation_client=None,
                                    embedding_client=None, template_parser=None
                                    ).create_collection_name(project_id=project.project_id)

    items = []
    for asset in assets:
        config = asset.asset_config or {}
        processing = asset.asset_processing or {}
        chunks = chunk_counts.get(asset.id, 0)
        indexed_points = vectordb_client.count_by_metadata(collection_name, {"file_id": asset.asset_name})

        if processing.get("status"):
            processing_status = processing["status"]
        else:  # assets processed before status tracking existed
            processing_status = "processed" if chunks > 0 else "pending"

        items.append({
            "file_id": asset.asset_name,
            "document_name": config.get("original_name") or ProjectController().get_original_file_name(asset.asset_name),
            "doc_type": config.get("doc_type") or "other",
            "course": config.get("course"),
            "topic": config.get("topic"),
            "size_bytes": asset.asset_size,
            "uploaded_at": asset.asset_pushed_at.isoformat() if asset.asset_pushed_at else None,
            "processing_status": processing_status,
            "processing_error": processing.get("error"),
            "processed_at": processing.get("processed_at"),
            "chunks": chunks,
            "indexed_points": indexed_points,
            "indexing_status": _indexing_status(chunks, indexed_points),
        })

    total_chunks = sum(chunk_counts.values())
    total_points = vectordb_client.count_by_metadata(collection_name, None) \
        if vectordb_client.is_collection_existed(collection_name) else 0

    summary = {
        "total_documents": len(items),
        "lectures": sum(i["doc_type"] == "lecture" for i in items),
        "labs": sum(i["doc_type"] == "lab" for i in items),
        "other": sum(i["doc_type"] == "other" for i in items),
        "indexed_documents": sum(i["indexing_status"] == "indexed" for i in items),
        "awaiting_processing": sum(i["processing_status"] == "pending" for i in items),
        "awaiting_indexing": sum(i["processing_status"] == "processed" and i["indexing_status"] != "indexed"
                                 for i in items),
        "processing_errors": sum(i["processing_status"] == "failed" for i in items),
        "total_chunks": total_chunks,
        "total_vectors": total_points,
        # vectors that do not belong to current chunks (e.g. after deleting / re-processing a file)
        "index_needs_rebuild": total_points != total_chunks,
    }

    return JSONResponse(content={
        "signal": ResponseSignal.ASSETS_RETRIEVED.value,
        "project_id": project.project_id,
        "summary": summary,
        "assets": items,
    })


@data_router.delete("/asset/{project_id}/{file_id}", dependencies=[Depends(require_admin)])
async def delete_asset(request: Request, project_id: str, file_id: str):
    """Delete ONE course material and everything derived from it (admin only):
    its Qdrant points (metadata.file_id), its MongoDB chunks (chunk_asset_id), its file, its asset record.
    Other documents and the collection itself are not touched."""

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    asset = await asset_model.get_asset_record(asset_project_id=project.id, asset_name=file_id)
    if asset is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND,
                            content={"signal": ResponseSignal.FILE_ID_ERROR.value})

    vectordb_client = request.app.vectordb_client
    collection_name = NLPController(vectordb_client=vectordb_client, generation_client=None,
                                    embedding_client=None, template_parser=None
                                    ).create_collection_name(project_id=project.project_id)
    try:
        # 1. vectors of this file only
        vectors = vectordb_client.count_by_metadata(collection_name, {"file_id": asset.asset_name})
        if not vectordb_client.delete_by_metadata(collection_name, {"file_id": asset.asset_name}):
            raise RuntimeError("vector deletion refused")

        # 2. chunks of this asset only
        chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)
        chunks = await chunk_model.delete_chunks_by_asset_id(project_id=project.id, asset_id=asset.id)

        # 3. the stored file, only if it resolves inside this project's folder
        project_dir = os.path.realpath(ProjectController().get_project_path(project_id=project_id))
        stored_file = os.path.realpath(os.path.join(project_dir, asset.asset_name))
        file_removed = False
        if os.path.dirname(stored_file) == project_dir and os.path.isfile(stored_file):
            os.remove(stored_file)
            file_removed = True

        # 3b. the persistent database copy of this file (if enabled)
        if get_settings().STORE_FILES_IN_DB:
            file_store = await FileStoreModel.create_instance(db_client=request.app.db_client)
            await file_store.delete_file(project_id=project_id, file_id=asset.asset_name)

        # 4. the asset record
        await asset_model.delete_asset(asset_id=asset.id)
    except Exception as e:
        logger.error(f"Error while deleting asset {file_id}: {e}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={"signal": ResponseSignal.ASSET_DELETE_FAILED.value})

    return JSONResponse(content={
        "signal": ResponseSignal.ASSET_DELETED.value,
        "file_id": file_id,
        "deleted_chunks": chunks,
        "deleted_vectors": vectors,
        "file_removed": file_removed,
    })

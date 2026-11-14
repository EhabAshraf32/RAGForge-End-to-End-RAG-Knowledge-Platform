from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import ProcessingEnum

class ProcessController(BaseController):

    def __init__(self, project_id: str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    # def get_file_loader(self, file_id: str):

    #     file_ext = self.get_file_extension(file_id=file_id)
    #     file_path = os.path.join(
    #         self.project_path,
    #         file_id
    #     )
    #     if not os.path.exists(file_path):
    #         return None

    #     if file_ext == ProcessingEnum.TXT.value:
    #         return TextLoader(file_path, encoding="utf-8")

    #     if file_ext == ProcessingEnum.PDF.value:
    #         return PyMuPDFLoader(file_path)
        
    #     return None

    def get_file_path(self, file_id: str):

        # Exact match
        exact_path = os.path.join(
            self.project_path,
            file_id
        )

        if os.path.isfile(exact_path):
            return exact_path

        # Match file_id prefix
        if not os.path.isdir(self.project_path):
            return None

        for filename in os.listdir(self.project_path):

            if filename.startswith(f"{file_id}_"):
                return os.path.join(
                    self.project_path,
                    filename
                )

        return None


    def get_file_loader(self, file_id: str):

        file_path = self.get_file_path(file_id)

        if not file_path:
            return None

        file_ext = os.path.splitext(file_path)[-1].lower()

        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(
                file_path,
                encoding="utf-8"
            )

        if file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)

        return None

    def get_file_content(self, file_id: str):

        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.load()
        return None
    # Split priority (code-aware for C course material):
    # paragraph / blank line -> end of a "}" block -> end of a ";" statement -> line -> word.
    # keep_separator="end" keeps "}" / ";" attached to the code they close.
    # Splitting is done per loader Document (= per PDF page), so page metadata stays exact.
    CHUNK_SEPARATORS = ["\n\n", "}\n", ";\n", "\n", " ", ""]

    def get_chunk_metadata(self, loader_metadata: dict, file_id: str, asset_config: dict=None):
        """
        Build the metadata stored with every chunk (MongoDB chunk_metadata -> Qdrant payload).
        Course fields come from Asset.asset_config (set at upload); page comes from the loader.
        Loader keys are kept, except the absolute server paths (`source`, `file_path`),
        which are environment specific and are replaced by `file_id` / `document_name`.
        """
        asset_config = asset_config or {}
        metadata = {
            k: v for k, v in (loader_metadata or {}).items()
            if k not in ("source", "file_path")
        }

        page = metadata.get("page")
        metadata.update({
            "file_id": file_id,
            "document_name": asset_config.get("original_name") or self.get_original_file_name(file_id),
            "doc_type": asset_config.get("doc_type") or "other",
            "course": asset_config.get("course"),
            "topic": asset_config.get("topic"),
            # PyMuPDF pages are 0-based; page_number is the human (1-based) page
            "page_number": page + 1 if isinstance(page, int) else None,
        })
        return metadata

    def process_file_content(self, file_content: list, file_id: str,
                            chunk_size: int=1000, overlap_size: int=200,
                            asset_config: dict=None):

        text_splitter = RecursiveCharacterTextSplitter(
            separators=self.CHUNK_SEPARATORS,
            keep_separator="end",
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
        )

        file_content_texts = [
            rec.page_content
            for rec in file_content
        ]

        file_content_metadata = [
            self.get_chunk_metadata(
                loader_metadata=rec.metadata,
                file_id=file_id,
                asset_config=asset_config,
            )
            for rec in file_content
        ]

        chunks = text_splitter.create_documents(
            file_content_texts,
            metadatas=file_content_metadata
        )

        return chunks


    


from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json
import logging
import re

class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client, 
                 embedding_client,template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.logger = logging.getLogger(__name__)

    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List[int], 
                                   do_reset: bool = False):
        
        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # guard: an existing collection built with another embedding model cannot be reused
        existing_size = self.vectordb_client.get_collection_vector_size(collection_name=collection_name)
        if not do_reset and existing_size and existing_size != self.embedding_client.embedding_size:
            self.logger.error(
                f"Collection {collection_name} has vector size {existing_size} but the embedding model "
                f"produces {self.embedding_client.embedding_size}. Push with do_reset=1 (or use a new project) to re-index."
            )
            return False

        # step2: manage items
        texts = [ c.chunk_text for c in chunks ]
        # keep a link back to the MongoDB chunk in the vector payload
        metadata = [ {**(c.chunk_metadata or {}), "chunk_id": str(c.id) if c.id else None} for c in chunks ]
        vectors = [
            self.embedding_client.embed_text(text=text, 
                                             document_type=DocumentTypeEnum.DOCUMENT.value)
            for text in texts
        ]

        if any(not v for v in vectors):
            self.logger.error("Embedding failed for one or more chunks; aborting vector db insert")
            return False

        # step3: create collection if not exists
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # step4: insert into vector db
        is_inserted = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )

        return bool(is_inserted)

    def search_vector_db_collection(self, project: Project, text: str, limit: int = 10,
                                          doc_type: str = None):

        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: get text embedding vector
        vector = self.embedding_client.embed_text(text=text,
                                                 document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            return False

        # step3: do semantic search (optionally only lecture / lab / other chunks)
        filter_metadata = {"doc_type": doc_type} if doc_type else None
        try:
            results = self.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=vector,
                limit=limit,
                filter_metadata=filter_metadata,
            )
        except Exception as e:
            # e.g. collection not indexed yet, or vector size mismatch after changing the embedding model
            self.logger.error(f"Error while searching collection {collection_name}: {e}")
            return False

        if not results:
            return False

        return results

    #######################################################
    # Educational RAG helpers
    #######################################################

    USED_DOCUMENTS_PATTERN = re.compile(r"\[\[\s*USED_DOCUMENTS\s*:\s*([^\]]*)\]\]", re.IGNORECASE)

    def detect_query_language(self, text: str):
        """'ar' when the question contains Arabic words (code blocks ignored), otherwise 'en'."""
        text_without_code = re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)
        return "ar" if re.search(r"[؀-ۿ]{2,}", text_without_code) else "en"

    def get_document_source(self, document):
        """Normalized source info built ONLY from the retrieved chunk metadata
        (works for new chunks and for chunks indexed before course metadata existed)."""
        metadata = getattr(document, "metadata", None) or {}

        page_number = metadata.get("page_number")
        if page_number is None and isinstance(metadata.get("page"), int):
            page_number = metadata["page"] + 1

        document_name = metadata.get("document_name") or self.get_original_file_name(
            metadata.get("file_id") or metadata.get("source")
        )

        return {
            "document_name": document_name or "unknown",
            "doc_type": metadata.get("doc_type") or "other",
            "page_number": page_number,
            "course": metadata.get("course"),
            "topic": metadata.get("topic"),
        }

    def get_sources(self, documents: list, with_score: bool = False):
        """De-duplicated (document, page) sources in retrieval order."""
        sources, seen = [], set()
        for doc in documents:
            source = self.get_document_source(doc)
            key = (source["document_name"], source["page_number"], source["doc_type"])
            if key in seen:
                continue
            seen.add(key)
            if with_score:
                source["score"] = round(float(doc.score), 4)
            sources.append(source)
        return sources

    def extract_used_documents(self, answer: str, retrieved_documents: list):
        """Strip the [[USED_DOCUMENTS: ...]] marker from the answer and return the cited documents.
        Only numbers that exist in the retrieved list are accepted, so a source can never be invented.
        If the model did not write the marker, all retrieved documents are returned."""
        if not answer:
            return answer, list(retrieved_documents)

        matches = self.USED_DOCUMENTS_PATTERN.findall(answer)
        clean_answer = self.USED_DOCUMENTS_PATTERN.sub("", answer).strip()

        if not matches:
            return clean_answer, list(retrieved_documents)

        numbers = [int(n) for n in re.findall(r"\d+", matches[-1])]
        used = [retrieved_documents[n - 1] for n in dict.fromkeys(numbers)
                if 1 <= n <= len(retrieved_documents)]
        return clean_answer, used

    def get_history_messages(self, chat_history: list):
        """Convert client-side history [{role, content}] into provider messages.
        Only user/assistant roles are accepted (clients cannot inject system prompts)."""
        if not chat_history:
            return []

        max_messages = self.app_settings.RAG_HISTORY_MAX_MESSAGES or 0
        max_characters = self.app_settings.RAG_HISTORY_MAX_CHARACTERS
        role_map = {
            "user": self.generation_client.enums.USER.value,
            "assistant": self.generation_client.enums.ASSISTANT.value,
        }

        messages = []
        for turn in chat_history[-max_messages:] if max_messages else []:
            role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", None)
            content = turn.get("content") if isinstance(turn, dict) else getattr(turn, "content", None)
            if role not in role_map or not content:
                continue
            messages.append(self.generation_client.construct_prompt(
                prompt=content[:max_characters] if max_characters else content,
                role=role_map[role],
            ))
        return messages

    def retrieve_documents(self, project: Project, query: str, limit: int,
                                 doc_type: str = None, chat_history: list = None):
        """Retrieve for the question; for follow-ups also retrieve with the previous user question
        prepended (cheap contextual retrieval, no extra LLM call) and merge by best score."""
        results = self.search_vector_db_collection(project=project, text=query, limit=limit, doc_type=doc_type) or []

        previous_questions = [
            (t.get("content") if isinstance(t, dict) else getattr(t, "content", None))
            for t in (chat_history or [])
            if (t.get("role") if isinstance(t, dict) else getattr(t, "role", None)) == "user"
        ]
        previous_questions = [q for q in previous_questions if q]

        if previous_questions:
            contextual_query = f"{previous_questions[-1]}\n{query}"
            contextual_results = self.search_vector_db_collection(
                project=project, text=contextual_query, limit=limit, doc_type=doc_type) or []

            merged = {}
            for doc in list(results) + list(contextual_results):
                if doc.text not in merged or doc.score > merged[doc.text].score:
                    merged[doc.text] = doc
            results = sorted(merged.values(), key=lambda d: d.score, reverse=True)[:limit]

        return results

    def answer_rag_question(self, project: Project, query: str, limit: int = 10,
                                  doc_type: str = None, chat_history: list = None):
        """
        Returns (answer, full_prompt, chat_history, sources)
          - sources: {"cited": [...], "retrieved": [...]} built by the backend from retrieved metadata.
        `chat_history` (optional) is the client-side conversation: [{"role": "user"|"assistant", "content": str}]
        """
        answer, full_prompt, messages, sources = None, None, None, None

        # step1: retrieve related documents
        retrieved_documents = self.retrieve_documents(
            project=project,
            query=query,
            limit=limit,
            doc_type=doc_type,
            chat_history=chat_history,
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, messages, sources

        # step2: Construct LLM prompt (Arabic or English templates, following the student's language)
        language = self.detect_query_language(query)
        system_prompt = self.template_parser.get("rag", "system_prompt", language=language)

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "doc_type": source["doc_type"],
                    "doc_name": source["document_name"],
                    "page": source["page_number"] if source["page_number"] is not None else "n/a",
                    "chunk_text": doc.text,
            }, language=language)
            for idx, (doc, source) in enumerate(
                (d, self.get_document_source(d)) for d in retrieved_documents
            )
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        }, language=language)

        # step3: Construct Generation Client Prompts (system + previous turns)
        messages = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ] + self.get_history_messages(chat_history)

        full_prompt = "\n\n".join([ documents_prompts,  footer_prompt])

        # step4: Retrieve the Answer
        raw_answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=messages
        )

        if not raw_answer:
            return None, full_prompt, messages, None

        # step5: sources come from retrieved metadata, never from LLM text
        answer, used_documents = self.extract_used_documents(raw_answer, retrieved_documents)
        sources = {
            "cited": self.get_sources(used_documents),
            "retrieved": self.get_sources(retrieved_documents, with_score=True),
        }

        return answer, full_prompt, messages, sources


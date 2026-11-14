from ..LLMInterface import LLMInterface
from ..LLMEnums import ChatRoleEnums, DocumentTypeEnum
import logging
import os


class HuggingFaceProvider(LLMInterface):
    """
    Local Hugging Face provider.
    `sentence-transformers` / `transformers` are imported lazily inside the setters,
    so they are only required when GENERATION_BACKEND / EMBEDDING_BACKEND = "HUGGINGFACE".
    """

    def __init__(
        self,
        default_input_max_characters: int=1000,
        default_generation_max_output_tokens: int=1000,
        default_generation_temperature: float=0.1,
        default_generation_input_max_characters: int=20000,
    ):

        # embedding input limit
        self.default_input_max_characters = default_input_max_characters
        # generation prompt (per message) limit
        self.default_generation_input_max_characters = default_generation_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.embedding_model = None
        self.generation_model = None

        self.embedding_model_id = None
        self.embedding_size = None
        self.generation_model_id = None

        # HF chat templates use the same role names as OpenAI ("system" / "user" / "assistant")
        self.enums = ChatRoleEnums

        self.logger = logging.getLogger(__name__)

    def process_text(self, text: str, max_characters: int=None):
        max_characters = max_characters if max_characters else self.default_input_max_characters
        return text[:max_characters].strip()

    #######################################################
    # Embeddings
    #######################################################

    def set_embedding_model(self, model_id: str, embedding_size: int = None):
        self.embedding_model_id = model_id

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.logger.error("sentence-transformers is not installed (required for HUGGINGFACE embeddings)")
            return

        # CPU by default; set HF_EMBEDDING_DEVICE=cuda to use a GPU
        self.embedding_model = SentenceTransformer(model_id, device=os.environ.get("HF_EMBEDDING_DEVICE", "cpu"))

        if embedding_size:
            self.embedding_size = embedding_size
        else:
            get_dim = getattr(self.embedding_model, "get_sentence_embedding_dimension", None)
            self.embedding_size = get_dim() if get_dim else None

    def embed_text(self, text: str, document_type: str = None):

        if self.embedding_model is None:
            self.logger.error("Embedding model for HuggingFace was not set")
            return None

        # instruction-tuned models (e.g. Qwen3-Embedding) define a "query" prompt for search queries;
        # models without prompts (e.g. bge-m3) embed queries and documents the same way
        encode_kwargs = {}
        prompts = getattr(self.embedding_model, "prompts", None) or {}
        if document_type == DocumentTypeEnum.QUERY.value and "query" in prompts:
            encode_kwargs["prompt_name"] = "query"

        try:
            embedding = self.embedding_model.encode(
                self.process_text(text),
                convert_to_numpy=True,
                normalize_embeddings=True,
                **encode_kwargs,
            )
        except Exception as e:
            self.logger.error(f"Error while embedding text with HuggingFace: {type(e).__name__}: {e}")
            return None

        return embedding.tolist()

    #######################################################
    # Generation
    #######################################################

    def set_generation_model(self, model_id: str):

        self.generation_model_id = model_id

        try:
            from transformers import pipeline
        except ImportError:
            self.logger.error("transformers is not installed (required for HUGGINGFACE generation)")
            return

        self.generation_model = pipeline(
            "text-generation",
            model=model_id,
        )

    def generate_text(self, prompt: str, chat_history: list=None, max_output_tokens: int=None,
                            temperature: float = None):

        if self.generation_model is None:
            self.logger.error("Generation model for HuggingFace was not set")
            return None

        if chat_history is None:
            chat_history = []

        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history.append(
            self.construct_prompt(prompt=prompt, role=ChatRoleEnums.USER.value)
        )

        try:
            output = self.generation_model(
                chat_history,
                max_new_tokens=max_output_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False,
            )
        except Exception as e:
            self.logger.error(f"Error while generating text with HuggingFace: {type(e).__name__}: {e}")
            return None

        if not output or "generated_text" not in output[0]:
            self.logger.error("Error while generating text with HuggingFace")
            return None

        generated = output[0]["generated_text"]

        # some transformers versions return the whole conversation as a list of messages
        if isinstance(generated, list):
            generated = generated[-1].get("content") if generated else None

        return generated

    #######################################################

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt, max_characters=self.default_generation_input_max_characters),
        }

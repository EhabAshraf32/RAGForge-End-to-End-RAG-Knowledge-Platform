from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI
import logging

class OpenAIProvider(LLMInterface):

    def __init__(self, api_key: str, base_url: str=None,
                       default_input_max_characters: int=1000,
                       default_generation_max_output_tokens: int=1000,
                       default_generation_temperature: float=0.1,
                       default_generation_input_max_characters: int=20000,
                       timeout: float=120,
                       max_retries: int=2):

        self.api_key = api_key
        # base_url=None -> api.openai.com ; e.g. "https://openrouter.ai/api/v1" -> OpenRouter
        self.base_url = base_url

        # embedding input limit
        self.default_input_max_characters = default_input_max_characters
        # generation prompt (per message) limit
        self.default_generation_input_max_characters = default_generation_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client = OpenAI(
            api_key = self.api_key,
            base_url = self.base_url,
            timeout = timeout,
            # inside a fallback chain a rate-limited provider should hand over immediately
            max_retries = max_retries,
        )

        self.enums=OpenAIEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str, max_characters: int=None):
        max_characters = max_characters if max_characters else self.default_input_max_characters
        return text[:max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list=None, max_output_tokens: int=None,
                            temperature: float = None):

        if chat_history is None:
            chat_history = []

        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history.append(
            self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value)
        )

        try:
            response = self.client.chat.completions.create(
                model = self.generation_model_id,
                messages = chat_history,
                max_tokens = max_output_tokens,
                temperature = temperature
            )
        except Exception as e:
            # e.g. 402 insufficient credits, 429 rate limit, timeout, unknown model id
            self.logger.error(f"Error while generating text with OpenAI-compatible API ({self.base_url}): "
                              f"{type(e).__name__}: {e}")
            return None

        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("Error while generating text with OpenAI")
            return None

        content = response.choices[0].message.content
        if not content:
            # e.g. a reasoning model spent the whole max_tokens budget before answering
            self.logger.error(f"Empty completion (finish_reason={response.choices[0].finish_reason})")
            return None

        return content


    def embed_text(self, text: str, document_type: str = None):
        
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None
        
        try:
            response = self.client.embeddings.create(
                model = self.embedding_model_id,
                input = self.process_text(text),
            )
        except Exception as e:
            self.logger.error(f"Error while embedding text with OpenAI-compatible API: {type(e).__name__}: {e}")
            return None

        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error("Error while embedding text with OpenAI")
            return None

        return response.data[0].embedding

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt, max_characters=self.default_generation_input_max_characters)
        }
    


    


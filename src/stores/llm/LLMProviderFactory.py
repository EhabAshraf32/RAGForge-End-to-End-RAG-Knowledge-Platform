
from .LLMEnums import LLMEnums
from .providers import OpenAIProvider, CoHereProvider, HuggingFaceProvider

class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str):
        if provider == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key = self.config.OPENAI_API_KEY,
                base_url = self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE,
                default_generation_input_max_characters=self.config.GENERATION_DAFAULT_MAX_INPUT_CHARACTERS,
                timeout=self.config.OPENAI_API_TIMEOUT,
            )

        if provider == LLMEnums.COHERE.value:
            return CoHereProvider(
                api_key = self.config.COHERE_API_KEY,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE,
                default_generation_input_max_characters=self.config.GENERATION_DAFAULT_MAX_INPUT_CHARACTERS,
            )

        if provider == LLMEnums.HUGGINGFACE.value:
            return HuggingFaceProvider(
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE,
                default_generation_input_max_characters=self.config.GENERATION_DAFAULT_MAX_INPUT_CHARACTERS,
            )

        if provider == LLMEnums.CLOUDFLARE.value:
            # Workers AI exposes an OpenAI-compatible API, so the existing OpenAIProvider is reused
            if not self.config.CLOUDFLARE_ACCOUNT_ID or not self.config.CLOUDFLARE_API_TOKEN:
                return None
            return OpenAIProvider(
                api_key = self.config.CLOUDFLARE_API_TOKEN,
                base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.config.CLOUDFLARE_ACCOUNT_ID}/ai/v1",
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE,
                default_generation_input_max_characters=self.config.GENERATION_DAFAULT_MAX_INPUT_CHARACTERS,
                timeout=self.config.OPENAI_API_TIMEOUT,
            )

        # NOTE: OpenRouter / any OpenAI-compatible gateway is already supported through
        # GENERATION_BACKEND="OPENAI" + OPENAI_API_URL (passed as base_url above).
        # if provider == LLMEnums.OPENROUTER.value:
        #   return OpenAIProvider(
        #         api_key = self.config.OPENAI_API_KEY,
        #         base_url = self.config.OPENROTER_API_URL,
        #         default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
        #         default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
        #         default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
        #     )

        return None

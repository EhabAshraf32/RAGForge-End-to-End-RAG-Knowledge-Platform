
from .LLMEnums import LLMEnums
from .providers import OpenAIProvider, CoHereProvider, HuggingFaceProvider, FallbackProvider

class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str, max_retries: int = 2):
        if provider == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key = self.config.OPENAI_API_KEY,
                base_url = self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE,
                default_generation_input_max_characters=self.config.GENERATION_DAFAULT_MAX_INPUT_CHARACTERS,
                timeout=self.config.OPENAI_API_TIMEOUT,
                max_retries=max_retries,
            )

        if provider in (LLMEnums.GROQ.value, LLMEnums.GEMINI.value):
            # OpenAI-compatible free tiers
            key, base_url = {
                LLMEnums.GROQ.value: (getattr(self.config, "GROQ_API_KEY", None), "https://api.groq.com/openai/v1"),
                LLMEnums.GEMINI.value: (getattr(self.config, "GEMINI_API_KEY", None),
                                        "https://generativelanguage.googleapis.com/v1beta/openai"),
            }[provider]
            if not key:
                return None
            return OpenAIProvider(
                api_key = key,
                base_url = base_url,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE,
                default_generation_input_max_characters=self.config.GENERATION_DAFAULT_MAX_INPUT_CHARACTERS,
                timeout=self.config.OPENAI_API_TIMEOUT,
                max_retries=max_retries,
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

    def create_generation_client(self, backend: str, model_id: str, fallbacks: str = None):
        """The generation client used by the app: a single provider, or - when GENERATION_FALLBACKS is set -
        a FallbackProvider trying `backend:model_id` first and then each `BACKEND:MODEL_ID[#DAILY_CAP]` in order.
        `#N` limits that provider to N answers per UTC day (e.g. to protect a quota shared with embeddings)."""
        specs = [(backend, model_id, None)]
        for item in (fallbacks or "").split(","):
            item = item.strip()
            if item:
                fb_backend, _, fb_model = item.partition(":")
                fb_model, _, cap = fb_model.partition("#")
                specs.append((fb_backend.strip().upper(), fb_model.strip(), int(cap) if cap.strip() else None))

        if len(specs) == 1:
            client = self.create(provider=backend)
            if client is not None:
                client.set_generation_model(model_id=model_id)
            return client

        providers, names, caps = [], [], []
        for spec_backend, spec_model, cap in specs:
            client = self.create(provider=spec_backend, max_retries=0)  # hand over at once on 429
            if client is None or not spec_model:
                continue
            client.set_generation_model(model_id=spec_model)
            providers.append(client)
            names.append(f"{spec_backend}:{spec_model}")
            caps.append(cap)
        return FallbackProvider(providers, names, daily_caps=caps) if providers else None

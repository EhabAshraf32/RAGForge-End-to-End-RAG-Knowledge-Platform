from ..LLMInterface import LLMInterface
import logging
import time


class FallbackProvider(LLMInterface):
    """
    Generation through an ordered chain of providers (e.g. several free tiers).
    The first provider that returns an answer wins; a provider that fails (rate limit, quota, timeout)
    is skipped for `cooldown_seconds` so later questions do not wait on it again.
    All providers in a chain must share the same message format (OpenAI-compatible chat).
    """

    def __init__(self, providers: list, names: list = None, cooldown_seconds: int = 60, daily_caps: list = None):
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self.providers = providers
        self.names = names or [f"provider-{i}" for i in range(len(providers))]
        self.cooldown_seconds = cooldown_seconds
        # optional max answers per UTC day per provider (None = unlimited); counted per process
        self.daily_caps = list(daily_caps) if daily_caps else [None] * len(providers)
        self._used_today = {}
        self._failed_at = {}
        self.last_used = None
        self.enums = providers[0].enums
        self.logger = logging.getLogger(__name__)

    # generation models are configured per provider when the chain is built
    def set_generation_model(self, model_id: str):
        pass

    def set_embedding_model(self, model_id: str, embedding_size: int):
        raise NotImplementedError("FallbackProvider is for generation only")

    def embed_text(self, text: str, document_type: str = None):
        return None

    def construct_prompt(self, prompt: str, role: str):
        return self.providers[0].construct_prompt(prompt=prompt, role=role)

    def _cooling_down(self, index: int):
        failed = self._failed_at.get(index)
        return failed is not None and time.monotonic() - failed < self.cooldown_seconds

    @staticmethod
    def _today():
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _cap_reached(self, index: int):
        cap = self.daily_caps[index]
        return cap is not None and self._used_today.get((self._today(), index), 0) >= cap

    def generate_text(self, prompt: str, chat_history: list = None, max_output_tokens: int = None,
                      temperature: float = None):
        history = list(chat_history or [])
        allowed = [i for i in range(len(self.providers)) if not self._cap_reached(i)]   # caps are never bypassed
        candidates = [i for i in allowed if not self._cooling_down(i)]
        # if every allowed provider is cooling down, still try them rather than failing immediately
        for index in candidates or allowed:
            answer = self.providers[index].generate_text(
                prompt=prompt, chat_history=list(history),     # a fresh copy per attempt
                max_output_tokens=max_output_tokens, temperature=temperature,
            )
            if self.daily_caps[index] is not None:   # count every attempt against a capped quota
                key = (self._today(), index)
                self._used_today = {k: v for k, v in self._used_today.items() if k[0] == key[0]}
                self._used_today[key] = self._used_today.get(key, 0) + 1
            if answer:
                self._failed_at.pop(index, None)
                self.last_used = self.names[index]
                if chat_history is not None:   # same side effect as a single provider
                    chat_history.append(self.construct_prompt(prompt=prompt, role=self.enums.USER.value))
                return answer
            self._failed_at[index] = time.monotonic()
            self.logger.warning(f"Generation provider {self.names[index]} failed; trying the next one")
        self.last_used = None
        self.logger.error("All generation providers failed")
        return None

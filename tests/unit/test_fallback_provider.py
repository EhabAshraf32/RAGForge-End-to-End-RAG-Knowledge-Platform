"""Generation fallback chain (several free tiers tried in order)."""
from types import SimpleNamespace

import pytest

from conftest import nlp_settings
from stores.llm.LLMEnums import OpenAIEnums
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.providers import FallbackProvider, OpenAIProvider


class Fake:
    enums = OpenAIEnums

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def construct_prompt(self, prompt, role):
        return {"role": role, "content": prompt}

    def generate_text(self, prompt, chat_history=None, max_output_tokens=None, temperature=None):
        chat_history.append(self.construct_prompt(prompt, "user"))  # providers mutate their copy
        self.calls.append(list(chat_history))
        return self.answers.pop(0) if self.answers else None


def test_first_success_wins_and_history_is_not_duplicated():
    a, b = Fake([None]), Fake(["from b"])
    chain = FallbackProvider([a, b], ["A", "B"])
    history = [{"role": "system", "content": "sys"}]
    assert chain.generate_text("q", chat_history=history) == "from b"
    assert chain.last_used == "B"
    assert [m["role"] for m in b.calls[0]] == ["system", "user"]       # b got a clean copy
    assert [m["role"] for m in history] == ["system", "user"]          # caller sees one user turn


def test_failed_provider_is_skipped_during_cooldown(monkeypatch):
    import sys
    import stores.llm.providers  # noqa: F401
    mod = sys.modules["stores.llm.providers.FallbackProvider"]
    now = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: now[0])
    a, b = Fake([None, "a later"]), Fake(["b1", "b2"])
    chain = FallbackProvider([a, b], ["A", "B"], cooldown_seconds=60)
    assert chain.generate_text("q1", chat_history=[]) == "b1"
    assert chain.generate_text("q2", chat_history=[]) == "b2"
    assert len(a.calls) == 1                     # A not retried while cooling down
    now[0] += 61
    assert chain.generate_text("q3", chat_history=[]) == "a later"


def test_all_fail_returns_none():
    chain = FallbackProvider([Fake([None]), Fake([None])])
    assert chain.generate_text("q", chat_history=[]) is None and chain.last_used is None


def test_factory_builds_chain_with_groq_and_openrouter():
    cfg = nlp_settings(GROQ_API_KEY="gsk-test")
    f = LLMProviderFactory(cfg)
    chain = f.create_generation_client("GROQ", "openai/gpt-oss-120b",
                                       "GROQ:qwen/qwen3.8-27b, OPENAI:nvidia/nemotron-3.5-super-vl:free")
    assert isinstance(chain, FallbackProvider)
    assert chain.names == ["GROQ:openai/gpt-oss-120b", "GROQ:qwen/qwen3.8-27b",
                           "OPENAI:nvidia/nemotron-3.5-super-vl:free"]   # ':' inside the model id is kept
    groq = chain.providers[0]
    assert str(groq.client.base_url).rstrip("/") == "https://api.groq.com/openai/v1"
    assert groq.client.max_retries == 0 and chain.providers[2].generation_model_id == "nvidia/nemotron-3.5-super-vl:free"


def test_factory_single_provider_without_fallbacks():
    client = LLMProviderFactory(nlp_settings()).create_generation_client("OPENAI", "m1", None)
    assert isinstance(client, OpenAIProvider) and client.generation_model_id == "m1"
    assert client.client.max_retries == 2


def test_unconfigured_fallback_is_skipped():
    chain = LLMProviderFactory(nlp_settings(GROQ_API_KEY=None)).create_generation_client(
        "OPENAI", "m1", "GROQ:some-model")
    assert chain.names == ["OPENAI:m1"]

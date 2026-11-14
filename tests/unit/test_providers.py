"""LLM / embedding provider unit tests (no network: fake clients or httpx MockTransport)."""
import json
import sys
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from conftest import nlp_settings
from stores.llm.LLMEnums import DocumentTypeEnum
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.providers import OpenAIProvider, CoHereProvider, HuggingFaceProvider


# --------------------------------------------------------------------------- fakes
class FakeOpenAIClient:
    def __init__(self, content="ok", raise_exc=None):
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(("chat", kw))
                if raise_exc:
                    raise raise_exc
                msg = SimpleNamespace(content=content)
                return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

        class _Embeddings:
            def create(self, **kw):
                outer.calls.append(("embed", kw))
                return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

        self.chat = SimpleNamespace(completions=_Completions())
        self.embeddings = _Embeddings()


class FakeCohereClient:
    def __init__(self):
        self.calls = []

    def embed(self, **kw):
        self.calls.append(("embed", kw))
        return SimpleNamespace(embeddings=SimpleNamespace(float=[[0.1, 0.2]]))

    def chat(self, **kw):
        self.calls.append(("chat", kw))
        return SimpleNamespace(text="ok")


class FakeSentenceTransformer:
    def __init__(self, prompts=None, dim=8):
        self.prompts = prompts or {}
        self.dim = dim
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, text, **kw):
        self.calls.append((text, kw))
        v = np.ones(self.dim, dtype=np.float32)
        return v / np.linalg.norm(v)


LONG_CONTEXT = "x" * 5000 + "\n## Question:\nWHAT_IS_A_POINTER"


def make_cohere(**kw):
    """CoHereProvider imports `cohere` lazily; give it a stub module so no SDK/network is needed."""
    stub = SimpleNamespace(Client=lambda api_key: FakeCohereClient())
    sys.modules.setdefault("cohere", stub)
    p = CoHereProvider(api_key="k", **kw)
    p.client = FakeCohereClient()
    return p


# --------------------------------------------------------------------------- B1 separate limits
def test_b1_openai_generation_prompt_not_truncated_but_embedding_is():
    p = OpenAIProvider(api_key="k", default_input_max_characters=1024,
                       default_generation_input_max_characters=20000)
    p.client = FakeOpenAIClient()
    p.set_generation_model("m")
    p.set_embedding_model("e", 2)

    p.generate_text(prompt=LONG_CONTEXT, chat_history=[])
    assert "WHAT_IS_A_POINTER" in p.client.calls[0][1]["messages"][-1]["content"]

    p.embed_text("y" * 5000, document_type=DocumentTypeEnum.DOCUMENT.value)
    assert len(p.client.calls[1][1]["input"]) == 1024


def test_b1_generation_limit_still_applies():
    p = OpenAIProvider(api_key="k", default_generation_input_max_characters=100)
    assert len(p.construct_prompt("z" * 500, "user")["content"]) == 100


def test_b1_cohere_generation_prompt_not_truncated():
    p = make_cohere(default_input_max_characters=1024)
    p.set_generation_model("m")
    p.generate_text(prompt=LONG_CONTEXT)
    assert "WHAT_IS_A_POINTER" in p.client.calls[0][1]["message"]


def test_b1_factory_wires_separate_limits():
    cfg = nlp_settings(GENERATION_DAFAULT_MAX_INPUT_CHARACTERS=12345, COHERE_API_KEY="k")
    sys.modules.setdefault("cohere", SimpleNamespace(Client=lambda api_key: FakeCohereClient()))
    f = LLMProviderFactory(cfg)
    for name in ("OPENAI", "COHERE", "HUGGINGFACE"):
        prov = f.create(name)
        assert prov.default_input_max_characters == 1024
        assert prov.default_generation_input_max_characters == 12345
    assert f.create("UNKNOWN") is None


# --------------------------------------------------------------------------- B3
def test_b3_cohere_query_vs_document_input_type():
    p = make_cohere()
    p.set_embedding_model("e", 2)
    p.embed_text("q", document_type=DocumentTypeEnum.QUERY.value)
    p.embed_text("d", document_type=DocumentTypeEnum.DOCUMENT.value)
    assert p.client.calls[0][1]["input_type"] == "search_query"
    assert p.client.calls[1][1]["input_type"] == "search_document"


def test_cohere_package_not_required_for_other_backends():
    import stores.llm.providers.CoHereProvider as mod
    assert "cohere" not in vars(mod), "cohere must be imported lazily (only when the COHERE backend is used)"


# --------------------------------------------------------------------------- B9
def test_b9_no_shared_mutable_chat_history():
    p = OpenAIProvider(api_key="k")
    p.client = FakeOpenAIClient()
    p.set_generation_model("m")
    p.generate_text(prompt="one")
    p.generate_text(prompt="two")
    assert len(p.client.calls[0][1]["messages"]) == 1
    assert len(p.client.calls[1][1]["messages"]) == 1


# --------------------------------------------------------------------------- error handling
def test_openai_api_error_returns_none_instead_of_raising():
    p = OpenAIProvider(api_key="k")
    p.client = FakeOpenAIClient(raise_exc=RuntimeError("402 Insufficient credits"))
    p.set_generation_model("m")
    assert p.generate_text("hi") is None


def test_openai_empty_completion_returns_none():
    p = OpenAIProvider(api_key="k")
    p.client = FakeOpenAIClient(content=None)
    p.set_generation_model("m")
    assert p.generate_text("hi") is None


def test_generation_model_not_set_returns_none():
    p = OpenAIProvider(api_key="k")
    assert p.generate_text("hi") is None


# --------------------------------------------------------------------------- OpenRouter wiring (HTTP level)
def test_openrouter_base_url_model_and_params_are_actually_sent():
    """The OpenAI SDK must call the configured OpenRouter URL with the configured model, not api.openai.com."""
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": seen["body"]["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hello from mock"}}],
        })

    from openai import OpenAI
    cfg = nlp_settings(OPENAI_API_KEY="sk-or-test", OPENAI_API_URL="https://openrouter.ai/api/v1",
                       GENERATION_DAFAULT_MAX_TOKENS=777, GENERATION_DAFAULT_TEMPERATURE=0.3)
    p = LLMProviderFactory(cfg).create("OPENAI")
    p.set_generation_model("google/gemma-4-31b-it:free")
    # same configuration as the provider, only the transport is mocked
    p.client = OpenAI(api_key=p.api_key, base_url=p.base_url,
                      http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    assert p.generate_text("Say hi") == "hello from mock"
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-or-test"
    assert seen["body"]["model"] == "google/gemma-4-31b-it:free"
    assert seen["body"]["max_tokens"] == 777
    assert seen["body"]["temperature"] == 0.3


def test_openai_provider_initialization_uses_config():
    cfg = nlp_settings(OPENAI_API_URL="https://openrouter.ai/api/v1", OPENAI_API_TIMEOUT=42)
    p = LLMProviderFactory(cfg).create("OPENAI")
    assert str(p.client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert p.client.timeout == 42


# --------------------------------------------------------------------------- Hugging Face provider
def test_b10_huggingface_is_a_real_provider():
    cfg = nlp_settings(GENERATION_DAFAULT_MAX_TOKENS=50)
    hf = LLMProviderFactory(cfg).create("HUGGINGFACE")
    assert isinstance(hf, HuggingFaceProvider)
    assert hf.enums.SYSTEM.value == "system"


def test_hf_generation_with_fake_pipeline():
    hf = HuggingFaceProvider()
    hf.generation_model = lambda msgs, **kw: [{"generated_text": "answer"}]
    hist = [hf.construct_prompt("sys", hf.enums.SYSTEM.value)]
    assert hf.generate_text("q", chat_history=hist) == "answer"
    assert hist[-1] == {"role": "user", "content": "q"}

    hf.generation_model = lambda msgs, **kw: [{"generated_text": msgs + [{"role": "assistant", "content": "a2"}]}]
    assert hf.generate_text("q") == "a2"


def test_hf_embedding_dimension_and_query_prompt(monkeypatch):
    fake = FakeSentenceTransformer(prompts={"query": "Instruct: ...\nQuery: "}, dim=8)
    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *a, **kw: fake)

    hf = HuggingFaceProvider(default_input_max_characters=10)
    hf.set_embedding_model("some/model", embedding_size=None)
    assert hf.embedding_size == 8  # read from the model when not configured

    vec = hf.embed_text("a long document text", document_type=DocumentTypeEnum.DOCUMENT.value)
    assert len(vec) == 8 and abs(np.linalg.norm(vec) - 1) < 1e-5
    assert "prompt_name" not in fake.calls[-1][1]
    assert fake.calls[-1][0] == "a long doc"  # embedding input limit applied

    hf.embed_text("query", document_type=DocumentTypeEnum.QUERY.value)
    assert fake.calls[-1][1]["prompt_name"] == "query"


def test_hf_embedding_without_prompts_embeds_queries_plainly(monkeypatch):
    fake = FakeSentenceTransformer(prompts={}, dim=4)
    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *a, **kw: fake)
    hf = HuggingFaceProvider()
    hf.set_embedding_model("BAAI/bge-m3", embedding_size=4)
    hf.embed_text("q", document_type=DocumentTypeEnum.QUERY.value)
    assert "prompt_name" not in fake.calls[-1][1]


def test_hf_embedding_errors_return_none():
    hf = HuggingFaceProvider()
    assert hf.embed_text("x") is None  # model not set

    class Broken(FakeSentenceTransformer):
        def encode(self, *a, **kw):
            raise RuntimeError("boom")
    hf.embedding_model = Broken()
    assert hf.embed_text("x") is None

"""
LIVE tests - call real services / load real models. Skipped unless RUN_LIVE_TESTS=1.

    RUN_LIVE_TESTS=1 python -m pytest tests/live -m live -s

Credentials are read from src/.env (never printed). Optional overrides:
    LIVE_GENERATION_MODEL      OpenRouter model id   (default: GENERATION_MODEL_ID from src/.env)
    LIVE_EMBEDDING_MODEL       HF embedding model    (default: EMBEDDING_MODEL_ID from src/.env if HUGGINGFACE, else BAAI/bge-m3)
    LIVE_EMBEDDING_SIZE        expected dimension    (default: EMBEDDING_MODEL_SIZE or 1024)
    LIVE_HF_GENERATION_MODEL   small local HF chat model to verify the optional HF generation backend
"""
import os
import time

import numpy as np
import pytest

from conftest import SRC_ENV_FILE, nlp_settings

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to run live tests"),
]


def env():
    from dotenv import dotenv_values
    if not os.path.exists(SRC_ENV_FILE):
        pytest.skip("src/.env not found")
    return dotenv_values(SRC_ENV_FILE)


def test_live_openrouter_generation_smoke():
    from stores.llm.LLMProviderFactory import LLMProviderFactory
    e = env()
    if not e.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set in src/.env")
    model = os.environ.get("LIVE_GENERATION_MODEL") or e.get("GENERATION_MODEL_ID")
    cfg = nlp_settings(OPENAI_API_KEY=e["OPENAI_API_KEY"], OPENAI_API_URL=e.get("OPENAI_API_URL"),
                       GENERATION_DAFAULT_MAX_TOKENS=1500, OPENAI_API_TIMEOUT=120)
    p = LLMProviderFactory(cfg).create("OPENAI")
    p.set_generation_model(model)
    assert "openrouter.ai" in str(p.client.base_url), "generation must go through OpenRouter"

    t0 = time.perf_counter()
    out = p.generate_text("Reply with exactly one C statement that declares an int named x equal to 5, "
                          "inside a ```c code block. No explanation.")
    print(f"\n[live] OpenRouter model={model} latency={time.perf_counter() - t0:.1f}s -> {out!r}")
    assert out, "no completion (see log: credits / rate limit / model availability)"
    assert "int x" in out.replace("  ", " ")


def _embedding_model():
    e = env()
    if e.get("EMBEDDING_BACKEND") == "HUGGINGFACE" and e.get("EMBEDDING_MODEL_ID"):
        default, size = e["EMBEDDING_MODEL_ID"], int(e.get("EMBEDDING_MODEL_SIZE") or 1024)
    else:
        default, size = "BAAI/bge-m3", 1024
    return os.environ.get("LIVE_EMBEDDING_MODEL", default), int(os.environ.get("LIVE_EMBEDDING_SIZE", size))


def test_live_huggingface_embedding_smoke():
    from stores.llm.providers import HuggingFaceProvider
    from stores.llm.LLMEnums import DocumentTypeEnum
    model_id, size = _embedding_model()
    p = HuggingFaceProvider(default_input_max_characters=2000)
    t0 = time.perf_counter()
    p.set_embedding_model(model_id, size)
    load = time.perf_counter() - t0
    assert p.embedding_model is not None

    doc = p.embed_text("The do-while loop executes its body at least once.", DocumentTypeEnum.DOCUMENT.value)
    other = p.embed_text("A pointer stores the memory address of a variable.", DocumentTypeEnum.DOCUMENT.value)
    q_en = p.embed_text("Which loop always runs at least one time?", DocumentTypeEnum.QUERY.value)
    q_ar = p.embed_text("ما هي الحلقة التي تنفذ مرة واحدة على الأقل؟", DocumentTypeEnum.QUERY.value)

    print(f"\n[live] HF embedding {model_id}: load {load:.1f}s, dim {len(doc)}")
    assert len(doc) == size == p.embedding_size
    for q in (q_en, q_ar):  # cross-lingual: both queries closer to the do-while chunk
        assert np.dot(q, doc) > np.dot(q, other)


@pytest.mark.skipif(not os.environ.get("LIVE_HF_GENERATION_MODEL"),
                    reason="set LIVE_HF_GENERATION_MODEL (e.g. a small instruct model) to test HF generation")
def test_live_huggingface_generation_smoke():
    from stores.llm.providers import HuggingFaceProvider
    model_id = os.environ["LIVE_HF_GENERATION_MODEL"]
    p = HuggingFaceProvider(default_generation_max_output_tokens=40)
    p.set_generation_model(model_id)
    hist = [p.construct_prompt("You are a helpful assistant.", p.enums.SYSTEM.value)]
    t0 = time.perf_counter()
    out = p.generate_text("What keyword declares an integer variable in C? Answer in one word.", chat_history=hist)
    print(f"\n[live] HF generation {model_id}: {time.perf_counter() - t0:.1f}s -> {out!r}")
    assert out and isinstance(out, str)

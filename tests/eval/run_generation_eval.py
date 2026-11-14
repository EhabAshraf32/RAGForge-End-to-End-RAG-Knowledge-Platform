"""
Generation benchmark through the REAL project pipeline:
  HuggingFaceProvider (embeddings) -> temporary Qdrant -> NLPController.answer_rag_question
  -> OpenAIProvider pointed at OpenRouter (OPENAI_API_KEY / OPENAI_API_URL read from src/.env).

Usage (from repo root):
    python tests/eval/run_generation_eval.py --models google/gemma-4-31b-it:free other/model \
        [--embedding BAAI/bge-m3] [--out results_dir]

Makes 12 real API calls per model (paced for free-tier rate limits). The API key is never printed.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "src")
sys.path.insert(0, SRC)

NOT_AVAILABLE = ["not available in the provided course material", "not available in the provided",
                 "غير موجودة في مادة المقرر", "غير موجود في مادة المقرر", "غير متوفرة في مادة المقرر"]


def arabic_ratio(text):
    text = re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)
    ar = len(re.findall(r"[؀-ۿ]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return ar / max(1, ar + lat)


def text_blocks(answer):
    return re.findall(r"```(?:text|txt|output|plaintext)?\s*\n(.*?)```", answer or "", flags=re.DOTALL)


def check(q, answer, raw, sources):
    a = (answer or "").lower()
    res = {}
    res["has_answer"] = bool(answer)
    res["marker_emitted"] = bool(re.search(r"\[\[\s*USED_DOCUMENTS", raw or "", re.I))
    if q.get("need_c_block"):
        res["c_code_block"] = "```c" in (answer or "")
    if q.get("need_text_block") or q.get("expected_output"):
        res["text_output_block"] = "```text" in (answer or "")
    if q.get("expected_output"):
        exp = " ".join(q["expected_output"].split())
        res["correct_output"] = any(" ".join(b.split()) == exp or exp in " ".join(b.split())
                                    for b in text_blocks(answer)) if text_blocks(answer) else exp in a
    if q.get("expect_any"):
        res["expected_content_any"] = any(s.lower() in a for s in q["expect_any"])
    if q.get("expect_all"):
        res["expected_content_all"] = all(s.lower() in a for s in q["expect_all"])
    if q.get("expect_arabic"):
        res["answered_in_arabic"] = arabic_ratio(answer) > 0.3
    if q.get("expect_not_available"):
        res["says_not_available"] = any(p.lower() in a for p in NOT_AVAILABLE)
        res["no_cited_sources"] = sources is not None and len(sources["cited"]) == 0
    elif q.get("grounded"):
        res["cites_sources"] = sources is not None and len(sources["cited"]) > 0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--embedding", default="BAAI/bge-m3")
    ap.add_argument("--embedding-size", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between calls (free tier: ~20 req/min)")
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "mini_rag_gen_eval"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from dotenv import dotenv_values
    env = dotenv_values(os.path.join(SRC, ".env"))
    os.environ.setdefault("APP_NAME", "eval")  # BaseController needs Settings; avoid reading secrets twice
    for k in ["APP_VERSION", "FILE_MAX_SIZE", "FILE_DEFAULT_CHUNK_SIZE", "MONGODB_URL", "MONGODB_DATABASE",
              "GENERATION_BACKEND", "EMBEDDING_BACKEND", "VECTOR_DB_BACKEND", "VECTOR_DB_PATH"]:
        os.environ.setdefault(k, "1" if "SIZE" in k else "eval")
    os.environ.setdefault("FILE_ALLOWED_TYPES", '["application/pdf"]')
    os.chdir(tempfile.gettempdir())  # no .env in cwd

    from bson import ObjectId
    from stores.llm.LLMProviderFactory import LLMProviderFactory
    from stores.vectordb.providers import QdrantDBProvider
    from stores.llm.templatess.template_parser import TemplateParser
    from controllers import NLPController
    from models.db_schemes import DataChunk, Project

    cfg = SimpleNamespace(OPENAI_API_KEY=env.get("OPENAI_API_KEY"), OPENAI_API_URL=env.get("OPENAI_API_URL"),
                          COHERE_API_KEY=None, INPUT_DAFAULT_MAX_CHARACTERS=2000,
                          GENERATION_DAFAULT_MAX_INPUT_CHARACTERS=20000,
                          GENERATION_DAFAULT_MAX_TOKENS=args.max_tokens, GENERATION_DAFAULT_TEMPERATURE=0.1,
                          OPENAI_API_TIMEOUT=180)
    print("generation base_url:", cfg.OPENAI_API_URL)
    factory = LLMProviderFactory(cfg)
    embedding = factory.create("HUGGINGFACE")
    t0 = time.perf_counter()
    embedding.set_embedding_model(args.embedding, args.embedding_size)
    print(f"embedding model {args.embedding} loaded in {time.perf_counter() - t0:.1f}s, dim={embedding.embedding_size}")

    qdir = tempfile.mkdtemp(prefix="gen_eval_qdrant_")
    qdrant = QdrantDBProvider(db_path=qdir, distance_method="cosine")
    qdrant.connect()

    data = json.load(open(os.path.join(HERE, "retrieval_dataset.json"), encoding="utf-8"))
    project = Project(_id=ObjectId(), project_id="geneval")
    chunks = [DataChunk(_id=ObjectId(), chunk_text=c["text"], chunk_order=i + 1, chunk_project_id=project.id,
                        chunk_asset_id=ObjectId(),
                        chunk_metadata={"document_name": f"{c['id'].split('_')[0].upper()}.pdf",
                                        "doc_type": c["doc_type"], "page_number": i + 1, "course": "CS101"})
              for i, c in enumerate(data["corpus"])]

    questions = json.load(open(os.path.join(HERE, "generation_questions.json"), encoding="utf-8"))["questions"]
    summary = {}
    for model_id in args.models:
        generation = factory.create("OPENAI")
        generation.set_generation_model(model_id)
        ctrl = NLPController(vectordb_client=qdrant, generation_client=generation, embedding_client=embedding,
                             template_parser=TemplateParser(language="en", default_language="en"))
        if not qdrant.is_collection_existed("collection_geneval"):
            assert ctrl.index_into_vector_db(project, chunks, list(range(len(chunks))), do_reset=True)

        raw_holder = {}
        original = generation.generate_text

        def capture(*a, **kw):
            raw_holder["raw"] = original(*a, **kw)
            return raw_holder["raw"]
        generation.generate_text = capture

        results, failures = [], 0
        for q in questions:
            raw_holder.clear()
            t0 = time.perf_counter()
            answer, _, _, sources = ctrl.answer_rag_question(project, q["q"], limit=4)
            latency = time.perf_counter() - t0
            checks = check(q, answer, raw_holder.get("raw"), sources)
            passed = sum(checks.values())
            results.append({"id": q["id"], "category": q["category"], "latency_s": round(latency, 1),
                            "checks": checks, "score": f"{passed}/{len(checks)}",
                            "cited": sources["cited"] if sources else None, "answer": answer})
            print(f"[{model_id}] {q['id']:18s} {passed}/{len(checks)} {latency:5.1f}s "
                  f"{'' if answer else 'NO ANSWER'}")
            failures = failures + 1 if not answer else 0
            if failures >= 3:
                print(f"[{model_id}] 3 consecutive failures -> stopping this model (see server log above)")
                break
            time.sleep(args.pause)

        total = sum(sum(r["checks"].values()) for r in results)
        possible = sum(len(r["checks"]) for r in results)
        answered = [r for r in results if r["answer"]]
        summary[model_id] = {
            "questions_run": len(results), "answered": len(answered),
            "checks_passed": f"{total}/{possible}",
            "median_latency_s": sorted(r["latency_s"] for r in answered)[len(answered) // 2] if answered else None,
        }
        safe = re.sub(r"[^\w.-]", "_", model_id)
        json.dump(results, open(os.path.join(args.out, f"gen_{safe}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        with open(os.path.join(args.out, f"gen_{safe}.md"), "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"\n\n---\n## {r['id']} ({r['category']}) score {r['score']} - {r['latency_s']}s\n"
                        f"checks: {r['checks']}\ncited: {r['cited']}\n\n{r['answer']}\n")

    print(json.dumps(summary, indent=1))
    json.dump(summary, open(os.path.join(args.out, "gen_summary.json"), "w"), indent=1)
    qdrant.client.close()


if __name__ == "__main__":
    main()

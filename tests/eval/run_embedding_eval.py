"""
Embedding model retrieval benchmark (local CPU, sentence-transformers).

Usage (from repo root):
    python tests/eval/run_embedding_eval.py BAAI/bge-m3 [--out results.json]

Measures hit@1 / hit@3 / MRR per language group (en / ar / mixed), model load time,
per-chunk and per-query latency (one text at a time, the way NLPController embeds),
process RAM, and embedding dimension.
Queries use the model's "query" prompt when the model defines one (e.g. Qwen3-Embedding).
"""
import argparse
import json
import os
import time

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))


def rss_gb():
    return psutil.Process().memory_info().rss / 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.load(open(os.path.join(HERE, "retrieval_dataset.json"), encoding="utf-8"))
    corpus, queries = data["corpus"], data["queries"]

    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(os.cpu_count())

    rss0 = rss_gb()
    t0 = time.perf_counter()
    model = SentenceTransformer(args.model_id, device="cpu")
    load_s = time.perf_counter() - t0
    rss_loaded = rss_gb()

    prompts = getattr(model, "prompts", {}) or {}
    query_kwargs = {"prompt_name": "query"} if "query" in prompts else {}

    # warm-up
    model.encode("warm up", normalize_embeddings=True)

    t0 = time.perf_counter()
    doc_vecs = np.array([model.encode(c["text"], normalize_embeddings=True) for c in corpus])
    doc_s = (time.perf_counter() - t0) / len(corpus)

    t0 = time.perf_counter()
    q_vecs = np.array([model.encode(q["q"], normalize_embeddings=True, **query_kwargs) for q in queries])
    q_s = (time.perf_counter() - t0) / len(queries)
    rss_peak = rss_gb()

    ids = [c["id"] for c in corpus]
    sims = q_vecs @ doc_vecs.T

    per_lang = {}
    details = []
    for qi, q in enumerate(queries):
        order = np.argsort(-sims[qi])
        ranked = [ids[i] for i in order]
        rank = ranked.index(q["gold"]) + 1
        g = per_lang.setdefault(q["lang"], {"n": 0, "hit1": 0, "hit3": 0, "mrr": 0.0})
        g["n"] += 1
        g["hit1"] += rank == 1
        g["hit3"] += rank <= 3
        g["mrr"] += 1.0 / rank
        details.append({"q": q["q"], "lang": q["lang"], "gold": q["gold"], "rank": rank, "top3": ranked[:3]})

    total = {"n": 0, "hit1": 0, "hit3": 0, "mrr": 0.0}
    for g in per_lang.values():
        for k in total:
            total[k] += g[k]
    summary = {}
    for name, g in list(per_lang.items()) + [("all", total)]:
        summary[name] = {"n": g["n"], "hit@1": round(g["hit1"] / g["n"], 3),
                         "hit@3": round(g["hit3"] / g["n"], 3), "mrr": round(g["mrr"] / g["n"], 3)}

    result = {
        "model": args.model_id,
        "dimension": int(doc_vecs.shape[1]),
        "query_prompt_used": bool(query_kwargs),
        "load_seconds": round(load_s, 1),
        "ms_per_chunk": round(doc_s * 1000, 1),
        "ms_per_query": round(q_s * 1000, 1),
        "ram_gb_model": round(rss_loaded - rss0, 2),
        "ram_gb_process_peak": round(rss_peak, 2),
        "summary": summary,
        "misses": [d for d in details if d["rank"] > 1],
    }
    if args.out:
        json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # ensure_ascii=True: safe on Windows consoles that cannot print Arabic
    print(json.dumps(result, ensure_ascii=True, indent=1))


if __name__ == "__main__":
    main()

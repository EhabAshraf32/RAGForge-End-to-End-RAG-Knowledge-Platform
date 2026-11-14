"""
End-to-end run of the HOSTED configuration: both Streamlit apps with API_BASE_URL="embedded" (the FastAPI
backend runs inside the Streamlit process), using the real external services from the deployment:
Cloudflare Workers AI bge-m3, Qdrant Cloud, OpenRouter, and MongoDB (--mongo-url, default: local).

Secrets are read from a dotenv file (default E:/From-Transformers-To-Proffisional/deploy-secrets.env) and
never printed. Uses a NEW project id; removes only its own Qdrant collection and MongoDB database.

    python tests/e2e/run_embedded_e2e.py [--mongo-url mongodb+srv://...]
"""
import argparse
import json
import os
import secrets
import sys
import time
import uuid

import pymongo
import pymupdf
import requests
from dotenv import dotenv_values

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FRONTEND = os.path.join(REPO, "streamlit_app")
SRC = os.path.join(REPO, "src")
RESULTS = []


def step(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def pdf_bytes(pages):
    doc = pymupdf.open()
    for t in pages:
        doc.new_page().insert_text((50, 72), t, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def texts(at):
    return "\n".join(str(e.value) for k in ("markdown", "success", "error", "warning", "info", "caption")
                     for e in getattr(at, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secrets", default="E:/From-Transformers-To-Proffisional/deploy-secrets.env")
    ap.add_argument("--mongo-url", default=None)
    ap.add_argument("--model", default="nvidia/nemotron-3.5-super-vl:free")
    args = ap.parse_args()
    sec = dotenv_values(args.secrets)
    mongo_url = args.mongo_url or sec.get("MONGODB_URL") or dotenv_values(os.path.join(SRC, ".env"))["MONGODB_URL"]

    sys.path.insert(0, SRC)
    from helpers.security import hash_password

    run = uuid.uuid4().hex[:8]
    project, db_name = f"embed{run}", f"mini_rag_embed_{run}"
    admin_pw, code = secrets.token_urlsafe(12), secrets.token_urlsafe(8)
    os.chdir(REPO)  # no .env here: configuration comes only from the variables below
    os.environ.update({
        "API_BASE_URL": "embedded", "ADMIN_DEFAULT_PROJECT_ID": project,
        "APP_NAME": "mini-RAG", "APP_VERSION": "e2e", "FILE_ALLOWED_TYPES": '["application/pdf"]',
        "FILE_MAX_SIZE": "10", "FILE_DEFAULT_CHUNK_SIZE": "512000", "STORE_FILES_IN_DB": "true",
        "MONGODB_URL": mongo_url, "MONGODB_DATABASE": db_name,
        "GENERATION_BACKEND": "OPENAI", "OPENAI_API_URL": "https://openrouter.ai/api/v1",
        "OPENAI_API_KEY": dotenv_values(os.path.join(SRC, ".env"))["OPENAI_API_KEY"],
        "GENERATION_MODEL_ID": args.model, "GENERATION_DAFAULT_MAX_TOKENS": "3000",
        "GENERATION_DAFAULT_TEMPERATURE": "0.1",
        "EMBEDDING_BACKEND": "CLOUDFLARE", "EMBEDDING_MODEL_ID": "@cf/baai/bge-m3", "EMBEDDING_MODEL_SIZE": "1024",
        "INPUT_DAFAULT_MAX_CHARACTERS": "1024",
        "CLOUDFLARE_ACCOUNT_ID": sec["CLOUDFLARE_ACCOUNT_ID"], "CLOUDFLARE_API_TOKEN": sec["CLOUDFLARE_API_TOKEN"],
        "VECTOR_DB_BACKEND": "QDRANT", "VECTOR_DB_PATH": "qdrant_unused", "VECTOR_DB_DISTANCE_METHOD": "cosine",
        "QDRANT_URL": sec["QDRANT_URL"], "QDRANT_API_KEY": sec["QDRANT_API_KEY"],
        "JWT_SECRET_KEY": secrets.token_urlsafe(48), "ADMIN_USERNAME": "ta_e2e",
        "ADMIN_PASSWORD_HASH": hash_password(admin_pw), "STUDENT_ACCESS_CODE_HASH": hash_password(code),
        "STUDENT_PROJECT_ID": project, "TRUSTED_FRONTEND_SECRET": secrets.token_urlsafe(24),
    })
    os.environ["FRONTEND_SHARED_SECRET"] = os.environ["TRUSTED_FRONTEND_SECRET"]
    sys.path.insert(0, FRONTEND)
    from streamlit.testing.v1 import AppTest
    print(f"run={run} project={project} mongo_db={db_name} (Qdrant Cloud, Cloudflare bge-m3, {args.model})")

    try:
        admin = AppTest.from_file(os.path.join(FRONTEND, "admin_app.py"), default_timeout=900).run()
        admin.text_input[0].input("ta_e2e")
        admin.text_input[1].input(admin_pw)
        admin.button[0].click().run()
        step("admin login (embedded backend started)", bool(admin.metric) and not admin.exception, str(admin.exception or ""))

        admin.file_uploader[0].set_value([("lecture7.pdf", pdf_bytes([
            "Lecture 7: Arrays in C\nAn array stores elements of the same type. Indexes start at 0.\n"
            "int marks[5] = {90, 80, 70, 60, 50};  marks[0] is 90 and marks[4] is 50.",
            "Accessing marks[5] is out of bounds: C does not check it and the behavior is undefined."]),
            "application/pdf")]).run()
        next(t for t in admin.text_input if t.label.startswith("Name for file")).input("Lecture 7 - Arrays.pdf")
        admin.radio[0].set_value("Lecture")
        next(t for t in admin.text_input if t.label == "Course").input("CS101")
        next(t for t in admin.text_input if t.label == "Topic").input("arrays")
        next(b for b in admin.button if b.label == "Upload").click().run()
        step("upload PDF with metadata (stored in GridFS)", "Uploaded: Lecture 7 - Arrays.pdf" in texts(admin))
        next(b for b in admin.button if b.label == "Process documents").click().run()
        step("process", "Lecture 7 - Arrays.pdf: 2 chunks" in texts(admin))
        t0 = time.perf_counter()
        next(b for b in admin.button if b.label == "Build search index").click().run()
        step("index with Cloudflare bge-m3 into Qdrant Cloud", "2 chunks indexed" in texts(admin),
             f"{time.perf_counter() - t0:.1f}s")
        row = admin.dataframe[0].value.iloc[0]
        step("status table", "indexed" in row["Indexing"] and row["Type"] == "Lecture", f"{row['Processing']} / {row['Indexing']}")

        student = AppTest.from_file(os.path.join(FRONTEND, "student_app.py"), default_timeout=300).run()
        student.text_input[0].input(code)
        student.button[0].click().run()
        step("student login", bool(student.chat_input))
        student.chat_input[0].set_value("What happens if I access marks[5] when the array has 5 elements?").run()
        msgs = student.session_state["messages"]
        cited = (msgs[-1].get("sources") or {}).get("cited", []) if msgs[-1]["role"] == "assistant" else []
        step("English question answered", msgs[-1]["role"] == "assistant", f"{len(msgs[-1]['content'])} chars")
        step("cites the uploaded lecture", any(s["document_name"] == "Lecture 7 - Arrays.pdf" and s["page_number"] == 2
                                               for s in cited), json.dumps(cited, ensure_ascii=False))
        time.sleep(4)
        student.chat_input[0].set_value("طيب ايه index أول عنصر في الـ array؟").run()
        msgs = student.session_state["messages"]
        step("Arabic follow-up answered", len(msgs) == 4 and msgs[-1]["role"] == "assistant",
             f"arabic={any('\u0600' <= c <= '\u06FF' for c in msgs[-1]['content'])}")

        from identity import api_base_url
        base = api_base_url()
        h = {"Authorization": f"Bearer {student.session_state['student_token']}"}
        codes = [requests.get(f"{base}/data/assets/{project}", headers=h).status_code,
                 requests.post(f"{base}/nlp/index/push/{project}", headers=h, json={}).status_code]
        step("student token rejected by admin endpoints", codes == [403, 403], str(codes))
        step("embedded backend listens on localhost only", base.startswith("http://127.0.0.1:"), base)
    except Exception as e:
        step("unexpected error", False, repr(e))
    finally:
        from qdrant_client import QdrantClient
        q = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
        if q.collection_exists(f"collection_{project}"):
            q.delete_collection(f"collection_{project}")
        pymongo.MongoClient(mongo_url).drop_database(db_name)
        print(f"cleanup: removed Qdrant collection_{project} and MongoDB database {db_name}", flush=True)
    print(f"\nEMBEDDED E2E: {sum(RESULTS)}/{len(RESULTS)} passed")
    os._exit(0 if RESULTS and all(RESULTS) else 1)


if __name__ == "__main__":
    main()

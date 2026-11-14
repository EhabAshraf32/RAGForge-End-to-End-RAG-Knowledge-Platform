"""
Full isolated end-to-end run of the REAL platform:
  FastAPI backend (uvicorn subprocess) + real MongoDB (NEW temp database) + Qdrant (temp dir)
  + real Hugging Face embeddings + real OpenRouter generation
  + both Streamlit apps (launched as servers, and driven with Streamlit AppTest).

Isolation / safety:
  * credentials are generated in memory for this run only and passed to the backend through
    environment variables (which take priority over src/.env - the file is not modified or printed);
  * MONGODB_DATABASE / VECTOR_DB_PATH / STUDENT_PROJECT_ID point to brand-new temporary values;
  * only what this run created is removed at the end (temp DB, temp Qdrant dir, the e2e upload folder).

Usage (repo root, project environment):
    python tests/e2e/run_e2e.py --model nvidia/nemotron-3.5-super-vl:free
"""
import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import pymongo
import pymupdf
import requests
from dotenv import dotenv_values

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "src")
FRONTEND = os.path.join(REPO, "streamlit_app")
sys.path.insert(0, SRC)
sys.path.insert(0, FRONTEND)

LECTURE_PAGES = [
    "Lecture 5: Loops in C\nThe for loop has three parts: initialization; condition; update.\n"
    "for (int i = 1; i <= 10; i++) {\n    printf(\"%d \", i);\n}\nThis loop prints the numbers from 1 to 10.",
    "The while loop checks its condition before each iteration, so its body may never execute.\n"
    "The do-while loop checks the condition after the body, so the body executes at least once:\n"
    "do {\n    scanf(\"%d\", &num);\n} while (num <= 0);",
]

RESULTS = []


def step(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    return ok


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url, proc, seconds=300):
    for _ in range(seconds):
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            return False
        time.sleep(1)
    return False


def pdf_bytes(pages):
    doc = pymupdf.open()
    for t in pages:
        doc.new_page().insert_text((50, 72), t, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def app_texts(at):
    out = []
    for kind in ("markdown", "success", "error", "warning", "info", "caption"):
        out += [e.value for e in getattr(at, kind)]
    return "\n".join(str(t) for t in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--embedding", default="BAAI/bge-m3")
    ap.add_argument("--embedding-size", type=int, default=1024)
    args = ap.parse_args()

    from helpers.security import hash_password

    run_id = uuid.uuid4().hex[:8]
    db_name, project_id = f"mini_rag_e2e_{run_id}", f"e2e{run_id}"
    work = tempfile.mkdtemp(prefix="mini_rag_e2e_")
    qdrant_dir = os.path.join(work, "qdrant")
    upload_dir = os.path.join(SRC, "assets", "files", project_id)
    admin_user, admin_pass, student_code = "ta_e2e", secrets.token_urlsafe(16), secrets.token_urlsafe(12)

    mongo_url = dotenv_values(os.path.join(SRC, ".env"))["MONGODB_URL"]
    mongo = pymongo.MongoClient(mongo_url, serverSelectionTimeoutMS=3000)
    assert db_name not in mongo.list_database_names()

    api_port, admin_port, student_port = free_port(), free_port(), free_port()
    api_base = f"http://127.0.0.1:{api_port}/api/v1"
    backend_env = dict(os.environ)
    backend_env.update({
        "MONGODB_DATABASE": db_name, "VECTOR_DB_PATH": qdrant_dir,
        "EMBEDDING_BACKEND": "HUGGINGFACE", "EMBEDDING_MODEL_ID": args.embedding,
        "EMBEDDING_MODEL_SIZE": str(args.embedding_size),
        "GENERATION_BACKEND": "OPENAI", "GENERATION_MODEL_ID": args.model, "GENERATION_DAFAULT_MAX_TOKENS": "3000",
        "JWT_SECRET_KEY": secrets.token_urlsafe(48), "ADMIN_USERNAME": admin_user,
        "ADMIN_PASSWORD_HASH": hash_password(admin_pass), "STUDENT_ACCESS_CODE_HASH": hash_password(student_code),
        "STUDENT_PROJECT_ID": project_id, "PYTHONDONTWRITEBYTECODE": "1",
    })
    frontend_env = dict(os.environ, API_BASE_URL=api_base, ADMIN_DEFAULT_PROJECT_ID=project_id)
    print(f"run={run_id} db={db_name} project={project_id} model={args.model} embedding={args.embedding}")

    logs, procs = {}, {}
    try:
        def start(name, cmd, cwd, env):
            logs[name] = open(os.path.join(work, f"{name}.log"), "w", encoding="utf-8")
            procs[name] = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=logs[name], stderr=subprocess.STDOUT)

        t0 = time.perf_counter()
        start("backend", [sys.executable, "-m", "uvicorn", "main:app", "--port", str(api_port), "--workers", "1"],
              SRC, backend_env)
        step("backend started (embedding model loaded)", wait_http(f"{api_base}/", procs["backend"]),
             f"{time.perf_counter() - t0:.1f}s")
        for name, port, script in [("admin_ui", admin_port, "admin_app.py"), ("student_ui", student_port, "student_app.py")]:
            start(name, [sys.executable, "-m", "streamlit", "run", script, "--server.port", str(port),
                         "--server.headless", "true", "--browser.gatherUsageStats", "false"], FRONTEND, frontend_env)
        for name, port in [("admin_ui", admin_port), ("student_ui", student_port)]:
            ok = wait_http(f"http://127.0.0.1:{port}/_stcore/health", procs[name], 120)
            page = requests.get(f"http://127.0.0.1:{port}/", timeout=10).status_code if ok else None
            step(f"{name} Streamlit server serves", ok and page == 200, f"port {port}, page HTTP {page}")

        # the AppTest-driven apps use the same backend as the servers above
        os.environ["API_BASE_URL"] = api_base
        os.environ["ADMIN_DEFAULT_PROJECT_ID"] = project_id
        import settings
        settings.API_BASE_URL, settings.ADMIN_DEFAULT_PROJECT_ID = api_base, project_id
        from streamlit.testing.v1 import AppTest

        # 1. admin logs in
        admin = AppTest.from_file(os.path.join(FRONTEND, "admin_app.py"), default_timeout=900).run()
        admin.text_input[0].input(admin_user)
        admin.text_input[1].input(admin_pass)
        admin.button[0].click().run()
        step("1. admin logs in", bool(admin.metric) and not admin.exception)

        # 2-3. upload a lecture PDF with type + metadata
        admin.file_uploader[0].set_value([("lecture5.pdf", pdf_bytes(LECTURE_PAGES), "application/pdf")]).run()
        next(t for t in admin.text_input if t.label.startswith("Name for file")).input("Lecture 5 - Loops.pdf")
        admin.radio[0].set_value("Lecture")
        next(t for t in admin.text_input if t.label == "Course").input("CS101")
        next(t for t in admin.text_input if t.label == "Topic").input("loops")
        next(b for b in admin.button if b.label == "Upload").click().run()
        step("2-3. admin uploads lecture PDF with metadata", "Uploaded: Lecture 5 - Loops.pdf" in app_texts(admin))

        # 4. backend processes it
        next(b for b in admin.button if b.label == "Process documents").click().run()
        step("4. backend processes it", "Lecture 5 - Loops.pdf: 2 chunks" in app_texts(admin), app_texts(admin)[-120:])

        # 5. backend indexes it
        t0 = time.perf_counter()
        next(b for b in admin.button if b.label == "Build search index").click().run()
        step("5. backend indexes it (real bge-m3)", "2 chunks indexed" in app_texts(admin),
             f"{time.perf_counter() - t0:.1f}s")
        row = admin.dataframe[0].value.iloc[0]
        step("   dashboard shows final status", "indexed" in row["Indexing"] and row["Type"] == "Lecture"
             and row["Course"] == "CS101", f"{row['Processing']} / {row['Indexing']}")

        # 6. student accesses the chat
        student = AppTest.from_file(os.path.join(FRONTEND, "student_app.py"), default_timeout=300).run()
        student.text_input[0].input(student_code)
        student.text_input[1].input("Mona")
        student.button[0].click().run()
        step("6. student accesses the chat", bool(student.chat_input) and "admin_token" not in student.session_state)

        # 7-8. question answered from the uploaded material, with the correct source
        t0 = time.perf_counter()
        student.chat_input[0].set_value("What is the difference between while and do-while? Show an example.").run()
        dt = time.perf_counter() - t0
        msgs = student.session_state["messages"]
        answer = msgs[-1]["content"] if msgs and msgs[-1]["role"] == "assistant" else ""
        cited = (msgs[-1].get("sources") or {}).get("cited", []) if answer else []
        step("7. student question answered (real OpenRouter)", bool(answer) and "```c" in answer,
             f"{dt:.1f}s, {len(answer)} chars")
        step("8. answer cites the uploaded lecture", any(s["document_name"] == "Lecture 5 - Loops.pdf"
             and s["doc_type"] == "lecture" and s["page_number"] == 2 for s in cited),
             json.dumps(cited, ensure_ascii=False))
        sources_ui = student.chat_message[1].expander[0].markdown[0].value if student.chat_message[1].expander else ""
        step("   sources rendered in the chat", "Lecture 5 - Loops.pdf" in sources_ui and "page 2" in sources_ui,
             sources_ui)

        # 9. follow-up question (Arabic) using chat_history
        time.sleep(4)
        t0 = time.perf_counter()
        student.chat_input[0].set_value("طيب اكتبلي مثال تاني بالـ for loop وايه الناتج؟").run()
        msgs = student.session_state["messages"]
        follow = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else ""
        step("9. follow-up question answered with history", bool(follow) and len(msgs) == 4
             and "```c" in follow, f"{time.perf_counter() - t0:.1f}s, arabic={any('؀' <= c <= 'ۿ' for c in follow)}")

        # 10. student tries admin endpoints directly
        h = {"Authorization": f"Bearer {student.session_state['student_token']}"}
        codes = [requests.get(f"{api_base}/data/assets/{project_id}", headers=h).status_code,
                 requests.post(f"{api_base}/nlp/index/push/{project_id}", headers=h, json={}).status_code,
                 requests.delete(f"{api_base}/data/asset/{project_id}/x.pdf", headers=h).status_code,
                 requests.post(f"{api_base}/data/upload/{project_id}", headers=h,
                               files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}).status_code]
        step("10. student calling admin endpoints is rejected", codes == [403, 403, 403, 403], str(codes))
        no_token = requests.post(f"{api_base}/nlp/index/answer/{project_id}", json={"text": "x"}).status_code
        step("    anonymous answer request is rejected", no_token == 401, str(no_token))

        out = os.path.join(tempfile.gettempdir(), f"mini_rag_e2e_{run_id}_chat.json")
        json.dump(student.session_state["messages"], open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("chat transcript saved to", out)
    except Exception as e:
        step("unexpected error", False, repr(e))
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
        for f in logs.values():
            f.close()
        tail = open(os.path.join(work, "backend.log"), encoding="utf-8", errors="replace").readlines()[-12:] \
            if os.path.exists(os.path.join(work, "backend.log")) else []
        print("---- backend log (tail) ----\n" + "".join(tail))
        mongo.drop_database(db_name)
        shutil.rmtree(upload_dir, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
        print(f"cleanup: dropped {db_name}; removed {upload_dir} and {work}")

    print(f"\nE2E: {sum(RESULTS)}/{len(RESULTS)} steps passed")
    sys.exit(0 if RESULTS and all(RESULTS) else 1)


if __name__ == "__main__":
    main()

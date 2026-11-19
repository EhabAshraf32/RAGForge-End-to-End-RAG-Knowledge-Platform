"""
Load a folder of course PDFs into the course project through the EXISTING backend API
(upload -> process -> build index), using the deployment configuration (Atlas, Qdrant Cloud, Cloudflare).

The backend runs in-process on 127.0.0.1 (same as the embedded Streamlit mode); an admin token is minted
locally with the deployment JWT secret. Nothing is printed except file names and counts.

    python tools/ingest_course_files.py --folder E:/From-Transformers-To-Proffisional/course_files \
        --secrets E:/From-Transformers-To-Proffisional/streamlit-secrets-admin.toml

Document type is taken from the file name: names containing "lab" / "sheet" / "exercise" -> Lab,
"lec" / "lecture" / "chapter" -> Lecture, otherwise Other. Already uploaded names are skipped.
"""
import argparse
import os
import re
import socket
import sys
import threading
import time  # noqa: F811  (also used for retry pauses)
import tomllib

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")


def guess_doc_type(name: str):
    n = name.lower()
    if re.search(r"lab|sheet|exercise|assignment|tutorial|section", n):
        return "lab"
    if re.search(r"lec|lecture|chapter|ch\d|slides|week", n):
        return "lecture"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--secrets", required=True, help="Streamlit secrets TOML of the deployment")
    ap.add_argument("--course", default="Structured Programming using C")
    args = ap.parse_args()

    conf = tomllib.load(open(args.secrets, "rb"))
    for k, v in conf.items():
        os.environ[k] = str(v)
    project = conf["STUDENT_PROJECT_ID"]
    os.chdir(REPO)
    sys.path.insert(0, SRC)
    import uvicorn
    import main as backend
    from helpers.security import create_access_token

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(backend.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    api = f"http://127.0.0.1:{port}/api/v1"
    token = create_access_token("admin:ingest", "admin", conf["JWT_SECRET_KEY"], 120)
    h = {"Authorization": f"Bearer {token}"}

    existing = {a["document_name"] for a in requests.get(f"{api}/data/assets/{project}", headers=h).json()["assets"]}
    files = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(".pdf"))
    print(f"project {project}: {len(files)} PDF(s) in folder, {len(existing)} already uploaded")
    for name in files:
        if name in existing:
            print(f"  skip (already uploaded): {name}")
            continue
        with open(os.path.join(args.folder, name), "rb") as f:
            r = requests.post(f"{api}/data/upload/{project}", headers=h,
                              files={"file": (name, f, "application/pdf")},
                              data={"doc_type": guess_doc_type(name), "course": args.course})
        print(f"  upload {name} [{guess_doc_type(name)}]: {r.status_code} {r.json().get('signal')}")

    listing = requests.get(f"{api}/data/assets/{project}", headers=h).json()
    processed_any = False
    for a in listing["assets"]:
        if a["processing_status"] in ("pending", "failed"):
            r = requests.post(f"{api}/data/process/{project}", headers=h, json={"file_id": a["file_id"]}, timeout=900)
            print(f"  process {a['document_name']}: {r.status_code} {r.json().get('inserted_chunks', r.json().get('signal'))}")
            processed_any = processed_any or r.status_code == 200

    summary = requests.get(f"{api}/data/assets/{project}", headers=h).json()["summary"]
    if processed_any or summary["index_needs_rebuild"] or summary["awaiting_indexing"]:
        # full rebuild from MongoDB keeps the index exactly in sync with ALL documents (old + new);
        # verified afterwards and retried, because a failed rebuild leaves students without search
        for attempt in range(1, 4):
            r = requests.post(f"{api}/nlp/index/push/{project}", headers=h, json={"do_reset": 1}, timeout=3600)
            print(f"  build index (attempt {attempt}): {r.status_code} "
                  f"{r.json().get('inserted_items_count', r.json().get('signal'))} chunks")
            summary = requests.get(f"{api}/data/assets/{project}", headers=h).json()["summary"]
            if summary["total_chunks"] and summary["total_vectors"] == summary["total_chunks"]:
                break
            time.sleep(10 * attempt)
    else:
        print("  nothing new: the search index is already up to date")
    s = requests.get(f"{api}/data/assets/{project}", headers=h).json()["summary"]
    print(f"done: {s['total_documents']} documents ({s['lectures']} lectures, {s['labs']} labs), "
          f"{s['indexed_documents']} indexed, {s['processing_errors']} errors, {s['total_vectors']} vectors")
    ok = s["total_chunks"] > 0 and s["total_vectors"] == s["total_chunks"]
    if not ok:
        print("\n" + "!" * 70 + f"\n  WARNING: the search index is NOT complete ({s['total_vectors']} of "
              f"{s['total_chunks']} chunks).\n  Students will not get answers. Run this again in a few minutes;"
              "\n  if it keeps failing, check Cloudflare / Qdrant Cloud.\n" + "!" * 70)
    server.should_exit = True
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()

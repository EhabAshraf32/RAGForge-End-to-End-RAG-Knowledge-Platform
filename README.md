# 🧠 RAGForge

A modular **Retrieval-Augmented Generation (RAG)** application — a FastAPI backend that turns your documents into a searchable, question-answerable knowledge base, paired with a polished Streamlit UI on top of it.

```
┌──────────────────────────┐
│      Streamlit UI         │   Upload → Process → Index → Search → Chat
│  Modern RAG Dashboard     │
└────────────┬──────────────┘
             │ HTTP
             ▼
┌──────────────────────────┐
│      FastAPI Backend      │   Chunking, embeddings, retrieval, generation
└────────────┬──────────────┘
             ▼
┌──────────────────────────┐
│  MongoDB · Qdrant · LLM   │   Metadata · Vector index · OpenAI / Cohere
└──────────────────────────┘
```

## ✨ Features

- **Document upload & processing** — upload PDFs/text files per project, chunk them with configurable size/overlap
- **Vector indexing** — embed chunks and push them into a [Qdrant](https://qdrant.tech/) collection
- **Semantic search** — query the knowledge base and get back the most relevant chunks, ranked by similarity
- **RAG question answering** — ask natural-language questions and get answers grounded in your own documents
- **Pluggable LLM & embedding providers** — OpenAI and Cohere supported out of the box via a factory/provider pattern
- **Bilingual prompt templates** — English and Arabic RAG prompts (`src/stores/llm/templatess/locales`)
- **Streamlit dashboard** — a full product-style UI on top of the API: upload, process, index, search, and a ChatGPT-style RAG chat, all in one place

## 🎯 What this project demonstrates

- **Modular RAG architecture** built with the **Factory Design Pattern** — LLM and vector-DB providers are swappable behind `LLMInterface` / `VectorDBInterface`, with an end-to-end **LangChain + Hugging Face** pipeline for PDF ingestion, chunking, embedding generation, and retrieval
- **Asynchronous REST APIs** with **FastAPI**, **Pydantic** request/response schemas, **Motor** (async MongoDB driver), and **dependency injection** for document ingestion, querying, and chat — endpoints validated end-to-end with the bundled **Postman** collection
- **LLM + vector DB integration** — **Qdrant** and **MongoDB** working together with prompt engineering and conversation-aware context to generate grounded, source-aware RAG responses
- **Streamlit UI** for the full pipeline: document upload, processing, indexing, semantic search, and RAG chat
- **Dockerized** local infrastructure via `docker-compose` for reproducible setup

## 🏗️ Architecture

- **`src/`** — the FastAPI backend
  - `routes/` — API routes (`base`, `auth`, `data`, `nlp`) + `dependencies.py` (auth, roles, rate limits)
  - `controllers/` — business logic (upload validation, chunking, project/asset management)
  - `models/` — Mongo-backed data models and schemes (`Project`, `Asset`, `DataChunk`)
  - `stores/llm/` — LLM & embedding provider abstraction (`LLMInterface` + OpenAI/Cohere/HuggingFace providers)
  - `stores/vectordb/` — vector DB abstraction (`VectorDBInterface` + Qdrant provider)
- **`streamlit_app/`** — two Streamlit frontends that call the backend purely over HTTP (no RAG logic in the UI):
  `admin_app.py` (TA dashboard: upload / process / index / manage materials) and `student_app.py` (student chat)
- **`docker/`** — `docker-compose.yaml` to run MongoDB locally

## 📦 Requirements

- Python 3.12 (tested; 3.10 also passes the test suite)
- MongoDB (via Docker, or your own instance)
- An OpenRouter API key for generation (`OPENAI_API_KEY` + `OPENAI_API_URL=https://openrouter.ai/api/v1`)
- ~2.5 GB disk and ~2 GB RAM for the local `BAAI/bge-m3` embedding model (downloaded on first start; set `HF_HOME` to choose the cache location)

## 🚀 Getting started

### 1. Clone & create an environment

```bash
git clone https://github.com/<your-username>/RAGForge.git
cd RAGForge

conda create -n ragforge python=3.12
conda activate ragforge
```

> On Windows with little space on C:, create the environment on another drive and keep all caches there, e.g.
> `conda create -p E:\envs\mini-rag python=3.12`, then
> `conda env config vars set -p E:\envs\mini-rag PYTHONNOUSERSITE=1 HF_HOME=E:\hf_cache PIP_CACHE_DIR=E:\pip_cache`
> (`PYTHONNOUSERSITE=1` stops packages from the user site-packages on C: leaking into the environment).

### 2. Start MongoDB

```bash
cd docker
cp .env.example .env      # set MONGO_INITDB_ROOT_USERNAME / PASSWORD
docker compose up -d
cd ..
```

### 3. Configure & run the backend

```bash
cd src
pip install -r requirements.txt

cp env.example .env
```

Fill in `.env` — at minimum:

| Variable | Description |
|---|---|
| `APP_NAME`, `APP_VERSION` | App metadata returned by the health endpoint |
| `FILE_ALLOWED_TYPES` | Allowed upload MIME types, e.g. `["application/pdf","text/plain"]` |
| `FILE_MAX_SIZE` | Max upload size in MB |
| `FILE_DEFAULT_CHUNK_SIZE` | Byte-read chunk size for streaming uploads |
| `MONGODB_URL`, `MONGODB_DATABASE` | Mongo connection string & DB name |
| `GENERATION_BACKEND` / `EMBEDDING_BACKEND` | `OPENAI`, `COHERE` or `HUGGINGFACE` (local; needs `transformers` / `sentence-transformers`). OpenAI-compatible gateways (e.g. OpenRouter) use `OPENAI` + `OPENAI_API_URL` |
| `OPENAI_API_KEY` / `OPENAI_API_URL` / `COHERE_API_KEY` | Provider API keys / optional OpenAI-compatible base URL |
| `GENERATION_MODEL_ID` / `EMBEDDING_MODEL_ID` / `EMBEDDING_MODEL_SIZE` | Model selection |
| `INPUT_DAFAULT_MAX_CHARACTERS` | Max characters of one text sent to the **embedding** model — keep ≥ process `chunk_size` (default 1000) |
| `GENERATION_DAFAULT_MAX_INPUT_CHARACTERS` | Max characters of one message sent to the **generation** model (default 20000) |
| `GENERATION_DAFAULT_MAX_TOKENS` / `GENERATION_DAFAULT_TEMPERATURE` | Generation output length / temperature |
| `VECTOR_DB_BACKEND`, `VECTOR_DB_PATH`, `VECTOR_DB_DISTANCE_METHOD` | Qdrant config (local, file-based by default) |
| `PRIMARY_LANG`, `DEFAULT_LANG` | RAG prompt template locale (`en` / `ar`) |

**Security settings (required).** The API refuses to start without `JWT_SECRET_KEY`. From `src/`:

```bash
python -m helpers.security secret   # -> JWT_SECRET_KEY
python -m helpers.security hash     # -> ADMIN_PASSWORD_HASH (TA password) - run again for STUDENT_ACCESS_CODE_HASH
```

| Variable | Description |
|---|---|
| `JWT_SECRET_KEY` | token signing key (>= 32 chars) |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD_HASH` | TA login (scrypt hash, never the plain password) |
| `STUDENT_ACCESS_CODE_HASH` | hash of the course access code the TA shares with students |
| `STUDENT_PROJECT_ID` | the course project students can ask about (the TA uploads materials to it) |
| `ADMIN_TOKEN_EXPIRE_MINUTES` / `STUDENT_TOKEN_EXPIRE_MINUTES` | token lifetimes (default 60 / 480) |
| `LOGIN_RATE_LIMIT` / `LOGIN_RATE_LIMIT_WINDOW_SECONDS` | login attempts per client IP (default 10 per 900 s) |
| `STUDENT_ANSWER_RATE_LIMIT_PER_MINUTE` / `_PER_HOUR` | questions per student session (default 5 / 60) |

Run the API (**one worker**: embedded Qdrant and the in-memory rate limiter are per-process):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Verify it's up:

```bash
curl http://127.0.0.1:8000/api/v1/
# {"app_name":"RAGForge","app_version":"0.1"}
```

A ready-to-import [Postman collection](src/assets/mini-rag-app.postman_collection.json) is included for exploring the API directly.

### 4. Run the Streamlit apps

```bash
pip install -r streamlit_app/requirements.txt
cp streamlit_app/.env.example streamlit_app/.env   # API_BASE_URL, ADMIN_DEFAULT_PROJECT_ID (= STUDENT_PROJECT_ID)

streamlit run streamlit_app/admin_app.py   --server.port 8502   # TA dashboard (keep this URL private)
streamlit run streamlit_app/student_app.py --server.port 8501   # student chat
```

- **TA:** log in with `ADMIN_USERNAME` + password → upload PDFs (type Lecture / Lab / Other, course, topic) →
  **Process documents** → **Build search index** → check the status table. Delete materials from the Documents tab.
- **Students:** open the student app, enter the course access code (+ optional first name) and chat in Arabic or English.
  The conversation lives only in the browser session (`st.session_state`) and is sent as `chat_history`.

Permissions are enforced by the backend (admin token vs student token), not by the UI.

## 🔌 API reference

All routes are mounted under `/api/v1` and scoped to a `project_id` path parameter.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/` | Health check |
| `POST` | `/api/v1/data/upload/{project_id}` | Upload a document |
| `POST` | `/api/v1/data/process/{project_id}` | Chunk uploaded file(s) |
| `POST` | `/api/v1/nlp/index/push/{project_id}` | Embed chunks & push into the vector index |
| `GET` | `/api/v1/nlp/index/info/{project_id}` | Vector collection metadata |
| `POST` | `/api/v1/nlp/index/search/{project_id}` | Semantic search |
| `POST` | `/api/v1/nlp/index/answer/{project_id}` | RAG question answering |
| `POST` | `/api/v1/auth/admin/login` | TA login → admin token |
| `POST` | `/api/v1/auth/student/login` | course access code → student token |
| `GET` | `/api/v1/auth/me` | validate a token |
| `GET` | `/api/v1/data/assets/{project_id}` | course materials with processing / indexing status |
| `DELETE` | `/api/v1/data/asset/{project_id}/{file_id}` | delete one material (file, chunks, vectors) |

**Access:** health and login are public; `answer` needs a student or admin token (students only for
`STUDENT_PROJECT_ID`, rate limited); every other endpoint needs an admin token. Send `Authorization: Bearer <token>`.

### C course assistant options (all optional, backward compatible)

**Upload** (`multipart/form-data`): `file` plus optional `doc_type` (`lecture` | `lab` | `other`, default `other`), `course`, `topic`.
They are stored in the asset's `asset_config` together with the original file name and copied into every chunk's metadata
(`document_name`, `doc_type`, `page_number` (1-based), `course`, `topic`, `file_id`).

**Search / answer** body:

```json
{
  "text": "ايه الفرق بين while و do-while؟",
  "limit": 5,
  "doc_type": "lecture",
  "chat_history": [
    {"role": "user", "content": "explain the for loop"},
    {"role": "assistant", "content": "A for loop has three parts..."}
  ]
}
```

- `doc_type` restricts retrieval to lecture / lab / other chunks.
- `chat_history` (answer only) is the client-side conversation (e.g. Streamlit `st.session_state`); the backend is stateless.
  Only `user` / `assistant` roles are accepted; the last `RAG_HISTORY_MAX_MESSAGES` turns are used.
- Answers follow the student's language (Arabic or English templates), use ```` ```c ```` for code and ```` ```text ```` for output.
- The answer response adds `sources` (built by the backend from retrieved chunk metadata, never from LLM text):
  `sources.cited` = excerpts the model reported using (validated against what was retrieved; empty when the answer is
  not in the course material), `sources.retrieved` = everything retrieved, with similarity scores.

> Changing `EMBEDDING_MODEL_ID` / `EMBEDDING_MODEL_SIZE` requires re-indexing. Existing collections built with another
> model are refused (logged) instead of being mixed; use a new `project_id` or push with `do_reset=1`.

## 🧪 Tests

Run from the repository root (`pytest.ini` points at `src/`):

```bash
pip install -r tests/requirements.txt
python -m pytest                       # unit + integration + Streamlit (integration needs local MongoDB; uses a temp DB)
RUN_LIVE_TESTS=1 python -m pytest tests/live -s      # real OpenRouter + Hugging Face calls (uses src/.env)
python tests/eval/run_embedding_eval.py BAAI/bge-m3  # retrieval benchmark (EN / AR / mixed)
python tests/eval/run_generation_eval.py --models <openrouter-model-id>   # generation benchmark (12 API calls)
python tests/e2e/run_e2e.py --model <openrouter-model-id>                 # full platform run: backend + both Streamlit apps
```

Unit and integration tests never call external APIs and never touch existing MongoDB databases, Qdrant collections or `src/assets`.

## 🛠️ Tech stack

**Backend:** FastAPI · Motor (async MongoDB) · Qdrant · LangChain (text splitting) · PyMuPDF · OpenAI SDK (OpenRouter) · sentence-transformers (BAAI/bge-m3)
**Frontend:** Streamlit · Requests
**Security:** PyJWT (HS256 tokens) · scrypt password hashing · in-memory rate limiting

## 📄 License

See [LICENSE.txt](LICENSE.txt).

## 👤 Author

**Ehab Ashraf** — AI/ML Engineer (NLP, RAG, Agentic AI)
[GitHub](https://github.com/ehab-ashraf) · [LinkedIn](https://linkedin.com/in/ehab-ashraf)

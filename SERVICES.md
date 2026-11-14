# External services used by MINI-RAG (CIT014 C course assistant)

> Keep this file up to date whenever a service is added, removed or changed.
> **No secrets here** (the repository is public). Key values live only in the local files listed under
> "Where the keys are" - never commit them.

_Last updated: 2026-11-14_

## Live production stack (all free, no credit card)

| # | Service | Used for | Plan / limits (free) | Dashboard | Config keys (names only) |
|---|---|---|---|---|---|
| 1 | **Streamlit Community Cloud** | Hosts the public student page (`streamlit_app/student_app.py`) with the embedded FastAPI backend | Free. App sleeps after 12 h without visitors (next visit wakes it ~1 min) | https://share.streamlit.io | App settings -> **Secrets** (pasted from `streamlit-secrets-student.toml`) |
| 2 | **GitHub** | Source code. Streamlit deploys from it | Public repo `EhabAshraf32/RAGForge-End-to-End-RAG-Knowledge-Platform`, branch **`deploy/public-platform`** | https://github.com/EhabAshraf32/RAGForge-End-to-End-RAG-Knowledge-Platform | - |
| 3 | **MongoDB Atlas** | Main database: projects, documents (assets), chunks, and the PDFs themselves (GridFS) | **M0 free**: 512 MB. Cluster `mini-rag` (AWS Frankfurt), project "Default project", org "personal", database `mini_rag_prod`, DB user `minirag_app`. Paused after ~30 days with no connections (resume in dashboard) | https://cloud.mongodb.com | `MONGODB_URL`, `MONGODB_DATABASE`, `STORE_FILES_IN_DB` |
| 4 | **Qdrant Cloud** | Vector search index (1024-dim, cosine), collection `collection_cprog2026` | **Free cluster** `mini-rag-course` (package `free2`, AWS eu-central-1): 1 GB RAM / 4 GB disk. **Suspended after 1 week unused, deleted 4 weeks later** - open the app weekly or resume in the dashboard | https://cloud.qdrant.io | `QDRANT_URL`, `QDRANT_API_KEY` |
| 5 | **Cloudflare Workers AI** | (a) Embeddings `@cf/baai/bge-m3` (same model as local BAAI/bge-m3). (b) Last-resort answer model `@cf/meta/llama-3.3-70b-instruct-fp8-fast`, capped at 20/day | **10,000 neurons/day** shared by both uses; beyond that requests fail until 00:00 UTC (never billed on the Free plan) | https://dash.cloudflare.com -> AI -> Workers AI | `EMBEDDING_BACKEND=CLOUDFLARE`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` |
| 6 | **Groq** | Answer models #1-#3: `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-20b` | Per model: **1,000 requests/day, 8,000 tokens/min** | https://console.groq.com | `GROQ_API_KEY`, `GENERATION_BACKEND=GROQ`, `GENERATION_MODEL_ID` |
| 7 | **Google AI Studio (Gemini)** | Answer models #4-#5: `gemini-3.5-flash`, `gemini-3.5-flash-lite` | Free tier; limits not published, shown per project in AI Studio. Free-tier prompts may be used by Google to improve its products | https://aistudio.google.com | `GEMINI_API_KEY` |
| 8 | **OpenRouter** | Answer model #6: `nvidia/nemotron-3.5-super-vl:free` | **50 requests/day** for the account (no credits bought), 20/min | https://openrouter.ai | `OPENAI_API_KEY`, `OPENAI_API_URL` |

Answer-model order (each used only when the previous ones fail or hit a limit) is set in
`GENERATION_BACKEND` / `GENERATION_MODEL_ID` + `GENERATION_FALLBACKS`:
Groq gpt-oss-120b -> Groq qwen3.8-27b -> Groq gpt-oss-20b -> Gemini 3.5-flash -> Gemini 3.5-flash-lite -> OpenRouter nemotron -> Cloudflare llama-3.3-70b (#20/day).

## Local development only

| Service / tool | Used for | Link |
|---|---|---|
| Hugging Face Hub | Download of `BAAI/bge-m3` for local embeddings (cache in `E:\From-Transformers-To-Proffisional\.hf_cache`) | https://huggingface.co/BAAI/bge-m3 |
| Docker Desktop + MongoDB container | Local database for development and tests (`docker/docker-compose.yaml`) | https://www.docker.com/products/docker-desktop |
| Conda environment | Python 3.12 env at `E:\From-Transformers-To-Proffisional\envs\mini-rag` | https://www.anaconda.com |

## Accounts / keys created during setup that production does NOT use (clean up)

| Service | What exists | Why not used | Action |
|---|---|---|---|
| **Render** | Account "Ehab's workspace" + API key | Requires a credit card even for the free web service | Delete the API key (Account Settings -> API Keys) |
| **MongoDB Atlas service account** `mini-rag-setup` | Org-owner API credentials used once to create the cluster/user | Setup only | Delete it (Applications -> Service Accounts) |
| **Qdrant Cloud management key** | Account-level key used once to create the cluster | Setup only (the app uses a separate database key) | Delete it (Access Management -> Cloud Management Keys) |
| **GitHub Models** token (`github_pat_...`) | Fine-grained token with Models: read | Service no longer answers (returns plain `OK`) | Delete it (GitHub Settings -> Personal access tokens) |
| **MongoDB Atlas "Model API Key"** (`al-...`) | AI-model key, not a database key | Not applicable | Delete if unused |

## Evaluated and rejected (for reference)

| Service | Reason |
|---|---|
| Hugging Face Spaces (Docker) | Free accounts can no longer create Docker/Gradio CPU Spaces (PRO $9/month required) |
| Render | Credit card required |
| Cerebras | Only a one-time trial with a required payment method |
| Cohere | Excluded by the project owner (paid service dependency) |
| Extra OpenRouter `:free` models | All share the same 50/day account limit |

## Where the keys are (local files, outside the repository - private)

| File | Contents |
|---|---|
| `E:\From-Transformers-To-Proffisional\streamlit-secrets-student.toml` | Everything the public student app needs (paste into Streamlit -> Secrets) |
| `E:\From-Transformers-To-Proffisional\streamlit-secrets-admin.toml` | Same + admin credentials; used by `add_course_files.bat` |
| `E:\From-Transformers-To-Proffisional\production-credentials.txt` | Student access code, admin login, Atlas DB user password |
| `E:\From-Transformers-To-Proffisional\deploy-secrets.env` | All provider API keys and setup credentials |
| `Mini-Rag\src\.env` | Local development settings |

## Monitoring free limits

- **Groq**: console.groq.com -> Usage / Limits
- **Gemini**: aistudio.google.com -> rate-limit page of the project
- **OpenRouter**: https://openrouter.ai/activity (or `GET https://openrouter.ai/api/v1/key`)
- **Cloudflare**: dashboard -> AI -> Workers AI -> usage (neurons/day)
- **Atlas**: cluster -> Metrics (data size of 512 MB)
- **Qdrant Cloud**: cluster status (resume if suspended)
- **Streamlit**: app -> Manage app -> logs

## Change log

| Date | Change |
|---|---|
| 2026-11-14 | Initial registry: Streamlit Cloud, GitHub, Atlas, Qdrant Cloud, Cloudflare, Groq, Gemini, OpenRouter. GitHub Models and Render recorded as not used. |

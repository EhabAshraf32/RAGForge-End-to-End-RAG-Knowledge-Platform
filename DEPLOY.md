# Public deployment (zero cost, no credit card)

Free backend hosts tested for this project either need a card (Render's API and dashboard asked for one;
Fly.io, Railway, Koyeb, Oracle) or a paid plan (Hugging Face Docker Spaces need PRO). So the existing FastAPI
backend runs **inside each Streamlit app** (`API_BASE_URL = "embedded"`: real uvicorn on 127.0.0.1, same
code, auth, roles, rate limits and RAG pipeline), and all data lives in free external services:

```
 Student browser ──https──▶ Streamlit Cloud: student_app.py ─┐  (embedded FastAPI, localhost only)
 TA browser ──────https──▶ Streamlit Cloud: admin_app.py ────┤
                                                              ├─▶ MongoDB Atlas M0  (assets, chunks, PDFs in GridFS)
                                                              ├─▶ Qdrant Cloud free (vectors, 1024-d cosine)
                                                              ├─▶ Cloudflare Workers AI @cf/baai/bge-m3 (embeddings)
                                                              └─▶ OpenRouter :free model (answers)
```

| Part | Service | Free limits |
|---|---|---|
| Portals + backend | Streamlit Community Cloud | Apps sleep after 12 h without visitors (next visit wakes them, ~1 min). |
| Database + PDFs | MongoDB Atlas M0 | 512 MB. Paused after ~30 days without connections (resume in Atlas). |
| Vectors | Qdrant Cloud free (`mini-rag-course`, AWS eu-central-1) | 1 GB RAM / 4 GB disk. **Suspended after 1 week unused, deleted 4 weeks later.** |
| Embeddings | Cloudflare Workers AI bge-m3 | 10,000 neurons/day (~9 M tokens); then fails until 00:00 UTC, never billed on Free. |
| Answers | OpenRouter `nvidia/nemotron-3.5-super-vl:free` | **50 requests/day per account** (20/min). |

Secrets are prepared locally (outside the repository) in:
`E:\From-Transformers-To-Proffisional\streamlit-secrets-student.toml` and `...-admin.toml`.
The student app has no admin credentials; the admin app has no student access code.

## Steps
1. **MongoDB Atlas**: Database -> Connect -> Drivers -> copy `mongodb+srv://...` (with the user's password);
   Network Access -> `0.0.0.0/0`. Put it in both TOML files as `MONGODB_URL`.
2. **Streamlit Community Cloud** (https://share.streamlit.io, sign in with GitHub) -> Create app ->
   "Deploy a public app from GitHub":
   - Repository `EhabAshraf32/RAGForge-End-to-End-RAG-Knowledge-Platform`, branch `deploy/public-platform`,
     main file `streamlit_app/student_app.py`, App URL e.g. `cprog-c-ta`.
   - Advanced settings -> Python **3.12** -> Secrets: paste the whole `streamlit-secrets-student.toml`.
   - Repeat with main file `streamlit_app/admin_app.py`, App URL e.g. `cprog-c-admin`, and `streamlit-secrets-admin.toml`.

## Using it
- **TA**: admin URL -> log in (credentials in `production-credentials.txt`) -> Upload PDFs -> Process documents ->
  Build search index -> check the status table.
- **Students**: student URL -> course access code -> ask in English or Arabic.
- **Update**: push to `deploy/public-platform`; Streamlit Cloud redeploys both apps automatically.

## Monitoring
- OpenRouter usage: `GET https://openrouter.ai/api/v1/key`. Cloudflare: dashboard -> AI -> Workers AI.
- Atlas: cluster metrics (data size). Qdrant Cloud: cluster status (resume if suspended).
- Streamlit Cloud: app -> Manage app -> logs.

Nothing is billed when a limit is reached: answers fail with a clear message until the limit resets.

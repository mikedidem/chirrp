# Deploying CHIRRP as a free hosted demo

Goal: a single **HTTPS** link you can share. Architecture: **one Docker container**
(FastAPI serves the built Angular UI *and* the API on one port — no CORS, relative
`apiUrl`) on a **free Hugging Face Docker Space**, using a **free Neon Postgres**
(pgvector) database so RAG and the RAG↔PINN regulatory-context feature work.

Everything except RAG answer synthesis runs without an API key. HTTPS is required
for the voice input to work — Hugging Face provides it automatically.

Prereqs: a free [Neon](https://neon.tech) account, a free
[Hugging Face](https://huggingface.co) account, and `git`.

---

## Part A — Free database (Neon)

1. Create a Neon project → copy its **connection string**
   (`postgresql://USER:PASSWORD@HOST/neondb?sslmode=require`).
2. Enable pgvector + load the data. The fastest way (no re-embedding, no Gemini
   quota) is to copy your already-built local corpus into Neon:
   ```powershell
   # 1) dump the local policy_rag DB (Docker container must be running)
   docker exec llm_interface_db pg_dump -U rag -d policy_rag --no-owner --no-privileges > policy_rag.sql

   # 2) prepare Neon, then restore (psql ships with PostgreSQL; or use Neon's SQL editor)
   psql "postgresql://USER:PASSWORD@HOST/neondb?sslmode=require" -c "CREATE EXTENSION IF NOT EXISTS vector;"
   psql "postgresql://USER:PASSWORD@HOST/neondb?sslmode=require" -f policy_rag.sql
   ```
   *(Alternative: point the app at Neon and run `init_db` + `POST /rag/reindex` —
   but that re-embeds via Gemini and needs quota. The dump/restore above avoids that.)*
3. Note: the **RAG** path (sync `psycopg`) handles `?sslmode=require` fine. The
   **sessions** path (async `asyncpg`) may need `?ssl=require` instead; sessions are
   best-effort, so if they don't persist on Neon the rest of the demo is unaffected.

---

## Part B — Hugging Face Space

1. **Create the Space:** huggingface.co → *New* → *Space* → **SDK: Docker** → *Blank* →
   choose a neutral name (for anonymous review, don't use your real name).
2. **Add the HF header to `README.md`.** The Space's `README.md` must start with:
   ```yaml
   ---
   title: CHIRRP Hydro-AI Demo
   emoji: 💧
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
   (Add it above the existing README content, or use a short Space-only README.)
3. **Get the files into the Space repo.** A Space *is* a git repo:
   ```bash
   git clone https://huggingface.co/spaces/<you>/<space> hf-space
   # copy the project in (exclude local junk):
   #   Dockerfile, .dockerignore, requirements.txt, chirrp/, hydroproject-ui/ (NO node_modules),
   #   README.md (with the header above)
   # CRITICAL — also copy the untracked model/validation files (required, not in git):
   #   chirrp/core/pinn/artifacts/stage1.pth.tar
   #   chirrp/core/pinn/artifacts/stage2.pth.tar
   #   chirrp/core/validation/artifacts/reference_heads.npz
   #   chirrp/core/validation/bin/mf2005.*
   cd hf-space
   git add -A
   git commit -m "CHIRRP demo"
   git push
   ```
   Do **NOT** copy `.env`, `gemini_api.txt`, `.venv`, `node_modules`, or `dist`.
4. **Set Secrets** (Space → Settings → *Variables and secrets*), never in the repo:
   - `DATABASE_URL` = your Neon string
   - `RAG_DATABASE_URL` = your Neon string
   - `GEMINI_API_KEY` = your key (optional — only enables RAG *answers*; retrieval +
     RAG↔PINN context work without it)
5. Hugging Face builds the image (a few minutes — PyTorch is large) and runs it.
   Your link: **`https://<you>-<space>.hf.space`**.

---

## Verify the live demo
- Open the link → the Overview loads over HTTPS.
- `/studio` → run "reduce pumping by 15%" → decision readout + **Regulatory context** appears.
- `/accuracy` → click an anchor rate → real RMSE/R².
- Voice mic works (HTTPS) in Chrome/Edge.
- `…hf.space/healthz` → `{"status":"API is running"}`.

## Test the image locally first (optional)
```powershell
docker build -t chirrp .
docker run -p 7860:7860 -e DATABASE_URL="<neon>" -e RAG_DATABASE_URL="<neon>" -e GEMINI_API_KEY="<key>" chirrp
# open http://localhost:7860
```

## Notes
- The free Space **sleeps after inactivity** and wakes on the next visit (cold start ~30s).
- First build is slow (PyTorch/SciPy); later pushes are faster.
- For double-blind review: neutral Space name, no personal info on the page (there is none),
  and you're sharing a running app — not the git history.

# CHIRRP — single-container image for a free hosted demo (e.g. Hugging Face
# Docker Space). FastAPI serves the built Angular UI AND the API on one port,
# so there is no CORS and no separate Node runtime.
#
# Build:  docker build -t chirrp .
# Run:    docker run -p 7860:7860 --env-file .env chirrp
# syntax=docker/dockerfile:1

# ---------- Stage 1: build the Angular UI (browser-only, relative apiUrl) ----------
FROM node:20-bookworm-slim AS ui
WORKDIR /ui
COPY hydroproject-ui/package*.json ./
RUN npm ci
COPY hydroproject-ui/ ./
RUN npx ng build --configuration=hosted

# ---------- Stage 2: Python backend that also serves the built UI ----------
# Python 3.11 (flopy>=3.9.3 and current fastapi need >=3.10; the app code is 3.9-compatible).
FROM python:3.11-slim-bookworm AS app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STATIC_DIR=/app/uidist \
    RAG_CHUNK_MODE=deterministic
WORKDIR /app

# CPU-only PyTorch keeps the image far smaller; psycopg[binary] bundles libpq.
COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Backend source + trained model / MODFLOW validation artifacts.
# NOTE: stage1/stage2.pth.tar, reference_heads.npz and bin/mf2005.* are required
# and must be present in the build context (they are untracked in git — make sure
# they are committed to the Space repo or copied into the build context).
COPY chirrp/ ./chirrp/
RUN chmod +x chirrp/core/validation/bin/mf2005.linux || true

# Built UI from stage 1 → served by FastAPI at STATIC_DIR.
COPY --from=ui /ui/dist/ui-app/browser/ ./uidist/

# DATABASE_URL / RAG_DATABASE_URL / GEMINI_API_KEY are provided at RUNTIME
# (Hugging Face "Secrets"), never baked into the image.
EXPOSE 7860
CMD ["uvicorn", "chirrp.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]

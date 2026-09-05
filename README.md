# NextScene

**What to watch next — predicted from your watch history, explained in plain language.**

NextScene is an end-to-end movie recommender: a causal sequence transformer trained on MovieLens 20M, CLIP-based item embeddings, vector retrieval, and an optional LightGBM re-ranker — wrapped in a Netflix-style web app with per-pick explanations.

## Features

- **Personal library** — search the catalog, rate titles, and build a watchlist (`Cmd+K` search)
- **Sequence-based recommendations** — unlocks after 3+ movies in your library
- **Explanations** — every pick includes a plain-language "why" (ranker features + shared genres)
- **Genre preferences** — steer recommendations toward your favorite genres
- **Production-ready stack** — JWT auth, Postgres/pgvector, Docker, CI, and deploy configs for Render + Vercel

## How it works

```
Watch history → Causal transformer → Next-item embedding → Vector retrieval → Re-rank (optional) → Explain
```

1. **Encode** — fuse CLIP text + poster embeddings per movie (512-dim), stored in Postgres/pgvector
2. **Model** — SASRec-style causal transformer predicts the next-item embedding from watch history
3. **Retrieve** — cosine similarity over the catalog (numpy, or FAISS if enabled)
4. **Re-rank** *(optional)* — LightGBM over retrieval and user-history features
5. **Explain** — ranker feature importance plus shared-genre reasoning

## Tech stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS 4, TanStack Query, Zustand, Radix UI |
| **Backend** | FastAPI, SQLAlchemy 2, Alembic, Pydantic, JWT auth, Prometheus metrics |
| **ML** | PyTorch, Hugging Face Transformers (CLIP), InfoNCE contrastive loss, LightGBM |
| **Data** | MovieLens 20M, TMDb/OMDb/IMDb enrichment, FAISS (optional) |
| **Infra** | Docker, GitHub Actions CI, Render (API + Postgres), Vercel (frontend) |

Trained and evaluated on the full MovieLens 20M dataset (27,278 catalog items with embeddings, 138k user sequences). The backend has 58 Python modules across 8 route groups, 129 automated tests, and 3 Alembic migrations — CI runs lint, tests, and Docker builds on every PR.

## Quick start

### Prerequisites

- Python 3.12, Node.js 22, Docker (for Postgres)
- MovieLens 20M files in `data/movielens/` (see [data/movielens/README.txt](data/movielens/README.txt))

### 1. Database

```bash
docker compose up postgres -d
alembic upgrade head
```

### 2. Backend

```bash
pip install -r requirements-ml.txt
# Configure DATABASE_URL, JWT_SECRET, and optional API keys in .env or environment
uvicorn app.main:app --reload --app-dir backend
```

API docs: http://localhost:8000/docs

### 3. Frontend

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000, register, add a few movies (`Cmd+K`), and recommendations unlock after 3+ titles.

### Full stack (Docker)

```bash
docker compose --profile full up --build
```

## ML pipeline (offline)

Run from repo root with `PYTHONPATH=backend`:

```bash
# Ingest MovieLens + enrich metadata
python scripts/fetch_tmdb_metadata.py
python scripts/download_posters.py
python scripts/generate_clip_embeddings.py

# Build sequences and train
python scripts/build_training_sequences.py --output data/sequences
python scripts/train_transformer.py --output models/transformer-full-v2
python scripts/evaluate_transformer.py

# Optional: FAISS index and ranker
python scripts/build_faiss_index.py
python scripts/train_ranker.py
```

Package serving artifacts for Render:

```bash
./scripts/package_serving_artifacts.sh models/transformer-full-v2 serving-artifacts.tar.gz
```

## Deployment

- **API**: [Render](render.yaml) — Docker backend, managed Postgres, model weights pulled from a GitHub Release
- **Frontend**: Vercel — set `API_URL` to the Render API origin

The portfolio/demo deploy runs a trimmed catalog via `scripts/seed_portfolio.py`.

## Testing

```bash
PYTHONPATH=backend pytest tests -q
cd frontend && npm run lint && npm run build
```

## License

MovieLens data is subject to the [GroupLens usage license](data/movielens/README.txt). Code in this repository is for portfolio and educational use.

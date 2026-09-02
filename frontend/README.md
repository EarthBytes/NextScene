# NextScene — Frontend

Personalized movie recommendations with authentication.

## Setup

```bash
# Run DB migration first (from repo root)
alembic upgrade head

# Backend
uvicorn app.main:app --reload --app-dir backend

# Frontend
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

## Usage

1. Register at http://localhost:3000/register
2. Search and add movies (`Cmd+K` or "Add movies" button)
3. Add at least 3 movies to unlock personalized recommendations
4. Click "Why?" on any pick for a plain-language explanation

## Environment

| Variable | Default |
|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` |

#!/bin/sh
set -e
# Render free tier does not support preDeployCommand; migrate on boot instead.
alembic upgrade head
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --app-dir backend

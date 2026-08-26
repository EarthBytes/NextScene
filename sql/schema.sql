-- Core schema for Generative Recommendation System
-- Requires PostgreSQL with pgvector extension for production embeddings storage

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id          SERIAL PRIMARY KEY,
    signup_ts   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    profile_json JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS items (
    item_id       INTEGER PRIMARY KEY,  -- MovieLens movie_id
    imdb_id       TEXT,
    title         TEXT NOT NULL,
    description   TEXT,
    genres        TEXT[],
    image_url     TEXT,
    metadata_json JSONB DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id BIGSERIAL PRIMARY KEY,
    user_id        INTEGER NOT NULL,
    item_id        INTEGER NOT NULL REFERENCES items(item_id),
    ts             TIMESTAMPTZ NOT NULL,
    type           TEXT NOT NULL CHECK (type IN ('view', 'rating', 'tag', 'click', 'purchase')),
    context_json   JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_interactions_user_ts ON interactions (user_id, ts);
CREATE INDEX IF NOT EXISTS idx_interactions_item_id ON interactions (item_id);

CREATE TABLE IF NOT EXISTS item_embeddings (
    item_id    INTEGER PRIMARY KEY REFERENCES items(item_id),
    vector     vector(512),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_item_embeddings_updated ON item_embeddings (updated_at);

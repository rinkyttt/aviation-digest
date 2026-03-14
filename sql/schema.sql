-- Aviation Digest: one-time Supabase setup
-- Run this in Supabase SQL Editor

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE articles (
  id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  url          TEXT NOT NULL UNIQUE,          -- dedup key
  title        TEXT NOT NULL,
  published_at TIMESTAMPTZ,
  fetched_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  is_aviation  BOOLEAN NOT NULL DEFAULT FALSE,
  raw_content  TEXT,
  summary_en   TEXT,
  summary_zh   TEXT
);

CREATE TABLE digests (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  date          DATE NOT NULL UNIQUE,
  top_articles  JSONB NOT NULL DEFAULT '[]',
  -- shape: [{"article_id":"uuid","rank":1,"score":9.2,"reason":"..."}]
  shownotes_en  TEXT,
  shownotes_zh  TEXT,
  audio_en_url  TEXT,
  audio_zh_url  TEXT,
  email_sent    BOOLEAN NOT NULL DEFAULT FALSE
);

-- Indexes
CREATE INDEX ON articles (published_at DESC);
CREATE INDEX ON articles (is_aviation);
CREATE INDEX ON digests (date DESC);

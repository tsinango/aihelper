CREATE TABLE IF NOT EXISTS knowledge_review_group_build_jobs (
  job_id TEXT PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued', 'running', 'completed', 'failed')),
  reviewer TEXT NOT NULL,
  similarity_threshold NUMERIC(6,5) NOT NULL DEFAULT 0.76000,
  total_candidates INTEGER NOT NULL DEFAULT 0,
  processed_candidates INTEGER NOT NULL DEFAULT 0,
  created_groups INTEGER NOT NULL DEFAULT 0,
  error_message TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

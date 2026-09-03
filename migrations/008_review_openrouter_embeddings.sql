-- OpenRouter metadata for semantic review grouping.  The live vector columns
-- are migrated to Nemotron's 2048 dimensions in migration 009.

ALTER TABLE knowledge_review_groups
  ADD COLUMN IF NOT EXISTS embedding_provider TEXT NOT NULL DEFAULT 'openrouter';

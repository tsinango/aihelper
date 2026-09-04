-- V2 learning retrieval and comparison support.
--
-- This migration is additive. It does not alter V1 tables, create a vector
-- index, or expose a taxonomy. The V2 knowledge set is intentionally small,
-- so retrieval performs an exact scan over the existing 2048-dimensional
-- OpenRouter vectors when they are available.

ALTER TABLE v2_knowledge
  ADD COLUMN IF NOT EXISTS embedding vector(2048),
  ADD COLUMN IF NOT EXISTS embedding_model TEXT;

COMMENT ON COLUMN v2_knowledge.embedding IS
  'Optional OpenRouter Nemotron 2048-dimensional vector; Phase 2 uses exact scans only.';
COMMENT ON COLUMN v2_knowledge.embedding_model IS
  'Embedding model identifier; vectors are comparable only within the same model.';

ALTER TABLE v2_learning_proposals
  ADD COLUMN IF NOT EXISTS comparison_result TEXT NOT NULL DEFAULT 'NEW',
  ADD COLUMN IF NOT EXISTS clarification_question TEXT NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS comparison_reason TEXT NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS related_knowledge_ids BIGINT[] NOT NULL DEFAULT ARRAY[]::BIGINT[];

ALTER TABLE v2_inbox_messages
  DROP CONSTRAINT IF EXISTS v2_inbox_messages_message_type_check;

ALTER TABLE v2_inbox_messages
  ADD CONSTRAINT v2_inbox_messages_message_type_check CHECK (message_type IN (
    'text', 'question', 'clarification', 'proposal', 'confirmation',
    'correction', 'skip', 'unknown', 'summary', 'evidence'
  ));

-- The original status constraint predates UNCLEAR/pending_clarification.
ALTER TABLE v2_learning_proposals
  DROP CONSTRAINT IF EXISTS v2_learning_proposals_status_check;

ALTER TABLE v2_learning_proposals
  ADD CONSTRAINT v2_learning_proposals_status_check CHECK (status IN (
    'pending_confirmation', 'pending_clarification', 'confirmed', 'corrected',
    'skipped', 'unknown', 'rejected', 'superseded'
  ));

ALTER TABLE v2_learning_proposals
  DROP CONSTRAINT IF EXISTS v2_learning_proposals_comparison_result_check;

ALTER TABLE v2_learning_proposals
  ADD CONSTRAINT v2_learning_proposals_comparison_result_check
  CHECK (comparison_result IN ('NEW', 'CONFIRM', 'ENRICH', 'CONFLICT', 'UNCLEAR'));

ALTER TABLE v2_learning_proposals
  DROP CONSTRAINT IF EXISTS v2_learning_proposals_clarification_check;

ALTER TABLE v2_learning_proposals
  ADD CONSTRAINT v2_learning_proposals_clarification_check CHECK (
    (status = 'pending_clarification'
      AND comparison_result IN ('ENRICH', 'CONFLICT', 'UNCLEAR')
      AND char_length(btrim(clarification_question)) > 0)
    OR status <> 'pending_clarification'
  );

ALTER TABLE v2_learning_proposals
  DROP CONSTRAINT IF EXISTS v2_learning_proposals_unclear_status_check;

ALTER TABLE v2_learning_proposals
  ADD CONSTRAINT v2_learning_proposals_unclear_status_check CHECK (
    comparison_result <> 'UNCLEAR'
    OR status NOT IN ('pending_confirmation', 'confirmed')
  );

CREATE INDEX IF NOT EXISTS ix_v2_knowledge_embedding_model
  ON v2_knowledge(embedding_model)
  WHERE active = TRUE AND embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_v2_learning_proposals_comparison
  ON v2_learning_proposals(comparison_result, status, updated_at DESC);

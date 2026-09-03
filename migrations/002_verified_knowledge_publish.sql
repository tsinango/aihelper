-- Verified Knowledge publication workflow.
-- Additive migration: review candidates and their source evidence are retained.

ALTER TABLE verified_knowledge_candidates
  DROP CONSTRAINT IF EXISTS verified_knowledge_candidates_publication_status_check;
ALTER TABLE verified_knowledge_candidates
  ADD CONSTRAINT verified_knowledge_candidates_publication_status_check
  CHECK (publication_status IN ('draft', 'published', 'archived'));

ALTER TABLE verified_knowledge
  DROP CONSTRAINT IF EXISTS verified_knowledge_publication_status_check;
ALTER TABLE verified_knowledge
  ADD CONSTRAINT verified_knowledge_publication_status_check
  CHECK (publication_status IN ('draft', 'published', 'archived'));

-- A source candidate can have multiple immutable publication versions.
ALTER TABLE verified_knowledge
  DROP CONSTRAINT IF EXISTS verified_knowledge_source_candidate_id_key;

ALTER TABLE verified_knowledge
  ADD COLUMN IF NOT EXISTS procedure_steps JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS searchable_text TEXT NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS embedding vector(1024),
  ADD COLUMN IF NOT EXISTS published_by TEXT,
  ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1,
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;

CREATE UNIQUE INDEX IF NOT EXISTS ux_verified_knowledge_key_version
  ON verified_knowledge(knowledge_key, version);
CREATE INDEX IF NOT EXISTS ix_verified_knowledge_publication
  ON verified_knowledge(publication_status, production_answer_allowed);
CREATE INDEX IF NOT EXISTS ix_verified_knowledge_source_candidate
  ON verified_knowledge(source_candidate_id);
CREATE INDEX IF NOT EXISTS ix_verified_knowledge_embedding_hnsw
  ON verified_knowledge USING hnsw(embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS ix_verified_knowledge_searchable_fts
  ON verified_knowledge USING gin(to_tsvector('simple', searchable_text));

-- Backfill fields for records created by the review-only migration.
UPDATE verified_knowledge vk
SET procedure_steps = COALESCE(vc.effective_payload->'procedure_steps', '[]'::jsonb),
    evidence = COALESCE((
      SELECT jsonb_agg(jsonb_build_object(
        'evidence_id', e.evidence_id,
        'source_type', e.source_type,
        'document_id', e.document_id,
        'document_title', e.document_title,
        'page', e.page,
        'chunk_id', e.chunk_id,
        'excerpt', e.excerpt,
        'relation', e.effective_evidence_relation
      ) ORDER BY e.id)
      FROM verified_knowledge_candidate_evidence e
      WHERE e.candidate_id = vk.source_candidate_id
    ), '[]'::jsonb),
    searchable_text = concat_ws(' ', vk.knowledge_key, vk.title,
      coalesce(vk.question_patterns::text, ''), coalesce(vk.scope->'models', '[]'::jsonb)::text,
      coalesce(vk.claims::text, '')),
    updated_at = CURRENT_TIMESTAMP
FROM verified_knowledge_candidates vc
WHERE vc.candidate_id = vk.source_candidate_id;

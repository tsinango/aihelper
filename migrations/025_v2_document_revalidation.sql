-- Phase 5.2 document version revalidation: explicit version lineage, stored
-- comparison results, and a revalidate audit action.
--
-- Additive only.  New versions never overwrite old ones; old Knowledge rows
-- keep serving until their own evidence fails or an engineer explicitly
-- revalidates them.  The lineage column lets answer qualification exclude
-- older-lineage units only when a request explicitly names a newer version
-- of the same document key.

ALTER TABLE v2_document_versions
  ADD COLUMN IF NOT EXISTS previous_version_id BIGINT
    REFERENCES v2_document_versions(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS change_summary JSONB NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN v2_document_versions.previous_version_id IS
  'Human-confirmed predecessor; never guessed from filenames. NULL for first versions.';
COMMENT ON COLUMN v2_document_versions.change_summary IS
  'Stored section-level comparison (added/changed/removed/unmatched) plus the affected Knowledge list.';

CREATE INDEX IF NOT EXISTS ix_v2_document_versions_previous
  ON v2_document_versions(previous_version_id);

-- Revalidation is an auditable Knowledge change alongside edit/confirm.
ALTER TABLE v2_knowledge_history
  DROP CONSTRAINT IF EXISTS v2_knowledge_history_action_check;
ALTER TABLE v2_knowledge_history
  ADD CONSTRAINT v2_knowledge_history_action_check CHECK (action IN (
    'edit', 'deactivate', 'restore', 'move', 'confirm', 'revalidate'
  ));

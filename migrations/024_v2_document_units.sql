-- Phase 4.2 complete document knowledge units: structured details on
-- Knowledge and proposals, plus the document origin of proposal rows.
--
-- Additive only.  ``content`` stays the authoritative readable/retrievable
-- text; ``details_json`` carries the optional structured contract
-- (prerequisites, ordered steps, expected result, exceptions/warnings, rule
-- triggers).  When details are edited without content, content is
-- deterministically re-rendered from details so the two never disagree.

ALTER TABLE v2_knowledge
  ADD COLUMN IF NOT EXISTS details_json JSONB NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN v2_knowledge.details_json IS
  'Optional structured contract for procedure/rule/experience units; content remains the readable text.';

ALTER TABLE v2_learning_proposals
  ADD COLUMN IF NOT EXISTS details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS origin_document_version_id BIGINT
    REFERENCES v2_document_versions(id) ON DELETE SET NULL;

COMMENT ON COLUMN v2_learning_proposals.origin_document_version_id IS
  'Document version a document-learning proposal was extracted from; NULL for text-learning proposals.';

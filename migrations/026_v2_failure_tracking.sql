-- Phase 5.3 failure-driven improvement: record which Knowledge should have
-- matched a failed answer, so the retrieval-improvement gate is checkable.
--
-- Additive only.  The retrieval gate stays closed until at least 10 real
-- questions prove qualified evidence was in the library but not recalled;
-- this column stores the engineer's expected source IDs for exactly that
-- count.  No retrieval behavior changes here.

ALTER TABLE v2_answer_feedback
  ADD COLUMN IF NOT EXISTS expected_knowledge_ids BIGINT[] NOT NULL DEFAULT ARRAY[]::BIGINT[];

COMMENT ON COLUMN v2_answer_feedback.expected_knowledge_ids IS
  'Engineer-declared Knowledge IDs that should have been recalled; feeds the 10-case retrieval-improvement gate.';

-- Make deterministic review grouping the default while retaining an optional
-- V1.1/semantic comparison view.
ALTER TABLE knowledge_review_groups
  ADD COLUMN IF NOT EXISTS grouping_mode TEXT NOT NULL DEFAULT 'deterministic';
ALTER TABLE knowledge_review_groups
  DROP CONSTRAINT IF EXISTS knowledge_review_groups_grouping_mode_check;
ALTER TABLE knowledge_review_groups
  ADD CONSTRAINT knowledge_review_groups_grouping_mode_check
  CHECK (grouping_mode IN ('deterministic', 'v1_1', 'semantic'));

ALTER TABLE knowledge_review_group_build_jobs
  ADD COLUMN IF NOT EXISTS grouping_mode TEXT NOT NULL DEFAULT 'deterministic';
ALTER TABLE knowledge_review_group_build_jobs
  DROP CONSTRAINT IF EXISTS knowledge_review_group_build_jobs_grouping_mode_check;
ALTER TABLE knowledge_review_group_build_jobs
  ADD CONSTRAINT knowledge_review_group_build_jobs_grouping_mode_check
  CHECK (grouping_mode IN ('deterministic', 'v1_1', 'semantic'));

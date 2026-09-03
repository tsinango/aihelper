ALTER TABLE knowledge_review_groups
  ADD COLUMN IF NOT EXISTS review_note TEXT NOT NULL DEFAULT '';

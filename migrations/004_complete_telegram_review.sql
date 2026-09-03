-- Complete Telegram review coverage and explicit answer/scope fields.
--
-- Existing pilot/topic candidates remain the source of grouped knowledge.  The
-- CASE-* candidates below are a review fallback for every imported support
-- case that is not currently attached to any candidate.  They are deliberately
-- blank-answer candidates: a reviewer must transcribe/confirm the historical
-- answer before approval.

ALTER TABLE verified_knowledge_candidates
  ADD COLUMN IF NOT EXISTS answer_text TEXT NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS answer_status TEXT NOT NULL DEFAULT 'pending',
  ADD COLUMN IF NOT EXISTS scope_level TEXT NOT NULL DEFAULT 'unspecified';

ALTER TABLE verified_knowledge_candidates
  DROP CONSTRAINT IF EXISTS verified_knowledge_candidates_answer_status_check;
ALTER TABLE verified_knowledge_candidates
  ADD CONSTRAINT verified_knowledge_candidates_answer_status_check
  CHECK (answer_status IN ('pending', 'approved', 'needs_context', 'rejected', 'duplicate'));

ALTER TABLE verified_knowledge_candidates
  DROP CONSTRAINT IF EXISTS verified_knowledge_candidates_scope_level_check;
ALTER TABLE verified_knowledge_candidates
  ADD CONSTRAINT verified_knowledge_candidates_scope_level_check
  CHECK (scope_level IN ('generic', 'brand', 'family', 'series', 'model', 'conditional', 'unspecified'));

ALTER TABLE verified_knowledge
  ADD COLUMN IF NOT EXISTS answer_text TEXT NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS scope_level TEXT NOT NULL DEFAULT 'unspecified';

ALTER TABLE verified_knowledge
  DROP CONSTRAINT IF EXISTS verified_knowledge_scope_level_check;
ALTER TABLE verified_knowledge
  ADD CONSTRAINT verified_knowledge_scope_level_check
  CHECK (scope_level IN ('generic', 'brand', 'family', 'series', 'model', 'conditional', 'unspecified'));

-- Materialise missing review candidates for databases that already contain
-- imported Telegram cases.  The API repeats this idempotently for later cases.
INSERT INTO verified_knowledge_candidates
  (candidate_id, knowledge_key, title, knowledge_type, scope,
   question_patterns, claims, procedure_steps, conditions, exceptions, warnings,
   confidence, verification_status, review_status, publication_status,
   production_answer_allowed, frequency, ai_payload, effective_payload,
   answer_text, answer_status, scope_level)
SELECT
  'CASE-' || lpad(sc.id::text, 6, '0'),
  'telegram.case.' || sc.id::text,
  COALESCE(NULLIF(left(sc.root_question, 500), ''), 'Telegram support case #' || sc.id),
  'other',
          jsonb_build_object('brands', '[]'::jsonb, 'product_families', '[]'::jsonb,
                             'series', '[]'::jsonb,
                     'models', COALESCE(sc.models, '[]'::jsonb),
                     'hardware_revisions', '[]'::jsonb,
                     'firmware_versions', '[]'::jsonb, 'software_versions', '[]'::jsonb,
                     'operating_modes', '[]'::jsonb),
  CASE WHEN NULLIF(trim(sc.root_question), '') IS NULL
       THEN '[]'::jsonb ELSE jsonb_build_array(sc.root_question) END,
  '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb,
  'low', 'pending', 'pending', 'draft', FALSE, 1,
  jsonb_build_object('candidate_id', 'CASE-' || lpad(sc.id::text, 6, '0'),
                     'knowledge_key', 'telegram.case.' || sc.id::text,
                     'title', COALESCE(NULLIF(left(sc.root_question, 500), ''), 'Telegram support case #' || sc.id),
                     'knowledge_type', 'other', 'scope', jsonb_build_object('series', '[]'::jsonb, 'models', COALESCE(sc.models, '[]'::jsonb)),
                     'question_patterns', CASE WHEN NULLIF(trim(sc.root_question), '') IS NULL THEN '[]'::jsonb ELSE jsonb_build_array(sc.root_question) END,
                     'answer_text', ''),
  jsonb_build_object('candidate_id', 'CASE-' || lpad(sc.id::text, 6, '0'),
                     'knowledge_key', 'telegram.case.' || sc.id::text,
                     'title', COALESCE(NULLIF(left(sc.root_question, 500), ''), 'Telegram support case #' || sc.id),
                     'knowledge_type', 'other', 'scope', jsonb_build_object('series', '[]'::jsonb, 'models', COALESCE(sc.models, '[]'::jsonb)),
                     'question_patterns', CASE WHEN NULLIF(trim(sc.root_question), '') IS NULL THEN '[]'::jsonb ELSE jsonb_build_array(sc.root_question) END,
                     'answer_text', ''),
  COALESCE((
    SELECT string_agg(NULLIF(message.value->>'text', ''), E'\n' ORDER BY message.ordinality)
    FROM jsonb_array_elements(COALESCE(sc.messages, '[]'::jsonb)) WITH ORDINALITY AS message(value, ordinality)
    WHERE COALESCE(message.value->>'author', '') <> COALESCE(sc.root_author, '')
  ), ''),
  'pending', CASE WHEN jsonb_array_length(COALESCE(sc.models, '[]'::jsonb)) > 0 THEN 'model' ELSE 'generic' END
FROM support_cases sc
WHERE NOT EXISTS (
  SELECT 1 FROM verified_knowledge_candidate_cases cc
  WHERE cc.support_case_id = sc.id
)
ON CONFLICT (candidate_id) DO NOTHING;

INSERT INTO verified_knowledge_candidate_cases(candidate_id, support_case_id, case_position)
SELECT 'CASE-' || lpad(sc.id::text, 6, '0'), sc.id, 0
FROM support_cases sc
WHERE EXISTS (
  SELECT 1 FROM verified_knowledge_candidates vc
  WHERE vc.candidate_id = 'CASE-' || lpad(sc.id::text, 6, '0')
)
  AND NOT EXISTS (
    SELECT 1 FROM verified_knowledge_candidate_cases cc
    WHERE cc.support_case_id = sc.id
  )
ON CONFLICT (candidate_id, support_case_id) DO NOTHING;

-- Preserve all Telegram messages in the review thread with a safe default role.
INSERT INTO verified_knowledge_candidate_message_roles
  (candidate_id, support_case_id, message_index, ai_role, effective_role, ai_reason)
SELECT 'CASE-' || lpad(sc.id::text, 6, '0'), sc.id,
       (message.ordinality - 1)::integer, 'unconfirmed_claim', 'unconfirmed_claim',
       'No grouped AI candidate exists; reviewer must classify this message.'
FROM support_cases sc
JOIN jsonb_array_elements(COALESCE(sc.messages, '[]'::jsonb)) WITH ORDINALITY AS message(value, ordinality) ON TRUE
JOIN verified_knowledge_candidate_cases cc
  ON cc.support_case_id=sc.id AND cc.candidate_id='CASE-' || lpad(sc.id::text, 6, '0')
ON CONFLICT (candidate_id, support_case_id, message_index) DO NOTHING;

CREATE INDEX IF NOT EXISTS ix_review_candidates_answer_status
  ON verified_knowledge_candidates(answer_status, review_status);
CREATE INDEX IF NOT EXISTS ix_review_candidates_scope_level
  ON verified_knowledge_candidates(scope_level);

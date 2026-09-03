-- Replace the old local BGE-M3 vector shape with Nemotron Embed's 2048 dims.
-- Existing vectors are intentionally cleared because vectors from different
-- models/dimensions must never be mixed in retrieval.

DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw;
DROP INDEX IF EXISTS ix_verified_knowledge_embedding_hnsw;
DROP INDEX IF EXISTS ix_learning_examples_embedding_hnsw;
DROP INDEX IF EXISTS ix_case_memory_embedding_hnsw;

UPDATE document_chunks SET embedding = NULL, embedding_model = NULL WHERE embedding IS NOT NULL;
UPDATE verified_knowledge SET embedding = NULL WHERE embedding IS NOT NULL;
UPDATE knowledge_learning_examples SET embedding = NULL, embedding_model = NULL WHERE embedding IS NOT NULL;
UPDATE case_knowledge_memory SET embedding = NULL, embedding_model = NULL WHERE embedding IS NOT NULL;
UPDATE review_candidate_embeddings SET embedding = NULL WHERE embedding IS NOT NULL;

ALTER TABLE document_chunks
  ALTER COLUMN embedding TYPE vector(2048) USING NULL::vector(2048);
ALTER TABLE verified_knowledge
  ALTER COLUMN embedding TYPE vector(2048) USING NULL::vector(2048);
ALTER TABLE knowledge_learning_examples
  ALTER COLUMN embedding TYPE vector(2048) USING NULL::vector(2048);
ALTER TABLE case_knowledge_memory
  ALTER COLUMN embedding TYPE vector(2048) USING NULL::vector(2048);
ALTER TABLE review_candidate_embeddings
  ALTER COLUMN embedding TYPE vector(2048) USING NULL::vector(2048);

-- This pgvector installation caps both HNSW and IVFFlat indexes below 2048
-- dimensions. Keep exact cosine scans until pgvector is upgraded; the current
-- corpus is small and the SQL retrieval path remains fully correct.

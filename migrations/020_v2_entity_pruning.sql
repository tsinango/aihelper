-- Human-initiated, additive soft-pruning metadata for the V2 organization layer.
-- No row is physically deleted; active=false remains the visibility gate.

ALTER TABLE v2_entities
  ADD COLUMN IF NOT EXISTS deactivated_at TIMESTAMPTZ;

ALTER TABLE v2_entity_relations
  ADD COLUMN IF NOT EXISTS deactivated_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS ix_v2_entities_deactivated_at
  ON v2_entities(deactivated_at)
  WHERE active=FALSE;

CREATE INDEX IF NOT EXISTS ix_v2_entity_relations_deactivated_at
  ON v2_entity_relations(deactivated_at)
  WHERE active=FALSE;

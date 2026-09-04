-- Phase 2.1.2: minimal liveness record for the dedicated Inbox worker.
--
-- This is operational state only.  It does not change the durable job
-- envelope or any V1 table.

CREATE TABLE IF NOT EXISTS v2_inbox_workers (
  worker_name TEXT PRIMARY KEY,
  last_seen_at TIMESTAMPTZ NOT NULL
);

COMMENT ON TABLE v2_inbox_workers IS
  'Last liveness pulse from each dedicated V2 Inbox worker.';

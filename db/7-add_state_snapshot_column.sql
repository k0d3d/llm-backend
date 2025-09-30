-- Migration: add JSONB state_snapshot column and supporting indexes for HITL runs
-- This migration aligns the database schema with the JSONB persistence layer
-- implemented in `llm_backend/core/hitl/persistence.py`.

BEGIN;

-- 1. Add the state_snapshot column if it does not already exist
ALTER TABLE hitl_runs
    ADD COLUMN IF NOT EXISTS state_snapshot JSONB;

-- Ensure existing rows are initialized to an empty JSON object (avoid NULL issues)
UPDATE hitl_runs
SET state_snapshot = '{}'::JSONB
WHERE state_snapshot IS NULL;

-- 2. Create GIN indexes used for JSONB queries
CREATE INDEX IF NOT EXISTS hitl_runs_state_status_idx
    ON hitl_runs ((state_snapshot->>'status'));

CREATE INDEX IF NOT EXISTS hitl_runs_checkpoint_type_idx
    ON hitl_runs ((state_snapshot->'checkpoint_context'->>'type'));

CREATE INDEX IF NOT EXISTS hitl_runs_pending_actions_idx
    ON hitl_runs USING GIN ((state_snapshot->'pending_actions'));

CREATE INDEX IF NOT EXISTS hitl_runs_validation_issues_idx
    ON hitl_runs USING GIN ((state_snapshot->'validation_issues'));

CREATE INDEX IF NOT EXISTS hitl_runs_state_snapshot_gin_idx
    ON hitl_runs USING GIN (state_snapshot);

COMMIT;

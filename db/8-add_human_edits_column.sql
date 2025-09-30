-- Migration: add human_edits column to HITL runs
-- This migration adds persistent storage for human edits across HITL steps
-- implemented in `llm_backend/core/hitl/persistence.py`.

BEGIN;

-- Add human_edits column to hitl_runs table
ALTER TABLE hitl_runs
ADD COLUMN IF NOT EXISTS human_edits JSONB DEFAULT '{}'::jsonb;

-- Create index for human_edits queries
CREATE INDEX IF NOT EXISTS hitl_runs_human_edits_idx 
ON hitl_runs USING GIN (human_edits);

-- Add human_edits to state_snapshot indexes
CREATE INDEX IF NOT EXISTS hitl_runs_state_human_edits_idx 
ON hitl_runs ((state_snapshot->'human_edits'));

COMMIT;

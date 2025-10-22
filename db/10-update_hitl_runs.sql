-- Add missing HITL columns for checkpoint context and approval tracking
ALTER TABLE hitl_runs
ADD COLUMN IF NOT EXISTS checkpoint_context JSONB,
ADD COLUMN IF NOT EXISTS last_approval JSONB;

-- Ensure all existing runs have proper defaults for new columns
UPDATE hitl_runs 
SET 
    checkpoint_context = '{}',
    last_approval = '{}'
WHERE checkpoint_context IS NULL OR last_approval IS NULL;
-- Active: 1739390597810@@localhost@5432@tohju_db
-- Add session_id and user_id columns to hitl_runs table
ALTER TABLE hitl_runs 
ADD COLUMN IF NOT EXISTS session_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS user_id VARCHAR(255);

-- Add indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_hitl_runs_session_id ON hitl_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_hitl_runs_user_id ON hitl_runs(user_id);

-- Update existing records to extract session_id and user_id from original_input JSON
UPDATE hitl_runs 
SET 
    session_id = original_input->>'session_id',
    user_id = original_input->>'user_id'
WHERE session_id IS NULL OR user_id IS NULL;

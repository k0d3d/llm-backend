-- Fix the run_id type mismatch in hitl_pending_approvals table
-- The existing table has VARCHAR(36) but hitl_runs has UUID type

-- First, drop the existing table since it was created with wrong type
DROP TABLE IF EXISTS hitl_pending_approvals;

-- Recreate with correct UUID type for run_id
CREATE TABLE hitl_pending_approvals (
    id SERIAL PRIMARY KEY,
    approval_id VARCHAR(36) UNIQUE NOT NULL,
    run_id UUID NOT NULL,
    checkpoint_type VARCHAR(100) NOT NULL,
    context JSONB NOT NULL,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    status VARCHAR(50) DEFAULT 'pending' -- pending, responded, expired, cancelled
);

-- Create indexes for efficient querying
CREATE INDEX idx_hitl_pending_approvals_approval_id ON hitl_pending_approvals(approval_id);
CREATE INDEX idx_hitl_pending_approvals_run_id ON hitl_pending_approvals(run_id);
CREATE INDEX idx_hitl_pending_approvals_status ON hitl_pending_approvals(status);
CREATE INDEX idx_hitl_pending_approvals_expires_at ON hitl_pending_approvals(expires_at);
CREATE INDEX idx_hitl_pending_approvals_session_id ON hitl_pending_approvals(session_id);
CREATE INDEX idx_hitl_pending_approvals_user_id ON hitl_pending_approvals(user_id);

-- Add foreign key constraint to link with hitl_runs (now types match)
ALTER TABLE hitl_pending_approvals 
ADD CONSTRAINT fk_hitl_pending_approvals_run_id 
FOREIGN KEY (run_id) REFERENCES hitl_runs(run_id) ON DELETE CASCADE;

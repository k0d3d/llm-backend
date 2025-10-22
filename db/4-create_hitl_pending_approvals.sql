-- Create HITLPendingApproval table for database-backed approval persistence
-- This replaces the in-memory pending_approvals dictionary in WebSocketHITLBridge

CREATE TABLE IF NOT EXISTS hitl_pending_approvals (
    id SERIAL PRIMARY KEY,
    approval_id VARCHAR(36) UNIQUE NOT NULL,
    run_id VARCHAR(36) NOT NULL,
    checkpoint_type VARCHAR(100) NOT NULL,
    context JSONB NOT NULL,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    status VARCHAR(50) DEFAULT 'pending' -- pending, responded, expired, cancelled
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_hitl_pending_approvals_approval_id ON hitl_pending_approvals(approval_id);
CREATE INDEX IF NOT EXISTS idx_hitl_pending_approvals_run_id ON hitl_pending_approvals(run_id);
CREATE INDEX IF NOT EXISTS idx_hitl_pending_approvals_status ON hitl_pending_approvals(status);
CREATE INDEX IF NOT EXISTS idx_hitl_pending_approvals_expires_at ON hitl_pending_approvals(expires_at);
CREATE INDEX IF NOT EXISTS idx_hitl_pending_approvals_session_id ON hitl_pending_approvals(session_id);
CREATE INDEX IF NOT EXISTS idx_hitl_pending_approvals_user_id ON hitl_pending_approvals(user_id);

-- Add foreign key constraint to link with hitl_runs
DO $$
BEGIN
    ALTER TABLE hitl_pending_approvals 
    ADD CONSTRAINT fk_hitl_pending_approvals_run_id 
    FOREIGN KEY (run_id) REFERENCES hitl_runs(run_id) ON DELETE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN
        NULL; -- Constraint already exists
END $$;

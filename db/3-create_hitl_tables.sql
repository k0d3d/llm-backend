-- Create HITL tables if they don't exist (for fresh deployments)
-- This ensures all HITL tables exist with the complete schema

CREATE TABLE IF NOT EXISTS hitl_runs (
    run_id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    status VARCHAR(50) NOT NULL,
    current_step VARCHAR(50) NOT NULL,
    provider_name VARCHAR(100) NOT NULL,
    session_id VARCHAR(255),
    user_id VARCHAR(255),
    original_input JSONB NOT NULL,
    hitl_config JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    
    -- Step artifacts
    capabilities JSONB,
    suggested_payload JSONB,
    validation_issues JSONB,
    raw_response JSONB,
    processed_response TEXT,
    final_result TEXT,
    
    -- Human interactions
    pending_actions JSONB,
    approval_token VARCHAR(255),
    checkpoint_context JSONB,
    last_approval JSONB,
    
    -- Metrics
    total_execution_time_ms INTEGER DEFAULT 0,
    human_review_time_ms INTEGER DEFAULT 0,
    provider_execution_time_ms INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS hitl_step_events (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(36) NOT NULL,
    step VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    actor VARCHAR(255) NOT NULL,
    message TEXT,
    event_metadata JSONB
);

CREATE TABLE IF NOT EXISTS hitl_approvals (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(36) NOT NULL,
    approval_id VARCHAR(255) UNIQUE NOT NULL,
    checkpoint_type VARCHAR(100) NOT NULL,
    context JSONB NOT NULL,
    response JSONB,
    approved_by VARCHAR(255),
    approved_at TIMESTAMP,
    expires_at TIMESTAMP
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_hitl_runs_session_id ON hitl_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_hitl_runs_user_id ON hitl_runs(user_id);
CREATE INDEX IF NOT EXISTS idx_hitl_runs_status ON hitl_runs(status);
CREATE INDEX IF NOT EXISTS idx_hitl_step_events_run_id ON hitl_step_events(run_id);
CREATE INDEX IF NOT EXISTS idx_hitl_approvals_run_id ON hitl_approvals(run_id);
CREATE INDEX IF NOT EXISTS idx_hitl_approvals_approval_id ON hitl_approvals(approval_id);

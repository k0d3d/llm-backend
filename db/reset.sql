-- Database Reset Script for llm-backend HITL tables
-- WARNING: This will drop all HITL tables and data!
-- Use this for local development only

-- Drop all HITL tables (CASCADE removes dependent objects)
DROP TABLE IF EXISTS hitl_pending_approvals CASCADE;
DROP TABLE IF EXISTS hitl_approvals CASCADE;
DROP TABLE IF EXISTS hitl_step_events CASCADE;
DROP TABLE IF EXISTS hitl_runs CASCADE;

-- Recreate hitl_runs with complete schema (including all migrations)
CREATE TABLE hitl_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    checkpoint_context JSONB DEFAULT '{}',
    last_approval JSONB DEFAULT '{}',

    -- State management
    state_snapshot JSONB DEFAULT '{}',
    human_edits JSONB DEFAULT '{}',

    -- Metrics
    total_execution_time_ms INTEGER DEFAULT 0,
    human_review_time_ms INTEGER DEFAULT 0,
    provider_execution_time_ms INTEGER DEFAULT 0
);

-- Recreate hitl_step_events
CREATE TABLE hitl_step_events (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    step VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    actor VARCHAR(255) NOT NULL,
    message TEXT,
    event_metadata JSONB,
    FOREIGN KEY (run_id) REFERENCES hitl_runs(run_id) ON DELETE CASCADE
);

-- Recreate hitl_approvals
CREATE TABLE hitl_approvals (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    approval_id VARCHAR(255) UNIQUE NOT NULL,
    checkpoint_type VARCHAR(100) NOT NULL,
    context JSONB NOT NULL,
    response JSONB,
    approved_by VARCHAR(255),
    approved_at TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES hitl_runs(run_id) ON DELETE CASCADE
);

-- Recreate hitl_pending_approvals
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
    status VARCHAR(50) DEFAULT 'pending',
    FOREIGN KEY (run_id) REFERENCES hitl_runs(run_id) ON DELETE CASCADE
);

-- Create all indexes
-- hitl_runs indexes
CREATE INDEX idx_hitl_runs_session_id ON hitl_runs(session_id);
CREATE INDEX idx_hitl_runs_user_id ON hitl_runs(user_id);
CREATE INDEX idx_hitl_runs_status ON hitl_runs(status);
CREATE INDEX hitl_runs_state_status_idx ON hitl_runs ((state_snapshot->>'status'));
CREATE INDEX hitl_runs_checkpoint_type_idx ON hitl_runs ((state_snapshot->'checkpoint_context'->>'type'));
CREATE INDEX hitl_runs_pending_actions_idx ON hitl_runs USING GIN ((state_snapshot->'pending_actions'));
CREATE INDEX hitl_runs_validation_issues_idx ON hitl_runs USING GIN ((state_snapshot->'validation_issues'));
CREATE INDEX hitl_runs_state_snapshot_gin_idx ON hitl_runs USING GIN (state_snapshot);
CREATE INDEX hitl_runs_human_edits_idx ON hitl_runs USING GIN (human_edits);
CREATE INDEX hitl_runs_state_human_edits_idx ON hitl_runs ((state_snapshot->'human_edits'));

-- hitl_step_events indexes
CREATE INDEX idx_hitl_step_events_run_id ON hitl_step_events(run_id);

-- hitl_approvals indexes
CREATE INDEX idx_hitl_approvals_run_id ON hitl_approvals(run_id);
CREATE INDEX idx_hitl_approvals_approval_id ON hitl_approvals(approval_id);

-- hitl_pending_approvals indexes
CREATE INDEX idx_hitl_pending_approvals_approval_id ON hitl_pending_approvals(approval_id);
CREATE INDEX idx_hitl_pending_approvals_run_id ON hitl_pending_approvals(run_id);
CREATE INDEX idx_hitl_pending_approvals_status ON hitl_pending_approvals(status);
CREATE INDEX idx_hitl_pending_approvals_expires_at ON hitl_pending_approvals(expires_at);
CREATE INDEX idx_hitl_pending_approvals_session_id ON hitl_pending_approvals(session_id);
CREATE INDEX idx_hitl_pending_approvals_user_id ON hitl_pending_approvals(user_id);

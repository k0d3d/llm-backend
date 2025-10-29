-- Fix run_id type mismatch from VARCHAR(36) to UUID
-- This resolves the error: operator does not exist: character varying = uuid

-- Update hitl_runs table
ALTER TABLE hitl_runs
ALTER COLUMN run_id TYPE UUID USING run_id::uuid;

-- Update hitl_step_events table (foreign key reference)
ALTER TABLE hitl_step_events
ALTER COLUMN run_id TYPE UUID USING run_id::uuid;

-- Update hitl_approvals table (foreign key reference)
ALTER TABLE hitl_approvals
ALTER COLUMN run_id TYPE UUID USING run_id::uuid;

-- Fix the default value generation
ALTER TABLE hitl_runs
ALTER COLUMN run_id SET DEFAULT gen_random_uuid();

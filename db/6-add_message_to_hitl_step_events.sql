-- Add message and event_metadata columns to hitl_step_events
ALTER TABLE hitl_step_events
    ADD COLUMN IF NOT EXISTS message TEXT,
    ADD COLUMN IF NOT EXISTS event_metadata JSONB;

# Human-in-the-Loop (HITL) Integration Overview

## Current Architecture

* __Backend Core__ (`llm_backend/core/hitl/`)
  * `HITLOrchestrator` drives the end-to-end workflow with checkpoints for information review, payload review, API call, and response review.
  * `WebSocketHITLBridge` coordinates approval requests and responses via WebSocket, persisting pending approvals using the shared state manager.
  * `persistence.py` provides state storage with PostgreSQL and optional Redis caching. Tables involved:
    * `hitl_runs` – canonical run state, including `pending_actions`, `checkpoint_context`, `last_approval`.
    * `hitl_step_events` – audit log of state transitions.
    * `hitl_pending_approvals` – active approval tickets keyed by UUID.
* __API Layer__ (`llm_backend/api/endpoints/`)
  * `hitl.py` exposes `/api/hitl/run` for new runs, approval endpoints, and status queries.
  * `teams.py` backs the legacy `/api/teams/run?enable_hitl=true` flow but now reuses the shared WebSocket bridge & state manager.
* __Frontend__ (`pwa/src/`)
  * WebSocket hook (`useRagWebSocket.ts`) listens for `HITL_APPROVAL_REQUEST` messages and injects them into thread state.
  * `HITLValidationCard.tsx` renders validation details and provides approve / reject actions.
  * File uploads (`useFileUpload.ts`) clear state on unmount; attachment replay still required for HITL payloads.

## Recent Fixes & Improvements

* __Shared bridge instance__ – `shared_bridge.py` ensures all endpoints use the same `WebSocketHITLBridge` and state manager.
* __Database-backed approvals__ – Pending approvals now persist in `hitl_pending_approvals`, enabling cross-instance continuity.
* __Schema alignment__ – Converted `run_id` columns to UUID types and normalized JSON serialization to avoid serialization errors.
* __State persistence__ – `DatabaseStateStore.save_state()` now serializes enums, datetimes, and nested structures safely, and stores step events with UUID FK integrity.

## Known Gaps / Next Steps

1. **Attachment replay** – `_attachments_from_chat_history()` is a stub; implement retrieval of uploaded assets (e.g., via session-based message store) so resumed runs use real files instead of example inputs.
   - ✅ **Partially addressed**: Schema-aware attachment fallback now ensures user attachments are mapped to correct fields even when AI agents fail. See `docs/FORM_BASED_HITL.md` for details.
2. **Observability** – Surface `checkpoint_context`, `pending_actions`, and approval metadata in session-status APIs and structure logs for pause/resume events.
3. **Frontend UX** – Ensure the client re-fetches active HITL runs after reconnect and resubmits any required edits or files alongside approval responses.
4. **Testing** – Add integration tests around pause/resume cycles and database persistence (e.g., `tests/test_hitl_session_resumability.py`).

## Immediate Plan

* __Short term__
  * Instrument session endpoints to return the stored `checkpoint_context` so the UI can rehydrate paused runs.
  * Implement attachment discovery to propagate human-provided files into provider payloads.
* __Medium term__
  * Build metrics/logging dashboard for HITL pause frequency, approval latency, and failure modes.
  * Extend UI to show historical approvals and edits per session.
* __Long term__
  * Support multi-agent HITL scenarios and richer approval workflows (e.g., staged approvals, comment threads).

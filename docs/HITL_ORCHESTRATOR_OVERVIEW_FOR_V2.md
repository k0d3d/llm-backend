# HITL Orchestrator: Comprehensive System Overview (V2 Preparation)

This document provides a deep-dive into the Human-in-the-Loop (HITL) Orchestration system developed for the Tohju LLM Backend. It is intended for developers to understand the current architecture, state management, and agentic workflows to facilitate the design of Version 2.

---

## 1. Core Mission
The Orchestrator acts as a "Stateful Pipeline" that bridges high-level user intents (Natural Language) with low-level AI model requirements (Replicate API payloads). It ensures reliability by inserting human checkpoints whenever the AI is uncertain or lacks critical information.

## 2. Architectural Topology

### A. Component Hierarchy
1.  **`HITLOrchestrator`**: The "brain." It manages the transition between steps and maintains the execution state.
2.  **`AIProvider` (e.g., `ReplicateProvider`)**: The "interface." It knows the specific schema and execution mechanics of the target model.
3.  **`ReplicateTeam`**: The "intelligent engine." A Pydantic AI-driven multi-agent system that handles payload generation, validation, and attachment discovery.
4.  **`HybridStateManager`**: The "memory." Persists the execution state across Web and Worker boundaries using PostgreSQL (Persistence) and Redis (Speed/Task Queue).
5.  **`WebSocketHITLBridge`**: The "voice." Synchronizes state with the frontend UI for real-time human interaction.

### B. Execution Context (Web vs. Worker)
*   **Web API**: Receives the initial request, initializes the `run_id`, and enqueues the job into **RQ (Redis Queue)**.
*   **Worker**: Pops the job, recreates the orchestrator instance from the DB state, and executes the pipeline steps. 
*   *Critical Rule*: Worker processes must be fork-safe. We use lazy initialization for database engines and bridges to prevent stale file descriptors.

---

## 3. The Execution Pipeline

The orchestrator executes a linear sequence of steps, but can "pause" and "resume" at any point.

### Step 1: `FORM_INITIALIZATION`
*   **Action**: Analyzes the model's `example_input`.
*   **Logic**: Uses an AI classifier to distinguish between **CONTENT** (must be reset for user) and **CONFIG** (keep defaults).
*   **Result**: A structured `form_data` object in the state.

### Step 2: `INFORMATION_REVIEW` (The Dialogue Step)
*   **Action**: Compares current values against required fields.
*   **NL Mode**: If `use_natural_language_hitl` is enabled, it uses a `NLPromptGenerator` to ask the user for missing info via chat instead of a rigid form.
*   **Checkpoint**: Pauses execution if information is missing.

### Step 3: `PAYLOAD_REVIEW`
*   **Action**: Generates the final JSON payload for the Replicate API.
*   **Logic**: The `ReplicateTeam` agent merges:
    *   Original prompt
    *   User-submitted form fields
    *   Discovered attachments (URLs)
    *   Historical conversation context
*   **Authority**: In current V1.5, we strictly ignore `example_input` prompts to prevent "leakage" into production calls.

### Step 4: `API_CALL`
*   **Action**: Executes the model on Replicate.
*   **Recovery**: If the API returns a 422 (Validation Error), the orchestrator triggers an **Error Recovery Agent** to explain the fix to the user conversationally.

### Step 5: `RESPONSE_REVIEW` & `COMPLETED`
*   **Action**: Audits the output (extracts URLs, strips echoed prompts) and terminates the run.

---

## 4. Key Implementation Patterns

### Async Event Loop Safety
The orchestrator is designed to run in both FastAPI (async) and RQ Workers (sync-wrapped). 
*   **Pattern**: We use a robust `_add_step_event` handler that checks if a loop is running. If not, it uses `asyncio.run()`; if yes, it uses `loop.create_task()`.

### Attachment Discovery
A major feature of the orchestrator is "Intelligent Harvesting."
*   **Sync Harvest**: Scrapes URLs from the current prompt and `agent_tool_config`.
*   **Async Triage**: Uses a Pydantic AI agent to look back through the last 10 chat messages to find relevant images or files the user might be referring to (e.g., "Use that photo I sent earlier").

### State Persistence (`HITLState`)
The state is stored as a JSONB document in PostgreSQL (`state_snapshot`). This allows us to:
*   Resume a run on a different worker.
*   Reconstruct the exact agent memory.
*   Query analytics (e.g., "Which step fails most often?").

---

## 5. Lessons Learned for V2 Improvements

### 1. The "Deadlock" Prevention
*   **Problem**: Calling `asyncio.run()` or `ThreadPoolExecutor` inside an already running event loop caused freezes in production.
*   **V2 Requirement**: Ensure all Agent methods are natively async and awaited top-to-bottom. Avoid `run_until_complete` in production paths.

### 2. Payload Authority
*   **Problem**: Merging user input with `example_input` led to the model receiving "demo" prompts.
*   **V2 Requirement**: The payload generation agent should have a "Zero-Trust" policy towards example data. Use it for schema shape only, not for values.

### 3. Tenant & Context Integrity
*   **Problem**: Missing `tenant` or `operation_type` properties caused Zod validation errors in the Core API.
*   **V2 Requirement**: Standardize the `HITLMessageClient` to always include full context (`tenant`, `log_id`, `prompt`) in every outgoing message.

### 4. Concurrent Session Handling
*   **Problem**: A user sending two prompts quickly can create race conditions for the "Active Run."
*   **V2 Requirement**: Implement "Run Locking" or a "Queue-per-Session" logic to ensure messages are processed in order.

### 5. Multi-Agent Coordination
*   **Problem**: Single-agent payload creation is sometimes too complex for one prompt.
*   **V2 Proposal**: Use a "Planner-Executor" pattern where one agent identifies requirements and another builds the payload.

---

## 6. Development Tips
*   **Debugging**: Set `LOG_LEVEL=DEBUG`. Look for `🤖 Agent created payload` in worker logs to see what the AI is thinking.
*   **Testing**: Use `tests/test_hitl_flow.py` to simulate a full multi-step resume cycle.
*   **Database**: Check the `hitl_step_events` table to see the exact history of a failed run.

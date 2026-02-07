# HITL Orchestrator V3: Architecture Plan

## 1. Executive Summary

This document proposes a comprehensive overhaul of the Human-in-the-Loop (HITL) Orchestrator for the Tohju LLM Backend. The current system (V2) suffers from:
1.  **Data Leakage**: Demo values from `example_input` frequently contaminate production payloads.
2.  **Strict/Brittle Triggering**: Logic for pausing/resuming is spread across multiple layers, leading to false positives or missed checkpoints.
3.  **Hallucinations**: Agents struggle to distinguish between "user intent" and "example schema values".
4.  **State Fragmentation**: State is split between `HITLState`, `ReplicateTeam`, and the Worker execution context.

**The Goal**: A "Zero-Trust" Schema-Driven Architecture where the orchestration pipeline explicitly separates *Schema Definition* from *Payload Construction*.

---

## 2. Core Philosophy: The Assembly Line

We move from a "Manager with Helpers" model (Orchestrator + Team) to a strict "Assembly Line" pipeline.

**Phases:**
1.  **Blueprint Phase (Schema Extraction)**: Analyze the model's capabilities and `example_input` to create a strict `ModelSchema` definition. *No values are preserved here, only types and descriptions.*
2.  **Context Assembly Phase**: Gather all inputs (Prompt, Chat History, Attachments, User Profile) into a unified `RequestBundle`.
3.  **Fabrication Phase (Payload Builder)**: An AI Agent takes the `RequestBundle` + `ModelSchema` and produces a `CandidatePayload`. *Crucially, it never sees the original `example_input` values.*
4.  **Quality Control Phase (Validation)**: A strict validator checks `CandidatePayload` against `ModelSchema`.
    *   **Pass**: Send to API.
    *   **Fail (Ambiguous)**: Trigger HITL (Ask specific question).
    *   **Fail (Critical)**: Trigger HITL (Error Recovery).

---

## 3. Architecture Components

### A. The Schema Extractor (The Blueprint)
*   **Role**: Converts the messy `example_input` JSON into a clean Pydantic model or JSON Schema definition.
*   **Behavior**:
    *   Input: `{"prompt": "a photo of a cat", "width": 1024}`
    *   Output: `{"prompt": {"type": "string", "required": true}, "width": {"type": "integer", "default": 1024}}`
*   **Why**: By stripping the values immediately, we mathematically prevent "a photo of a cat" from ever leaking into a payload unless the user explicitly types it.

### B. The Context Assembler (The Bundle)
*   **Role**: Creates the single source of truth for "What the user wants".
*   **Behavior**: Aggregates:
    *   `run_input.prompt` (The raw text)
    *   `run_input.conversation` (The history)
    *   `run_input.attachments` (Explicit files)
    *   `state.human_edits` (Overrides from previous HITL steps)
*   **Why**: Prevents agents from needing to "look back" at raw tool configs or disparate state objects.

### C. The Payload Builder Agent (The Fabricator)
*   **Role**: Maps the `RequestBundle` to the `ModelSchema`.
*   **System Prompt Strategy**:
    *   "You are a translator. Convert this User Request into a JSON object matching this Schema."
    *   "The Schema provided defines KEYS and TYPES. It does NOT contain content."
    *   "If the user did not specify a value for 'width', use the Schema's default or leave it null."
*   **Dependencies**: `pydantic_ai` Agent.

### D. The Strict Validator (The Guard)
*   **Role**: Deterministic code (not AI). Checks if `CandidatePayload` satisfies `ModelSchema.required_fields`.
*   **Output**: `ValidationResult(valid=False, issues=[MissingField(name='image')])`.

---

## 4. The New Orchestrator Flow (State Machine)

The `HITLOrchestratorV3` will manage this loop:

```python
async def execute_run(self):
    # 1. Blueprint
    schema = SchemaExtractor.extract(self.tool_config)
    
    # 2. Context
    context = ContextAssembler.build(self.run_input, self.state)
    
    # 3. Fabrication
    payload = await PayloadBuilder.build(context, schema)
    
    # 4. Validation
    issues = Validator.validate(payload, schema)
    
    if issues:
        # 5. Interaction (HITL)
        if self.should_ask_human(issues):
            return self.trigger_checkpoint(issues)
        else:
            # Auto-fix or Fail
            pass
            
    # 6. Execution
    result = await self.provider.execute(payload)
    return result
```

---

## 5. Migration Strategy

To ensure stability while refactoring, we will implement V3 alongside V2.

1.  **Create `src/llm_backend/core/hitl/v3/`**: New directory.
2.  **Implement `SchemaExtractor`**: This is a pure utility function.
3.  **Implement `PayloadBuilder`**: A new Agent that uses the Schema.
4.  **Shadow Mode**: In the V2 Orchestrator, optionally run the V3 pipeline in the background and log the difference in payloads.
5.  **Cutover**: Switch the main entry point to use `HITLOrchestratorV3`.

## 6. Detailed Implementation Steps

### Step 1: Schema Extraction Logic
Refactor `form_field_classifier.py` logic. Instead of just "classifying", it should return a strictly typed definition object that separates `default_value` (safe config) from `example_value` (unsafe content).

### Step 2: Payload Builder Agent
Create a new `pydantic_ai` agent that accepts `SchemaDefinition` as a dependency, NOT `ExampleInput`.

### Step 3: Validation Logic
Move validation logic out of `ReplicateTeam` and into a standalone `Validator` class that operates on dictionaries/Pydantic models, returning a list of standardized `ValidationIssue` objects.

### Step 4: The V3 Orchestrator
Build a simplified Orchestrator that uses the above components. It should be much smaller than the current one ( < 500 lines).

## 8. Progress Update (Implementations)

As of the latest update, the core V3 components have been implemented in `src/llm_backend/core/hitl/v3/`:

1.  **`schema_extractor.py`**: Successfully parses `example_input` into a strict `ModelSchema`, explicitly separating values from structure.
2.  **`context_assembler.py`**: Aggregates prompts, history, and attachments into a unified `RequestContext`.
3.  **`payload_builder.py`**: A Pydantic AI agent that uses the *Schema* and *Context* to build payloads without seeing the original example values (Zero-Trust).
4.  **`validator.py`**: Deterministic validation logic.

### Next Steps for Integration
1.  **Create `HITLOrchestratorV3`**: Create the new orchestrator class that wires these components together.
2.  **Update Endpoint**: Modify `src/llm_backend/api/endpoints/hitl.py` to use `HITLOrchestratorV3` (either via a feature flag or strict replacement).
3.  **Testing**: Verify that the "example input leakage" bug is resolved by running a test case where the example input has a unique string that must NOT appear in the final payload.

## 7. Immediate Fixes for V2 (Interim)

While building V3, we can patch V2 to stop the bleeding:
1.  **Patch `ReplicateTeam`**: Modify it to accept `form_data` from the Orchestrator state and prioritize those values over `example_input`.
2.  **Patch `HITLOrchestrator`**: Ensure it passes `form_data.current_values` explicitly to `ReplicateTeam` or `provider.create_payload`.

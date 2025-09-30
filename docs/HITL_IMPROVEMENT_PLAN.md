





# Planning the AI-Agent Enhancements

## [ReplicateTeam](cci:2://file:///home/k0d3d/Projects/ixit-stuff/tohju/llm-backend/src/llm_backend/agents/replicate_team.py:12:0-304:43) Overview
[ReplicateTeam](cci:2://file:///home/k0d3d/Projects/ixit-stuff/tohju/llm-backend/src/llm_backend/agents/replicate_team.py:12:0-304:43) orchestrates a multi-agent pipeline: an information-gathering agent, a request-building agent (`replicate_agent()`), an API interaction agent, and a response auditor. Each agent already has access to tools (functions) and rich contextual prompts. This setup is ideal for modular validation agents inserted before the actual Replicate API call.

## Proposed Agent Layering

### 1. File Requirement Detection Agent
* __Purpose__  
  Mirrors [_requires_file_input()](cci:1://file:///home/k0d3d/Projects/ixit-stuff/tohju/llm-backend/src/llm_backend/core/hitl/validation.py:247:4-251:78) / [_check_model_compatibility()](cci:1://file:///home/k0d3d/Projects/ixit-stuff/tohju/llm-backend/src/llm_backend/core/hitl/validation.py:322:4-349:21) logic to declare whether image/audio artifacts are mandatory based on:
  * `tool_config.example_input` schema fields (e.g. `input_image`, `image`, `audio`, `file`).
  * Model metadata (`tool_config.model_name`, `tool_config.description`).
  * Run context (current prompt, user attachments, HITL edits).

* __Inputs__  
  * `tool_config` from [ReplicateTeam](cci:2://file:///home/k0d3d/Projects/ixit-stuff/tohju/llm-backend/src/llm_backend/agents/replicate_team.py:12:0-304:43).
  * `run_input.agent_tool_config["replicate-agent-tool"]["data"]` payload.
  * List of attachments already discovered or supplied via HITL.

* __Outputs__  
  * Requirement decision: `required_files` array with type tags (`image`, `audio`, etc.).
  * Blocking issue summary for the HITL layer if the required asset is missing.
  * Auto-resolve hints (e.g. “use last uploaded image”, “prompt user to upload”).

* __Tools__  
  * File schema analyzer tool: inspects `example_input` keys & value types.
  * Attachment resolver: queries HITL state or recent uploads.
  * Model taxonomy helper: fuzzy match model name/description against known patterns.

* __Placement__  
  * Runs immediately after [information_agent()](cci:1://file:///home/k0d3d/Projects/ixit-stuff/tohju/llm-backend/src/llm_backend/agents/replicate_team.py:94:4-143:32) returns but before `replicate_agent()` builds the payload.  
  * Feeds results into both HITL validation and subsequent agents.

### 2. Payload Validation Agent
* __Purpose__  
  Replaces [_validate_parameters()](cci:1://file:///home/k0d3d/Projects/ixit-stuff/tohju/llm-backend/src/llm_backend/core/hitl/validation.py:95:4-138:9) with an AI-driven validator ensuring the payload we’re about to send contains all required fields and matches declared types.

* __Inputs__  
  * Preliminary payload from the existing `replicate_agent()` (or earlier draft payload).
  * Required field list from File Requirement Agent.
  * HITL edits (`state.human_edits`) merged into context.

* __Checks__  
  * Required fields present and non-empty (prompt, image URLs, etc.).
  * Parameter coherence (no conflicting options, valid enums).
  * Inject helpful remediation suggestions for missing data, returning them to the HITL UI.

* __Outputs__  
  * Structured report: `blocking_issues`, `warnings`, `auto_fixable` flags.
  * Updated payload if straightforward fixes are possible (e.g., apply HITL-provided asset).
  * Guidance back to orchestrator (proceed, pause, or request human input).

* __Tools__  
  * Schema validator: cross-checks payload vs. `example_input` (type and presence).
  * HITL edit merger: maps `state.human_edits` fields into payload.
  * Prompt quality analyzer (optional reuse of existing heuristics).

* __Placement__  
  * Executes after `replicate_agent()` constructs the candidate payload but before API interaction.  
  * Its verdict drives whether we continue automatically or re-enter HITL.

### 3. Final Guard Agent Against `example_input`
* __Purpose__  
  Acts as the last checkpoint before calling Replicate, ensuring the payload precisely aligns with `example_input` schema and defaults.

* __Inputs__  
  * Final payload ready for submission.
  * `tool_config.example_input` and derived schema annotations.
  * Any “must match” invariants (field types, accepted value ranges).

* __Responsibilities__  
  * Strict schema enforcement: every required key from `example_input` must be present with acceptable value type/format.
  * Detect extraneous or malformed fields.
  * Confirm human edits are applied (e.g., if `input_image` now points to the HITL-uploaded asset).
  * Produce an auditable record (e.g., JSON diff between `example_input` base and final payload) for logging or UI display.

* __Outputs__  
  * `approved_payload` (identical to input if all good) or structured error requiring HITL intervention.
  * Summary for `response_audit_agent()` and persistence.

* __Tools__  
  * Diff generator & schema comparator.
  * Optional call back into [replicate_provider.create_payload()](cci:1://file:///home/k0d3d/Projects/ixit-stuff/tohju/llm-backend/src/llm_backend/providers/replicate_provider.py:49:4-162:9) fallback mapping to ensure consistency.

* __Placement__  
  * Directly wraps the call inside [api_interaction_agent()](cci:1://file:///home/k0d3d/Projects/ixit-stuff/tohju/llm-backend/src/llm_backend/agents/replicate_team.py:43:4-91:36).  
  * If it flags an issue, execution pauses; otherwise the API request proceeds.

## Workflow Integration

1. [information_agent()](cci:1://file:///home/k0d3d/Projects/ixit-stuff/tohju/llm-backend/src/llm_backend/agents/replicate_team.py:94:4-143:32) -> gather context, produce initial instructions.
2. **File Requirement Agent** -> declare required assets and update HITL context.
3. `replicate_agent()` -> build preliminary payload using example input + prompt + available assets.
4. **Payload Validation Agent** -> validate the payload, apply HITL edits, prompt for missing items if needed.
5. **Final Guard Agent** -> compare final payload with `example_input` to ensure schema fidelity.
6. [api_interaction_agent()](cci:1://file:///home/k0d3d/Projects/ixit-stuff/tohju/llm-backend/src/llm_backend/agents/replicate_team.py:43:4-91:36) -> only runs if all agents approve.

Each agent emits structured outputs consumable by HITL orchestration (`validation_summary`, `blocking_issues`, suggestions), ensuring consistent UX in the frontend ([HITLValidationCard.tsx](cci:7://file:///home/k0d3d/Projects/ixit-stuff/tohju/pwa/src/features/session/components/HITLValidationCard.tsx:0:0-0:0)).

## Next Steps for Implementation

* Define agent prompts and tool functions within [ReplicateTeam](cci:2://file:///home/k0d3d/Projects/ixit-stuff/tohju/llm-backend/src/llm_backend/agents/replicate_team.py:12:0-304:43).
* Extend `run_input` / state models to include discovered attachments and HITL edits for agent consumption.
* Update HITL orchestrator to interpret new agent outputs, pausing runs when blocking issues arise.
* Ensure the final guard’s audit record is stored/surfaced for debugging.

Let me know how deep you’d like to go on each agent’s prompt design or tool interfaces.
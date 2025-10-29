# HITL System Improvement Plan

**Date**: 2025-10-29
**Status**: Planned
**Branch**: To be created before implementation

---

## Problem Analysis

From production logs analysis, we identified several critical issues affecting the HITL (Human-in-the-Loop) system:

### Issue 1: Unnecessary Human Intervention
The system paused for approval when it shouldn't have:
- **Observation**: `blocking_issues: 0` but still `needs_review? True`
- **Context**: User request "Turn this to a cartoon" with an image attachment is complete
- **Impact**: All validation checkpoints passed (0 issues each) but system still paused
- **Result**: 48+ second execution time when it should auto-approve

### Issue 2: AI Agent Failures
Multiple OpenAI retry attempts followed by failures:
- `⚠️ Attachment resolver agent failed: Exceeded maximum retries (1) for result validation`
- **Pattern**: 7+ OpenAI API calls per request with repeated retries
- **Logs showing**:
  ```
  2025-10-29 00:06:33,897 [INFO] Retrying request to /chat/completions in 0.497866 seconds
  2025-10-29 00:06:57,372 [INFO] Retrying request to /chat/completions in 0.427733 seconds
  ```
- **Root cause**: pydantic-ai agents configured with max_retries=1 (too low)
- **Model**: Using `gpt-4.1-mini` (confirmed valid model from OpenAI pricing)

### Issue 3: Empty Payload Creation
After all the AI processing:
- `🤖 Agent created payload: {}`
- Validation error: `ERROR: input - Uploaded file not found in payload`
- **Problem**: Attachments were gathered (`['https://replicate.delivery/pbxt/...']`) but not properly mapped to payload
- **Impact**: Complete workflow failure after consuming API quota

### Issue 4: Unclear Control Flow
Multiple overlapping systems creating confusion:
- Form-based workflow vs intelligent agent workflow
- Legacy validation vs new validation system
- Manual fallbacks that don't actually fix the issue
- No clear decision tree for which path to take

---

## Root Causes

### 1. Over-Engineering
- **Too many AI agents**:
  - `field_analyzer.py` - Analyzes which fields are replaceable
  - `attachment_mapper.py` - Maps attachments to fields
  - `attachment_resolver.py` - Resolves attachment conflicts
  - `form_field_classifier.py` - Classifies form fields
  - `replicate_team.py` - Orchestrates payload creation
- **Issue**: Each agent can fail independently, creating cascading failures
- **Conflict**: Agents' retry logic conflicts with pydantic-ai's own retry mechanism

### 2. Unclear Responsibilities
- Orchestrator doesn't know when to use form data vs intelligent agent
- Provider's `create_payload` has dual paths but both can fail
- Validation happens at multiple levels (pre-execution, payload review, form validation)
- No single source of truth for "is this request complete?"

### 3. Policy Logic Issues
- `AUTO_WITH_THRESHOLDS` policy pauses even when `blocking_issues == 0`
- No clear rules for "when should we skip human approval entirely?"
- Threshold checking happens but doesn't respect zero blocking issues

### 4. Technical Debt
- Tests are completely outdated (using removed APIs like `start_run`, `execute_run`)
- Retry limits set too low (max 1 retry) causing unnecessary failures
- No integration tests for complete auto-approval flow

---

## Proposed Solution: Simplified HITL Architecture

### Core Principles
1. **Single Source of Truth**: One validation system, one payload creation path
2. **Fail Fast**: If AI agents fail, use deterministic fallbacks immediately
3. **Smart Auto-Approval**: Skip human review when inputs are complete and valid
4. **Minimal AI**: Only use AI where it adds clear value (reduce from 5 agents to 1)

### New Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       HITL Orchestrator                          │
│                                                                  │
│  1. INPUT VALIDATION (deterministic)                             │
│     ├─ Parse schema from example_input                          │
│     ├─ Check required fields against run_input                  │
│     ├─ Auto-discover attachments from chat history              │
│     └─ Decision: blocking_issues count                           │
│                                                                  │
│  2. AUTO-APPROVE DECISION (policy-based)                         │
│     ├─ IF blocking_issues == 0 AND policy == AUTO               │
│     │  └─ SKIP to step 3 (no human intervention)                │
│     ├─ IF blocking_issues == 0 AND policy == AUTO_WITH_THRESHOLDS│
│     │  └─ SKIP to step 3 (no human intervention)                │
│     └─ ELSE → PAUSE for human approval                          │
│                                                                  │
│  3. PAYLOAD CREATION (simple deterministic)                      │
│     ├─ Map run_input.prompt → schema.prompt field               │
│     ├─ Map attachments[0] → schema.input_image/input_audio      │
│     ├─ Use example_input defaults for remaining fields          │
│     ├─ Apply human edits if any (from approval response)        │
│     └─ Validate payload has all required fields                 │
│                                                                  │
│  4. EXECUTION                                                    │
│     ├─ Call provider.execute(payload)                           │
│     └─ Return response                                          │
└──────────────────────────────────────────────────────────────────┘
```

**Key Simplification**: Remove 4 out of 5 AI agents, keep only essential form classifier.

---

## Implementation Plan

### Phase 1: Quick Fixes (1-2 hours)
**Goal**: Make system usable for basic cases without full rewrite

**Changes**:
1. **Fix Auto-Approval Logic** (`orchestrator.py`)
   - Update `_step_information_review()`:
     ```python
     # NEW: Skip pause if no blocking issues
     if blocking_issues == 0 and policy in [HITLPolicy.AUTO, HITLPolicy.AUTO_WITH_THRESHOLDS]:
         print("✅ Auto-approving: no blocking issues")
         return {"continue": True}
     ```

2. **Increase Retry Limits** (all agent files)
   - Change `max_retries=1` to `max_retries=3` in pydantic-ai agent configs
   - Files: `attachment_resolver.py`, `field_analyzer.py`, `attachment_mapper.py`

3. **Add Deterministic Payload Fallback** (`replicate_provider.py`)
   - When AI agent returns empty payload, use simple field mapping:
     ```python
     if not agent_payload or not agent_payload.get("input"):
         # Fallback: simple deterministic mapping
         payload_input = {
             "prompt": prompt,
             **example_input  # Use example defaults
         }
         # Map first attachment to input_image if field exists
         if attachments and "input_image" in example_input:
             payload_input["input_image"] = attachments[0]
     ```

4. **Improve Logging**
   - Add clear decision logging for why we pause vs auto-approve
   - Log fallback usage when AI agents fail

**Expected Impact**:
- Basic requests auto-approve correctly
- Failures have deterministic fallback
- Better visibility into decision making

---

### Phase 2: Structural Refactoring (3-4 hours)
**Goal**: Simplify architecture by removing redundant AI agents

**Changes**:

1. **Remove Redundant AI Agents**
   - **DELETE**: `src/llm_backend/agents/attachment_resolver.py`
   - **DELETE**: `src/llm_backend/agents/field_analyzer.py`
   - **DELETE**: `src/llm_backend/agents/attachment_mapper.py`
   - **KEEP**: `src/llm_backend/agents/form_field_classifier.py` (for complex forms only)
   - **SIMPLIFY**: `src/llm_backend/agents/replicate_team.py` (remove AI-based attachment discovery)

2. **Implement Schema-Based Payload Creation** (`replicate_provider.py`)
   - New method: `_create_payload_deterministic()`
   - Algorithm:
     ```python
     def _create_payload_deterministic(self, prompt, attachments, hitl_edits):
         payload = {}

         # 1. Map prompt to prompt field
         for field in ["prompt", "text", "instruction"]:
             if field in self.example_input:
                 payload[field] = prompt
                 break

         # 2. Map attachments to media fields
         if attachments:
             for field in ["input_image", "image", "source_image"]:
                 if field in self.example_input:
                     payload[field] = attachments[0]
                     break

         # 3. Copy defaults from example_input for remaining fields
         for field, value in self.example_input.items():
             if field not in payload:
                 payload[field] = value

         # 4. Apply human edits (override defaults)
         if hitl_edits:
             payload.update(hitl_edits)

         return payload
     ```

3. **Consolidate Validation Checkpoints** (`validation.py`)
   - Remove separate checkpoint types (merge into single pre-check)
   - Simplified rules:
     - Missing required field = ERROR (blocking)
     - Missing optional field = WARNING (non-blocking)
     - Field type mismatch = WARNING (non-blocking)

4. **Update Orchestrator Flow** (`orchestrator.py`)
   - Remove form initialization step for simple requests
   - Only use form workflow when:
     - User explicitly requests it, OR
     - Schema has >5 configurable fields, OR
     - Model requires complex configuration

**Expected Impact**:
- 80% reduction in OpenAI API calls
- 5-10x faster execution (5-10s vs 48s)
- More predictable behavior
- Easier debugging

---

### Phase 3: Testing & Polish (1-2 hours)
**Goal**: Ensure reliability and maintainability

**Changes**:

1. **Rewrite Tests** (`tests/test_hitl_orchestrator.py`)
   - Match current API (provider, config, run_input)
   - Test cases:
     ```python
     def test_auto_approve_with_complete_input()
     def test_pause_with_missing_required_field()
     def test_deterministic_payload_creation()
     def test_attachment_mapping()
     def test_human_edits_applied()
     ```

2. **Add Integration Test** (`tests/test_hitl_integration.py`)
   - Test complete flow from run_input to provider execution
   - Mock external dependencies (WebSocket, Redis, PostgreSQL)
   - Verify auto-approval decisions

3. **Documentation Updates**
   - Update `docs/HITL_ORCHESTRATOR.md` with new simplified flow
   - Add decision tree diagram for auto-approval logic
   - Document deterministic payload creation algorithm

4. **Performance Monitoring**
   - Add timing metrics for each step
   - Log OpenAI API call count per request
   - Track auto-approval vs manual approval ratios

**Expected Impact**:
- Confidence in system behavior
- Easier onboarding for new developers
- Performance baseline for future improvements

---

## Files to Modify

### Core Logic Changes
1. **`src/llm_backend/core/hitl/orchestrator.py`**
   - Auto-approval logic improvements
   - Remove AI agent dependency calls
   - Simplify step pipeline

2. **`src/llm_backend/providers/replicate_provider.py`**
   - Add `_create_payload_deterministic()` method
   - Remove dependency on attachment resolver agent
   - Implement simple field mapping

3. **`src/llm_backend/core/hitl/validation.py`**
   - Consolidate checkpoint types
   - Simplify blocking issue detection
   - Remove AI-based field analysis

### Agent Cleanup
4. **`src/llm_backend/agents/attachment_resolver.py`** - DELETE
5. **`src/llm_backend/agents/field_analyzer.py`** - DELETE
6. **`src/llm_backend/agents/attachment_mapper.py`** - DELETE
7. **`src/llm_backend/agents/replicate_team.py`** - SIMPLIFY (remove AI discovery)
8. **`src/llm_backend/agents/form_field_classifier.py`** - KEEP (increase retry limit)

### Testing
9. **`tests/test_hitl_orchestrator.py`** - REWRITE completely
10. **`tests/test_hitl_integration.py`** - NEW FILE

### Documentation
11. **`docs/HITL_ORCHESTRATOR.md`** - UPDATE with new flow
12. **`docs/ARCHITECTURE.md`** - UPDATE architecture diagram

---

## Expected Outcomes

### Before (Current State)
```
OpenAI API Calls: 7+ per request
Runtime: ~48 seconds
Auto-Approval: Broken (pauses when blocking_issues=0)
Payload Creation: Fails with empty payload
User Experience: Frustrating (unnecessary wait times)
Cost: High (excessive API usage)
```

### After (Target State)
```
OpenAI API Calls: 0-1 per request (only for complex forms)
Runtime: ~5-10 seconds
Auto-Approval: Works correctly (skips when blocking_issues=0)
Payload Creation: Deterministic fallback always works
User Experience: Fast and seamless for simple requests
Cost: 85% reduction in API usage
```

---

## Risk Assessment

### Low Risk
- Phase 1 changes (quick fixes) are additive and have fallbacks
- Deterministic payload creation is simpler and more reliable
- Tests will catch regressions before deployment

### Medium Risk
- Removing AI agents might reduce "intelligence" for edge cases
- **Mitigation**: Keep fallback logging to identify cases that need AI
- **Mitigation**: Monitor success/failure rates before and after

### High Risk
- None identified. Changes are incremental and reversible.

---

## Rollout Strategy

1. **Create Feature Branch**
   ```bash
   git checkout -b hitl-system-simplification
   ```

2. **Implement Phase 1** (quick fixes)
   - Deploy to staging
   - Monitor for 24 hours
   - Check metrics: auto-approval rate, API call count, error rate

3. **If Phase 1 Successful**: Implement Phase 2 (refactoring)
   - Deploy to staging
   - Run load tests
   - Monitor for 48 hours

4. **If Phase 2 Successful**: Implement Phase 3 (testing & docs)
   - Final review
   - Deploy to production with gradual rollout
   - Monitor closely for first week

5. **Rollback Plan**: Keep old branch for 2 weeks in case issues arise

---

## Success Metrics

Track these metrics before and after:

1. **Performance**
   - Average request duration
   - P95 request duration
   - OpenAI API calls per request

2. **Reliability**
   - Success rate (completed vs failed requests)
   - Auto-approval accuracy (should approve vs did approve)
   - Empty payload occurrence rate

3. **User Experience**
   - Time to first response
   - Manual approval frequency
   - User satisfaction (if tracked)

4. **Cost**
   - OpenAI API costs per 1000 requests
   - Server CPU/memory usage

---

## Next Steps

1. **Review this plan** with team
2. **Create feature branch** before any code changes
3. **Start with Phase 1** implementation
4. **Monitor metrics** at each phase
5. **Iterate based on results**

---

**Document Status**: Ready for review and implementation
**Last Updated**: 2025-10-29
**Author**: Claude Code Analysis

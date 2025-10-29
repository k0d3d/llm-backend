#!/bin/bash
# Quick test script for Natural Language HITL using curl

API_URL="http://localhost:8811/api/teams/run?enable_hitl=true"

echo "======================================================================"
echo "Test 1: Complete Request (Should Auto-Skip)"
echo "======================================================================"
echo ""
echo "Sending request with ALL required fields..."
echo "Expected: Should auto-skip (no pause), ~5-10 seconds"
echo ""

curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A photo of a dog playing in a park",
    "user_email": "test@example.com",
    "user_id": "test-user-123",
    "agent_email": "agent@example.com",
    "session_id": "test-session-auto-skip",
    "message_type": "user_message",
    "agent_tool_config": {
      "replicate-agent-tool": {
        "data": {
          "name": "flux-dev",
          "model_name": "flux-dev",
          "description": "Text-to-image generation model",
          "example_input": {
            "prompt": "",
            "aspect_ratio": "16:9",
            "num_outputs": 1
          },
          "latest_version": "test-version"
        }
      }
    }
  }' | jq '.'

echo ""
echo ""
echo "======================================================================"
echo "Test 2: Missing Prompt (Should Trigger NL Conversation)"
echo "======================================================================"
echo ""
echo "Sending request WITHOUT prompt..."
echo "Expected: Should pause with natural language message"
echo ""

curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "",
    "user_email": "test@example.com",
    "user_id": "test-user-123",
    "agent_email": "agent@example.com",
    "session_id": "test-session-nl-mode",
    "message_type": "user_message",
    "agent_tool_config": {
      "replicate-agent-tool": {
        "data": {
          "name": "flux-dev",
          "model_name": "flux-dev",
          "description": "Text-to-image generation model",
          "example_input": {
            "prompt": "",
            "aspect_ratio": "16:9",
            "num_outputs": 1
          },
          "latest_version": "test-version"
        }
      }
    }
  }' | jq '.'

echo ""
echo ""
echo "======================================================================"
echo "Check Docker logs for:"
echo "  - 💬 Information Review: Natural language conversation mode"
echo "  - ✅ All required fields satisfied - auto-skipping"
echo "  - ⏸️ PAUSING for natural language input"
echo ""
echo "Run: docker compose logs -f web worker | grep -E '(💬|✅|⏸️|DEBUG)'"
echo "======================================================================"

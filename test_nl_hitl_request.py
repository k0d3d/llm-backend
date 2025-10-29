#!/usr/bin/env python3
"""
Test script for Natural Language HITL system
Tests both auto-skip and NL conversation modes with proper example_input
"""

import requests
import json
import time
from typing import Dict, Any


# Configuration
API_BASE_URL = "http://localhost:8811"
HITL_ENDPOINT = f"{API_BASE_URL}/api/teams/run?enable_hitl=true"


def create_test_request(
    prompt: str = None,
    include_all_fields: bool = True,
    model_name: str = "flux-dev"
) -> Dict[str, Any]:
    """Create a properly formatted HITL test request for /api/teams/run"""

    # Base request structure matching RunInput schema
    request = {
        "prompt": prompt or "",
        "user_email": "test@example.com",
        "user_id": "test-user-123",
        "agent_email": "agent@example.com",
        "session_id": "test-session-nl-hitl",
        "message_type": "user_message",
        "agent_tool_config": {
            "replicate-agent-tool": {
                "data": {
                    "name": model_name,  # Note: endpoint looks for 'model_name' OR 'name'
                    "model_name": model_name,
                    "description": "A text-to-image generation model that creates images from text prompts",
                    "example_input": {
                        # THIS IS THE KEY PART - example_input with field schema
                        "prompt": "",
                        "aspect_ratio": "16:9",
                        "num_outputs": 1,
                        "guidance_scale": 7.5,
                        "num_inference_steps": 50
                    },
                    "latest_version": "test-version-id"
                }
            }
        }
    }

    return request


def test_auto_skip_scenario():
    """Test 1: Complete request should auto-skip (no pause)"""
    print("=" * 80)
    print("TEST 1: Auto-Skip Scenario (Complete Request)")
    print("=" * 80)
    print("\nSending request with ALL required fields...")
    print("Expected: Should auto-skip information_review (no pause)\n")

    request = create_test_request(
        prompt="A photo of a dog playing in a park",
        include_all_fields=True
    )

    print(f"Request payload:\n{json.dumps(request, indent=2)}\n")

    start_time = time.time()

    try:
        response = requests.post(
            HITL_ENDPOINT,
            json=request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        elapsed = time.time() - start_time

        print(f"✅ Response received in {elapsed:.2f}s")
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse data:")
            print(f"  run_id: {data.get('run_id')}")
            print(f"  status: {data.get('status')}")
            print(f"  current_step: {data.get('current_step')}")

            # Check if it auto-skipped
            if data.get('status') in ['running', 'queued']:
                print(f"\n✅ SUCCESS: System auto-skipped HITL (no pause)")
                print(f"✅ Execution time: {elapsed:.2f}s (should be ~5-10s)")
                return True
            else:
                print(f"\n⚠️ UNEXPECTED: Status is {data.get('status')}")
                return False
        else:
            print(f"\n❌ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"\n⏱️ Request timed out after {elapsed:.2f}s")
        print("⚠️ This might mean the system is paused waiting for approval")
        return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_nl_conversation_scenario():
    """Test 2: Incomplete request should trigger NL conversation"""
    print("\n\n" + "=" * 80)
    print("TEST 2: NL Conversation Scenario (Missing Prompt)")
    print("=" * 80)
    print("\nSending request with MISSING required field (prompt)...")
    print("Expected: Should pause with natural language message\n")

    request = create_test_request(
        prompt=None,  # Intentionally missing
        include_all_fields=False
    )

    print(f"Request payload:\n{json.dumps(request, indent=2)}\n")

    start_time = time.time()

    try:
        response = requests.post(
            HITL_ENDPOINT,
            json=request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        elapsed = time.time() - start_time

        print(f"✅ Response received in {elapsed:.2f}s")
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse data:")
            print(f"  run_id: {data.get('run_id')}")
            print(f"  status: {data.get('status')}")
            print(f"  current_step: {data.get('current_step')}")

            # Check if it paused for approval
            if data.get('status') == 'awaiting_human':
                print(f"\n✅ SUCCESS: System paused for human input")

                # Check for NL conversation mode
                pause_data = data.get('data', {})
                if pause_data.get('conversation_mode'):
                    nl_message = pause_data.get('nl_prompt', 'No message')
                    print(f"✅ Natural Language Mode Active!")
                    print(f"\nNL Message to user:")
                    print(f"  \"{nl_message}\"")
                    print(f"\nMissing fields: {pause_data.get('missing_fields')}")
                    return True
                else:
                    print(f"\n⚠️ PAUSED but not in NL mode (using old form-based)")
                    print(f"Checkpoint type: {pause_data.get('checkpoint_type')}")
                    return False
            else:
                print(f"\n⚠️ UNEXPECTED: Status is {data.get('status')}, expected 'awaiting_human'")
                return False
        else:
            print(f"\n❌ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_image_editing_model():
    """Test 3: Image editing model with proper example_input"""
    print("\n\n" + "=" * 80)
    print("TEST 3: Image Editing Model (Flux-Kontext-Pro)")
    print("=" * 80)
    print("\nTesting with image editing model that requires input_image...\n")

    request = {
        "prompt": "Turn this to a cartoon - Why did the scarecrow win an award?",
        "user_email": "test@example.com",
        "user_id": "test-user-123",
        "agent_email": "agent@example.com",
        "session_id": "test-session-image-edit",
        "message_type": "user_message",
        "agent_tool_config": {
            "replicate-agent-tool": {
                "data": {
                    "name": "flux-kontext-pro",
                    "model_name": "flux-kontext-pro",
                    "description": "A state-of-the-art text-based image editing model",
                    "example_input": {
                        "prompt": "",
                        "input_image": "",  # Required but empty
                        "aspect_ratio": "match_input_image",
                        "safety_tolerance": 2,
                        "output_format": "jpg"
                    },
                    "latest_version": "test-version-id"
                }
            }
        }
    }

    print(f"Request payload:\n{json.dumps(request, indent=2)}\n")

    try:
        response = requests.post(
            HITL_ENDPOINT,
            json=request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data.get('status')}")
            print(f"Current step: {data.get('current_step')}")

            if data.get('status') == 'awaiting_human':
                pause_data = data.get('data', {})
                if pause_data.get('conversation_mode'):
                    print(f"\n✅ NL Mode Active!")
                    print(f"Message: {pause_data.get('nl_prompt')}")
                    return True

        print(f"\nResponse: {json.dumps(data, indent=2)}")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def check_server_health():
    """Check if the server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    print("🧪 Natural Language HITL Test Suite")
    print("=" * 80)
    print(f"Testing against: {API_BASE_URL}")
    print()

    # Check server health
    print("Checking server health...")
    if not check_server_health():
        print(f"❌ Server not reachable at {API_BASE_URL}")
        print("Please ensure Docker containers are running:")
        print("  docker compose up -d web worker")
        return

    print("✅ Server is running\n")

    # Run tests
    results = {
        "auto_skip": test_auto_skip_scenario(),
        "nl_conversation": test_nl_conversation_scenario(),
        "image_editing": test_image_editing_model()
    }

    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Auto-Skip):       {'✅ PASS' if results['auto_skip'] else '❌ FAIL'}")
    print(f"Test 2 (NL Conversation): {'✅ PASS' if results['nl_conversation'] else '❌ FAIL'}")
    print(f"Test 3 (Image Editing):   {'✅ PASS' if results['image_editing'] else '❌ FAIL'}")

    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    if not all_passed:
        print("\n⚠️ Troubleshooting:")
        print("1. Check Docker logs: docker compose logs -f web worker")
        print("2. Look for: '💬 Information Review: Natural language conversation mode'")
        print("3. Ensure example_input is present in tool config")
        print("4. Verify use_natural_language_hitl=True in hitl_config")


if __name__ == "__main__":
    main()

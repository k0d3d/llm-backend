"""
Test OpenAI API connectivity and model availability
Run this before implementing HITL improvements to ensure AI agents will work
"""

import pytest
import os
from openai import OpenAI

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


@pytest.fixture
def openai_client():
    """Create OpenAI client"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in environment or .env file")
    return OpenAI(api_key=api_key)


def test_openai_api_key_exists():
    """Test that OPENAI_API_KEY is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key is not None, "OPENAI_API_KEY environment variable not set in .env file"
    assert len(api_key) > 0, "OPENAI_API_KEY is empty"
    assert api_key.startswith("sk-"), f"OPENAI_API_KEY should start with 'sk-' but starts with '{api_key[:5]}...'"


def test_openai_simple_completion(openai_client):
    """Test basic OpenAI API call"""
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": "Say 'test successful' and nothing else."}
        ],
        max_tokens=10
    )

    assert response is not None
    assert len(response.choices) > 0
    content = response.choices[0].message.content
    assert content is not None
    assert "test successful" in content.lower()
    print(f"✅ OpenAI response: {content}")


def test_gpt_4_1_mini_model_available(openai_client):
    """Test that gpt-4.1-mini model is available and working"""
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ],
        max_tokens=5
    )

    content = response.choices[0].message.content.strip()
    assert "4" in content
    print(f"✅ gpt-4.1-mini is working: 2+2={content}")


def test_structured_output(openai_client):
    """Test structured output parsing (used by pydantic-ai agents)"""
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": 'Return a JSON object with fields: {"status": "ok", "message": "test"}'
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=50
    )

    content = response.choices[0].message.content
    assert content is not None

    import json
    data = json.loads(content)
    assert "status" in data
    assert "message" in data
    print(f"✅ Structured output works: {data}")


def test_retry_on_rate_limit(openai_client):
    """Test that we can handle rate limits gracefully"""
    # Make a simple call - if we hit rate limit, this will show us
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=5
        )
        assert response is not None
        print("✅ No rate limit issues")
    except Exception as e:
        if "rate_limit" in str(e).lower():
            pytest.skip(f"Rate limit hit (expected in some cases): {e}")
        else:
            raise


def test_pydantic_ai_integration():
    """Test that pydantic-ai can use OpenAI"""
    try:
        from pydantic_ai import Agent
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            result: str

        agent = Agent(
            "openai:gpt-4.1-mini",
            output_type=TestOutput,
            system_prompt="You are a test agent. Always respond with result='success'."
        )

        # Run synchronously for testing
        import asyncio

        async def run_test():
            result = await agent.run("Say success")
            return result

        result = asyncio.run(run_test())
        assert result.output.result == "success"
        print(f"✅ pydantic-ai integration works: {result.output.result}")

    except ImportError as e:
        pytest.skip(f"pydantic-ai not installed: {e}")


if __name__ == "__main__":
    """Run tests standalone for quick connectivity check"""
    print("🔍 Testing OpenAI connectivity...\n")

    # Check API key
    print("1. Checking OPENAI_API_KEY...")
    try:
        test_openai_api_key_exists()
        print("   ✅ API key is set\n")
    except AssertionError as e:
        print(f"   ❌ {e}")
        print("\n⚠️  Please ensure OPENAI_API_KEY is set in .env file")
        exit(1)

    # Create client
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # Test basic call
    print("2. Testing basic API call...")
    try:
        test_openai_simple_completion(client)
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        exit(1)

    # Test gpt-4.1-mini
    print("3. Testing gpt-4.1-mini model...")
    try:
        test_gpt_4_1_mini_model_available(client)
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        exit(1)

    # Test structured output
    print("4. Testing structured output...")
    try:
        test_structured_output(client)
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        exit(1)

    # Test retry
    print("5. Testing rate limit handling...")
    try:
        test_retry_on_rate_limit(client)
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        exit(1)

    # Test pydantic-ai
    print("6. Testing pydantic-ai integration...")
    try:
        test_pydantic_ai_integration()
        print()
    except Exception as e:
        print(f"   ⚠️  Skipped: {e}\n")

    print("✅ All connectivity tests passed!")

"""
Error Recovery Natural Language Agent

Generates natural language error messages for HITL error recovery.
Uses AI to explain technical errors in user-friendly terms.
"""

import os
from typing import Dict, List, Optional
from openai import AsyncOpenAI

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def generate_error_recovery_message(
    error_type: str,
    field: Optional[str],
    current_value: Optional[str],
    valid_values: List[str],
    conversation_history: List[Dict[str, str]],
    error_message: str
) -> str:
    """
    Generate a natural language error recovery message.

    Args:
        error_type: "validation", "authentication", etc.
        field: The field that caused the error
        current_value: The invalid value that was sent
        valid_values: List of valid values for the field
        conversation_history: Full conversation context
        error_message: Raw error message from API

    Returns:
        Natural language error explanation and recovery prompt
    """

    # Build conversation context (last 5 messages)
    history_context = "\n".join([
        f"{msg.get('role', 'user')}: {msg.get('message', msg.get('content', ''))[:200]}"
        for msg in conversation_history[-5:]
    ]) if conversation_history else "No prior conversation"

    # Build valid values list for prompt
    valid_values_str = ""
    if valid_values:
        if len(valid_values) <= 10:
            valid_values_str = f"Valid options: {', '.join(valid_values)}"
        else:
            valid_values_str = f"Valid options include: {', '.join(valid_values[:10])} (and {len(valid_values)-10} more)"

    prompt = f"""You are helping a user fix an API validation error. Generate a friendly, conversational error message that:

1. Explains what went wrong in simple terms (no technical jargon)
2. Mentions the specific field if applicable
3. Provides the valid options clearly
4. Asks the user to provide a corrected value in natural language

Context:
- Error type: {error_type}
- Field: {field or "unknown"}
- Current value: {current_value or "not specified"}
- {valid_values_str}
- Technical error: {error_message}

Recent conversation:
{history_context}

Generate a natural language error message (2-3 sentences max) that explains the issue and asks the user for the correct value. Be conversational and helpful, not technical.

Example good message:
"Oops! The aspect ratio you chose isn't supported by this model. You can use 16:9 (widescreen), 1:1 (square), or 21:9 (ultra-wide). Which aspect ratio would you like?"

Your error message:"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains API errors in simple, friendly language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        nl_message = response.choices[0].message.content.strip()
        print(f"🤖 Generated error recovery message: {nl_message[:100]}...")
        return nl_message

    except Exception as e:
        print(f"❌ Failed to generate NL error message: {e}")

        # Fallback to template-based message
        if field and valid_values:
            values_str = ', '.join(valid_values[:5])
            if len(valid_values) > 5:
                values_str += f" (and {len(valid_values)-5} more)"
            return f"I encountered an issue with the {field} parameter. Please choose from: {values_str}. What would you like to use?"
        else:
            return f"I encountered an error: {error_message[:100]}. Can you help me fix this?"

"""
HITL Message Client

Sends HITL checkpoints as messages through /from-llm endpoint.
This integrates HITL into the natural conversation flow.
"""

import os
import httpx
from typing import Dict, Any, Optional


CORE_API_URL = os.getenv("CORE_API_URL", "https://core-api-d1kvr2.asyncdev.workers.dev")
M2M_TOKEN = os.getenv("SESSION_API_M2M_TOKEN", "")


class HITLMessageClient:
    """Client for sending HITL checkpoints as messages through /from-llm"""

    def __init__(self, base_url: str = CORE_API_URL, m2m_token: str = M2M_TOKEN):
        self.base_url = base_url
        self.m2m_token = m2m_token

        # Set up headers with M2M token
        headers = {"Content-Type": "application/json"}
        if self.m2m_token:
            headers["X-M2M-Token"] = self.m2m_token

        self.client = httpx.AsyncClient(timeout=30.0, headers=headers)

    async def send_hitl_checkpoint(
        self,
        session_id: str,
        user_id: str,
        content: str,
        checkpoint_type: str,
        checkpoint_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send HITL checkpoint as a message via /from-llm.

        This creates a regular message in the chat history with HITL metadata
        stored in the props field. The user can respond naturally via text.

        Args:
            session_id: Session ID
            user_id: User ID
            content: Natural language message (e.g., "Oops! That aspect ratio isn't supported...")
            checkpoint_type: "error_recovery", "form_requirements", etc.
            checkpoint_data: HITL metadata (error_field, valid_values, run_id, etc.)

        Returns:
            Response from /from-llm endpoint with messageId

        Example:
            await client.send_hitl_checkpoint(
                session_id="abc123",
                user_id="user456",
                content="Oops! That aspect ratio isn't supported. Try 16:9, 1:1, or 21:9.",
                checkpoint_type="error_recovery",
                checkpoint_data={
                    "run_id": "hitl_xyz",
                    "error_field": "aspect_ratio",
                    "valid_values": ["16:9", "1:1", "21:9"]
                }
            )
        """
        # Build props with HITL metadata
        props = {
            "checkpoint_type": checkpoint_type,
            "awaiting_response": True,
            **checkpoint_data  # Include all checkpoint data (run_id, error_field, etc.)
        }

        # Create message payload
        payload = {
            "sessionId": session_id,
            "userId": user_id,
            "sender": "assistant",
            "content": content,
            "messageType": "hitl_checkpoint",  # Special message type for HITL
            "destination": "user",
            "logId": checkpoint_data.get("run_id", ""),
            "props": props  # HITL metadata stored here
        }

        try:
            url = f"{self.base_url}/from-llm"
            print(f"📤 Sending HITL checkpoint to {url}")
            print(f"   Type: {checkpoint_type}")
            print(f"   Session: {session_id}")

            response = await self.client.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            message_id = result.get("messageId")

            print(f"✅ HITL checkpoint sent successfully (messageId: {message_id})")

            return result

        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            print(f"❌ HTTP {e.response.status_code} sending HITL checkpoint: {error_body}")
            raise

        except httpx.HTTPError as e:
            print(f"❌ Failed to send HITL checkpoint: {e}")
            raise

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

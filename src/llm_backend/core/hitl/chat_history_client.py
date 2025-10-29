"""
Chat History Client

Fetches chat history from Core API (CF Workers) for context in HITL workflows.
"""

import os
import httpx
from typing import List, Dict, Optional


CORE_API_URL = os.getenv("CORE_API_URL", "https://core-api-d1kvr2.asyncdev.workers.dev")
M2M_TOKEN = os.getenv("SESSION_API_M2M_TOKEN", "")


class ChatHistoryClient:
    """Client for fetching chat history from Core API"""

    def __init__(self, base_url: str = CORE_API_URL, m2m_token: str = M2M_TOKEN):
        self.base_url = base_url
        self.m2m_token = m2m_token

        # Set up headers with M2M token
        headers = {}
        if self.m2m_token:
            headers["X-M2M-Token"] = self.m2m_token

        self.client = httpx.AsyncClient(timeout=10.0, headers=headers)

    async def get_session_history(
        self,
        session_id: str,
        include_system: bool = True,
        include_props: bool = True
    ) -> List[Dict[str, str]]:
        """
        Fetch chat history for a session.

        Args:
            session_id: The session ID to fetch history for
            include_system: Whether to include system messages
            include_props: Whether to include props field (for HITL context detection)

        Returns:
            List of messages in format:
            [
                {
                    "role": "user",
                    "content": "...",
                    "timestamp": "...",
                    "props": {...},  # Only if include_props=True
                    "message_type": "..."  # Only if include_props=True
                }
            ]
        """
        try:
            url = f"{self.base_url}/m2m/messages/session/{session_id}"
            print(f"📡 Fetching chat history from M2M endpoint: {url}")
            response = await self.client.get(url)
            response.raise_for_status()

            # Endpoint returns array directly, not wrapped in {messages: [...]}
            messages = response.json()
            if not isinstance(messages, list):
                messages = []

            # Convert to standardized format
            formatted_messages = []
            for msg in messages:
                role = self._normalize_role(msg.get("sender", "user"))

                # Skip system messages if requested
                if not include_system and role == "system":
                    continue

                message_data = {
                    "role": role,
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("created_at", "")
                }

                # Include props and message_type for HITL checkpoint detection
                if include_props:
                    if msg.get("props"):
                        message_data["props"] = msg.get("props")
                    if msg.get("message_type"):
                        message_data["message_type"] = msg.get("message_type")

                formatted_messages.append(message_data)

            return formatted_messages

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print(f"ℹ️ No chat history found for session {session_id}")
                return []
            print(f"⚠️ Failed to fetch chat history (HTTP {e.response.status_code}): {e}")
            return []
        except httpx.HTTPError as e:
            print(f"⚠️ Failed to fetch chat history: {e}")
            return []

    def _normalize_role(self, sender: str) -> str:
        """Normalize sender to standard roles"""
        sender_lower = sender.lower()
        if "user" in sender_lower:
            return "user"
        elif "assistant" in sender_lower or "agent" in sender_lower:
            return "assistant"
        else:
            return "system"

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

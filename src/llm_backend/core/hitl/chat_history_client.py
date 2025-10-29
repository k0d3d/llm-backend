"""
Chat History Client

Fetches chat history from Core API (CF Workers) for context in HITL workflows.
"""

import os
import httpx
from typing import List, Dict, Optional


CORE_API_URL = os.getenv("CORE_API_URL", "https://core-api-d1kvr2.asyncdev.workers.dev")


class ChatHistoryClient:
    """Client for fetching chat history from Core API"""

    def __init__(self, base_url: str = CORE_API_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=10.0)

    async def get_session_history(
        self,
        session_id: str,
        include_system: bool = True
    ) -> List[Dict[str, str]]:
        """
        Fetch chat history for a session.

        Args:
            session_id: The session ID to fetch history for
            include_system: Whether to include system messages

        Returns:
            List of messages in format:
            [
                {"role": "user", "content": "...", "timestamp": "..."},
                {"role": "assistant", "content": "...", "timestamp": "..."}
            ]
        """
        try:
            url = f"{self.base_url}/api/sessions/{session_id}/messages"
            response = await self.client.get(url)
            response.raise_for_status()

            data = response.json()
            messages = data.get("messages", [])

            # Convert to standardized format
            formatted_messages = []
            for msg in messages:
                role = self._normalize_role(msg.get("sender", "user"))

                # Skip system messages if requested
                if not include_system and role == "system":
                    continue

                formatted_messages.append({
                    "role": role,
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("created_at", "")
                })

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

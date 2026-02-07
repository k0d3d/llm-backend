import json
import requests
import httpx

from llm_backend.core.types.common import RunInput, MessageType, T_MessageType


def send_data_to_url(data: dict | str, url: str, crew_input: RunInput, message_type: T_MessageType = MessageType["AGENT_MESSAGE"]):
    """
    Send data to a server using a stream.
    """
    if crew_input is None:
        print(f"⚠️ send_data_to_url: crew_input is None, skipping request to {url}")
        return None

    # Build the correct user field
    payload_data = {
        "sessionId": getattr(crew_input, 'session_id', None),
        "content": data,
        "destination": getattr(crew_input, 'user_email', None),
        "userId": getattr(crew_input, 'user_id', None),
        "sender": getattr(crew_input, 'agent_email', None),
        "messageType": message_type,
        "logId": getattr(crew_input, 'log_id', None),
        "prompt": getattr(crew_input, 'prompt', None),
        "tenant": getattr(crew_input, 'tenant', 'tohju'),
        "operationType": data.get("operation_type", "") if isinstance(data, dict) else "text",
    }

    # Convert data to JSON format as it's typically expected for a POST request's body
    json_data = json.dumps(payload_data)

    try:
        # Send the POST request to the server
        return requests.post(
            url,
            data=json_data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        print(f"Error: An error occurred while requesting {url!r}.")
        print(f"Details: {e}")
        return None


async def send_data_to_url_async(data: dict | str, url: str, crew_input: RunInput, message_type: T_MessageType = MessageType["AGENT_MESSAGE"]):
    """
    Async variant of send_data_to_url using httpx.AsyncClient to avoid blocking the event loop.
    """

    payload_data = {
        "sessionId": crew_input.session_id,
        "content": data,
        "destination": crew_input.user_email,
        "userId": crew_input.user_id,
        "sender": crew_input.agent_email,
        "messageType": message_type,
        "logId": crew_input.log_id,
        "prompt": crew_input.prompt,
        "tenant": crew_input.tenant,
        "operationType": data.get("operation_type", "") if isinstance(data, dict) else "text",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            return await client.post(url, json=payload_data)
    except httpx.HTTPError as e:
        print(f"Error: An error occurred while requesting {url!r} (async).")
        print(f"Details: {e}")
        return None

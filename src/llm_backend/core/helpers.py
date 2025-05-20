import json

import requests

from llm_backend.core.types.common import RunInput, MessageType, T_MessageType



async def send_data_to_api(data: str, url: str, crew_input: RunInput, message_type: T_MessageType = MessageType["AGENT_MESSAGE"]):
    """
    Send data to a server using a stream.

    Args:
    - data_generator: A generator that yields data to send.
    - url: The URL of the server to send the data to.

    Returns:
    - response: The response object returned by the server.
    """

    # Build the correct user field
    # Construct the data payload with the corrected user field structure
    payload_data = {
        "sessionId": crew_input.session_id,
        "content": data,
        "userEmail": crew_input.user_email,
        "userId": crew_input.user_id,
        "agentEmail": crew_input.agent_email,
        "messageType": message_type,
        "logId": crew_input.log_id,
    }

    # Convert data to JSON format as it's typically expected for a POST request's body
    json_data = json.dumps(payload_data)

    # Send the POST request to the server
    return requests.post(
        url,
        data=json_data,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )

def send_data_to_url(data: str, url: str, crew_input: RunInput, message_type: T_MessageType = MessageType["AGENT_MESSAGE"]):
    """
    Send data to a server using a stream.

    Args:
    - data_generator: A generator that yields data to send.
    - url: The URL of the server to send the data to.

    Returns:
    - response: The response object returned by the server.
    """

    # Build the correct user field
    # Construct the data payload with the corrected user field structure
    payload_data = {
        "sessionId": crew_input.session_id,
        "content": data,
        "userEmail": crew_input.user_email,
        "userId": crew_input.user_id,
        "agentEmail": crew_input.agent_email,
        "messageType": message_type,
        "logId": crew_input.log_id,
        "prompt": crew_input.prompt
    }

    # Convert data to JSON format as it's typically expected for a POST request's body
    json_data = json.dumps(payload_data)

    # Send the POST request to the server
    return requests.post(
        url,
        data=json_data,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )

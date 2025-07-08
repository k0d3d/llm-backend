import os
import json
from pydantic_ai import ModelRetry
import requests
from typing import Any, Type
from pydantic import BaseModel, Field

from llm_backend.core.helpers import send_data_to_url
from llm_backend.core.types.common import RunInput, MessageType
from llm_backend.core.types.replicate import PayloadInput



REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
TOHJU_NODE_API = os.getenv("TOHJU_NODE_API", "https://api.tohju.com")
CORE_API_URL = os.getenv("CORE_API_URL", "https://core-api-d1kvr2.asyncdev.workers.dev")


def run_replicate(
    run_input: RunInput,
    model_params: dict,
    input: PayloadInput,
    operation_type: str,
    ):
        # Define the URL for the POST request
        url = "https://api.replicate.com/v1/predictions"

        # Define the headers, typically you'd need authorization and content type headers
        headers = {
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",  # Replace with your actual API key
            "Content-Type": "application/json",
        }

        # check if input is json string and attempt to convert it to object
        if isinstance(input, PayloadInput):
            try:
                # Attempt to load the JSON string into a Python object
                input = input.model_dump()
                # input = json.loads(input.replace("'", '"'))
            except json.JSONDecodeError as e:
                # Handle the case where the input string is not valid JSON
                input = {}

        example_input = model_params.get("example_input", {})
        body_input = {**example_input, **(input if input is not None else {}) }
        # return response_object

        # The body of the request
        body = {
            "input": body_input,
            "version": model_params.get("latest_version"),
            "webhook": f"{TOHJU_NODE_API}/api/webhooks/onReplicateComplete",
        }

        response = requests.post(url=url, headers=headers, json=body)

        # Check if the request was successful
        if response.status_code == 201 or response.status_code == 200:
            prediction = response.json()
            message_type = MessageType["REPLICATE_PREDICTION"]
            send_data_to_url(
                data={
                    "prediction": prediction,
                    "operation_type": operation_type,
                },
                url=f"{TOHJU_NODE_API}/api/webhooks/onReplicateStarted",
                crew_input=run_input,
                message_type=message_type,
            )

        return response.json(), response.status_code

class DictToClass(BaseModel, extra='allow'):
    """Input can be any schema ."""
    # def __init__(self, dictionary):
    #     # Iterate over each key-value pair in the dictionary
    #     for key, value in dictionary.items():
    #         # Set each dictionary key as an attribute of the class instance with its corresponding value
    #         setattr(self, key, value)

def convert_dict_to_class(dictionary):
    return DictToClass(**dictionary)



class InputPromptSchema(DictToClass, extra="allow"):
    """Input can be any schema ."""
    input: str = Field(
        ..., description="Input based on the rewritten example_input. Do not make up properties that are not in example_input. MUST be valid JSON"
    )

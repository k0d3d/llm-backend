

from enum import Enum
from pydantic import BaseModel
from typing import Optional, Union

class RunInput(BaseModel):
    prompt: str
    user_email: str
    user_id: str
    agent_email: str
    session_id: str
    message_type: str
    log_id: Optional[str] = None
    agent_tool_config: Optional[dict]
    selected_llm: Optional[dict] = None


class AgentTools(str, Enum):
    RSSTOOL = "rss-feed-agent-tool"
    WEB3TOOL = "web3-agent-tool"
    WOOTOOL = "woo-agent-tool"
    REPLICATETOOL = "replicate-agent-tool"


MessageType = {
    # user sends a message
    "USER_MESSAGE": "user_message",
    # agent sends a message or reply
    "AGENT_MESSAGE": "agent_message",
    # user replies to an agent message
    "USER_REPLY": "user_reply",
    # agent replies to a user task
    "BASIC_TASK_REPLY": "basic_task_reply",
    "TURBO_TASK_REPLY": "turbo_task_reply",
    # agent describes a basic task from a user message
    "BASIC_TASK_SPEC": "basic_task_spec",
    # agent describes a turbo task from a user message
    "TURBO_TASK_SPEC": "turbo_task_spec",
    # describes a task from a user message
    "TASK_SPEC": "task_spec",
    "REPLICATE_PREDICTION": "replicate_prediction",
}

T_MessageType = Union[str, type(MessageType)]

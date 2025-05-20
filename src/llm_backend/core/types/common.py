

from enum import Enum
from pydantic import BaseModel
from typing import List, Optional

class RunInput(BaseModel):
    prompt: str
    user_email: str
    user_id: str
    session_id: str
    message_type: str
    log_id: int = None
    document_url: Optional[str] = None
    agent_tool_config: Optional[dict]
    selected_llm: Optional[dict] = None


class AgentTools(str, Enum):
    RSSTOOL = "rss-feed-agent-tool"
    WEB3TOOL = "web3-agent-tool"
    WOOTOOL = "woo-agent-tool"
    REPLICATETOOL = "replicate-agent-tool"
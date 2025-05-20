

from pydantic import BaseModel
from typing import List, Optional

class RunInput(BaseModel):
    prompt: str
    user_email: str
    user_id: str
    chat_history: List[dict]
    session_id: str
    task_name: str = None
    agent_role: str = "SessionAgent"
    agent_email: str
    agent_type: str = "session_agent"
    agent_id: str
    agent_goal: str
    agent_backstory: str
    task_description: str = None
    task_expected_output: str = None
    message_type: str
    log_id: int = None
    document_context: Optional[str] = ""
    document_url: Optional[str] = None
    agent_tool_data: List[dict]
    matching_agents: List[dict]
    agent_tool_prompt: str = None
    agent_tool_config: Optional[dict]
    selected_llm: Optional[dict] = None
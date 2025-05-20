

from fastapi import APIRouter

from llm_backend.agents.replicate_agent import ReplicateTeam
from llm_backend.core.types.common import AgentTools, RunInput


router = APIRouter()


@router.post("/run")
def run_replicate_team(run_input: RunInput):
  agent_tool_config = run_input.agent_tool_config
  replicate_agent_tool_config = agent_tool_config.get(AgentTools.REPLICATETOOL)

  replicate_team = ReplicateTeam(
    prompt=run_input.prompt,
    tool_config=replicate_agent_tool_config,
    run_input=run_input,
  )
  
  return replicate_team.run()

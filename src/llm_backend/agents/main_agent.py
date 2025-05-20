from llm_backend.agents.replicate_agent import ReplicateTeam







def run_replicate_team():
    replicate_team = ReplicateTeam()
    replicate_team.run()


async def run_iter():
    replicate_team = ReplicateTeam()
    nodes = []
    # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
    async with replicate_team.replicate_agent.iter("Extract the properties from the example_input.", deps={"beta": 0.7, "seed": 0, "text": "StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis.", "alpha": 0.3, "diffusion_steps": 10, "embedding_scale": 1.5}) as agent_run:
        async for node in agent_run:
            # Each node represents a step in the agent's execution
            nodes.append(node)
    print(nodes)

from pydantic_ai import Agent, RunContext

roulette_agent = Agent(
    "openai:gpt-4o",
    deps_type=int,
    output_type=bool,
    system_prompt=(
        "Use the `roulette_wheel` function to see if the "
        "customer has won based on the number they provide."
    ),
)


@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
    """check if the square is a winner"""
    return "winner" if square == ctx.deps else "loser"



def run_roulette_agent():
  success_number = 5
  result = roulette_agent.run_sync("I bet five is the winner", deps=success_number)
  print(result.output)
  result = roulette_agent.run_sync("Put my money on square eighteen", deps=success_number)
  print(result.output)
  return result.output
    
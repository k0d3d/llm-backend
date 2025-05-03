from dotenv import load_dotenv
load_dotenv()
from llm_backend.agents.main_agent import run_roulette_agent

if __name__ == "__main__":
  run_roulette_agent()
from typing import Callable, Optional, List, Dict
from src.my_dynamic_mas.core.data_models.subtask import SubTask


def customer_service_agent(
    task: SubTask, available_tools: Dict, agents: Dict[str, Callable]
) -> Optional[List[SubTask]]:
    """Function representing the Customer Service agent's logic."""
    print(f"Customer Service Agent received task: '{task.description}'")

    if (
        "help customer" in task.description.lower()
        or "answer question" in task.description.lower()
    ):
        # Logic to interact with the customer using available_tools (e.g., a knowledge base)
        print("Customer Service: Assisting the customer...")
        # Example: Use a knowledge base tool
        knowledge = available_tools.get("knowledge_base")
        if knowledge:
            answer = knowledge(task.description, task.assets.get("customer_query"))
            return answer  # Could return a direct result, not a new subtask
        else:
            return "Customer Service: Knowledge base not available."
    elif "track order" in task.description.lower():
        # Logic to track an order using an order tracking tool
        tracking_tool = available_tools.get("order_tracker")
        if tracking_tool and task.assets.get("order_id"):
            tracking_info = tracking_tool(task.assets["order_id"])
            return tracking_info
        else:
            return "Customer Service: Unable to track order."
    else:
        print("Customer Service: Unable to handle this customer request.")
        return None

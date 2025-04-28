from typing import Callable, Optional, List, Dict
from src.my_dynamic_mas.core.data_models.subtask import SubTask


def store_manager_agent(
    task: SubTask, available_tools: Dict, agents: Dict[str, Callable]
) -> Optional[List[SubTask]]:
    """Function representing the Store Manager agent's logic."""
    print(f"Store Manager Agent received task: '{task.description}'")

    if "manage products" in task.description.lower():
        # Logic to handle product management tasks using available_tools
        print("Store Manager: Handling product management...")
        # Example: Delegate to a product creation tool
        return [
            SubTask(
                sender="Store Manager",
                description="Create a new product based on user details.",
                assets=task.assets,
                available_tools=["product_creation_tool"],
                destination="Product Creator",  # Assuming this agent exists
            )
        ]
    elif "manage orders" in task.description.lower():
        # Logic to handle order management tasks
        print("Store Manager: Handling order management...")
        # Example: Use an order processing tool
        return [
            SubTask(
                sender="Store Manager",
                description="Process a new order.",
                assets=task.assets,
                available_tools=["order_processing_system"],
                destination="Order Processor",  # Assuming this agent exists
            )
        ]
    else:
        print("Store Manager: Unable to handle this task.")
        return None

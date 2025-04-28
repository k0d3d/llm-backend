# src/my_dynamic_mas/agents/ceo_agent.py

from typing import Optional, List, Dict, Callable, Union
from src.my_dynamic_mas.core.data_models.subtask import SubTask
from src.my_dynamic_mas.core.data_models.flow_control import (
    ControlFlowStep,
    TaskStep,
    Condition,
)


def ceo_agent(
    task: SubTask, available_tools: Dict, agents: Dict[str, Callable]
) -> Optional[List[Union[SubTask, ControlFlowStep]]]:
    """Function representing the CEO agent's logic."""
    print(f"CEO Agent received task: '{task.description}'")

    if "find products, create products, create orders" in task.description.lower():
        flow = ControlFlowStep(
            flow_type="sequential",
            steps=[
                TaskStep(agent="Product Finder", inputs={"query": task.description}),
                TaskStep(
                    agent="Product Creator",
                    inputs={"product_details": "$product_finder_result"},
                ),  # Assuming result is stored in context
                TaskStep(
                    agent="Order Processor",
                    inputs={
                        "order_data": "$product_creator_result",
                        "user_info": task.assets.get("user"),
                    },
                ),
            ],
        )
        return [flow]
    elif "help customer" in task.description.lower():
        return [
            SubTask(
                destination="Customer Service",
                description=task.description,
                assets=task.assets,
            )
        ]
    else:
        return None

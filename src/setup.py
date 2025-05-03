# src/my_dynamic_mas/main.py

from ast import Dict, List
from typing import Callable, Optional, Union
from src.my_dynamic_mas.core.task_orchestrator import TaskOrchestrator
from src.my_dynamic_mas.agents.ceo_agent import ceo_agent
from src.my_dynamic_mas.agents.store_manager_agent import store_manager_agent
from src.my_dynamic_mas.agents.customer_service_agent import customer_service_agent
from src.my_dynamic_mas.tools.woocommerce_api import call_woocommerce_api
from src.my_dynamic_mas.core.function_registry import FunctionRegistry
from src.my_dynamic_mas.flow_control_elements.conditions.inventory_conditions import (
    is_product_in_stock,
)
from src.my_dynamic_mas.flow_control_elements.actions.order_actions import (
    create_order,
    log_message,
)
from src.my_dynamic_mas.core.data_models.subtask import SubTask
from src.my_dynamic_mas.core.data_models.flow_control import (
    ControlFlowStep,
    TaskStep,
    Condition,
    Action,
)


def check_user_premium(user_id: str, context: Dict) -> bool:
    premium_users = context.get("premium_users", [])
    return user_id in premium_users


def get_wishlist_items(user_id: str, context: Dict) -> list:
    wishlists = context.get("wishlists", {})
    return wishlists.get(user_id, [])


def process_wishlist_item(item: dict, context: Dict):
    print(f"Processing wishlist item: {item['name']}")
    return f"Processed: {item['name']}"


def log_message(message: str, context: Dict):
    print(f"LOG: {message}")


if __name__ == "__main__":
    agents = {
        "CEO": ceo_agent,
        "Store Manager": store_manager_agent,
        "Customer Service": customer_service_agent,
        "Product Finder": lambda task, tools, agents: {
            "products": [
                {"id": 1, "name": "Awesome Shoes"},
                {"id": 2, "name": "Cool Hat"},
            ]
        },
        "Product Reviewer": lambda task, tools, agents: {
            "reviews": ["Good!", "Excellent!"]
        },
        "Order Processor": lambda task, tools, agents: {"order_id": 456},
    }

    tools = {
        "woocommerce_api": call_woocommerce_api,
        # ... other tools ...
    }

    function_registry_instance = FunctionRegistry()
    function_registry_instance.register("is_product_available", is_product_in_stock)
    function_registry_instance.register("place_new_order", create_order)
    function_registry_instance.register("check_premium", check_user_premium)
    function_registry_instance.register("get_wishlist", get_wishlist_items)
    function_registry_instance.register("process_wishlist", process_wishlist_item)
    function_registry_instance.register("log_info", log_message)

    orchestrator = TaskOrchestrator(
        agents=agents,
        tools=tools,
        function_registry=function_registry_instance.registry,
    )

    # Example task demonstrating nested conditions and a loop
    complex_flow_task = SubTask(
        sender="user",
        description="Process user wishlist if premium.",
        assets={"user_id": "user1"},
        destination="CEO",
    )

    def ceo_complex_flow(
        task: SubTask, available_tools: Dict, agents: Dict[str, Callable]
    ) -> Optional[List[Union[SubTask, ControlFlowStep]]]:
        flow = ControlFlowStep(
            flow_type="conditional",
            condition=Condition(
                function="check_premium", args={"user_id": task.assets["user_id"]}
            ),
            on_true=[
                ControlFlowStep(
                    flow_type="sequential",
                    steps=[
                        TaskStep(
                            agent="Store Manager",
                            inputs={
                                "action": "get_wishlist",
                                "user_id": task.assets["user_id"],
                            },
                            callback="wishlist_items",
                        ),
                        ControlFlowStep(
                            flow_type="iterate",
                            iterator="wishlist_items_result",
                            iteration_variable="wishlist_item",
                            operation=Action(
                                function="process_wishlist",
                                args={"item": "$wishlist_item"},
                            ),
                        ),
                        Action(
                            function="log_info",
                            args={
                                "message": "Wishlist processing complete for premium user."
                            },
                        ),
                    ],
                )
            ],
            on_false=[
                Action(
                    function="log_info",
                    args={"message": f"User {task.assets['user_id']} is not premium."},
                )
            ],
        )
        return [flow]

    # Override CEO agent for this example
    orchestrator.agents["CEO"] = ceo_complex_flow
    orchestrator.context["premium_users"] = ["user1", "user2"]
    orchestrator.context["wishlists"] = {
        "user1": [
            {"id": 101, "name": "Fancy Gadget"},
            {"id": 102, "name": "Cool Thing"},
        ]
    }

    orchestrator.submit_task(complex_flow_task)
    results = orchestrator.run()
    print("Final Results:", results)

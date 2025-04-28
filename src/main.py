# src/dynamic_mas/main.py

from src.dynamic_mas.core.task_orchestrator import TaskOrchestrator
from src.dynamic_mas.agents.ceo_agent import ceo_agent
from src.dynamic_mas.agents.store_manager_agent import store_manager_agent
from src.dynamic_mas.agents.customer_service_agent import customer_service_agent
from src.dynamic_mas.tools.woocommerce_api import call_woocommerce_api
from src.dynamic_mas.core.function_registry import FunctionRegistry
from src.dynamic_mas.flow_control_elements.conditions.inventory_conditions import (
    is_product_in_stock,
)
from src.dynamic_mas.flow_control_elements.actions.order_actions import create_order
from src.dynamic_mas.core.data_models.subtask import SubTask

if __name__ == "__main__":
    agents = {
        "CEO": ceo_agent,
        "Store Manager": store_manager_agent,
        "Customer Service": customer_service_agent,
        "Product Finder": lambda task, tools, agents: {
            "products": [{"id": 1, "name": "Awesome Shoes"}]
        },  # Mock
        "Product Creator": lambda task, tools, agents: {"product_id": 123},  # Mock
        "Order Processor": lambda task, tools, agents: {"order_id": 456},  # Mock
    }

    tools = {
        "woocommerce_api": call_woocommerce_api,
        # ... other tools ...
    }

    function_registry_instance = FunctionRegistry()
    function_registry_instance.register("is_product_available", is_product_in_stock)
    function_registry_instance.register("place_new_order", create_order)

    orchestrator = TaskOrchestrator(
        agents=agents,
        tools=tools,
        function_registry=function_registry_instance.registry,
    )

    # Example initial task that should trigger the flow
    initial_task = SubTask(
        sender="user",
        description="Find products, create products, create orders for Trendy Threads store.",
        assets={"store_name": "Trendy Threads", "user": {"id": "user1"}},
        destination="CEO",
    )
    orchestrator.submit_task(initial_task)

    results = orchestrator.run()
    print("Final Results:", results)

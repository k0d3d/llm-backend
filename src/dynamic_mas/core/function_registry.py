from ast import Dict
from typing import Callable, Optional


class FunctionRegistry:
    def __init__(self):
        self.registry: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        """Registers a function with a given name."""
        self.registry[name] = func

    def get(self, name: str) -> Optional[Callable]:
        """Retrieves a function by its name."""
        return self.registry.get(name)


# Example usage in main.py:
if __name__ == "__main__":
    from src.my_dynamic_mas.flow_control_elements.conditions.inventory_conditions import (
        is_product_in_stock,
    )
    from src.my_dynamic_mas.flow_control_elements.actions.order_actions import (
        create_order,
    )

    registry = FunctionRegistry()
    registry.register("is_product_available", is_product_in_stock)
    registry.register("place_new_order", create_order)

    # Pass this registry to the TaskOrchestrator
    # orchestrator = TaskOrchestrator(agents=agents, tools=tools, function_registry=registry.registry)

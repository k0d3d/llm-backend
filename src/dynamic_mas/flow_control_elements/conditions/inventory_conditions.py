from ast import Dict


def is_product_in_stock(product_id: str, context: Dict) -> bool:
    # In a real scenario, you'd use a tool to check inventory
    inventory = context.get("inventory_data", {})
    return inventory.get(product_id, 0) > 0

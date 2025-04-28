

from typing import Dict
from src.dynamic_mas.tools.woocommerce_api import call_woocommerce_api


def create_order(order_details: Dict, context: Dict):
    response = call_woocommerce_api(path="orders", method="POST", data=order_details)
    return response.get("id")  # Return the order ID
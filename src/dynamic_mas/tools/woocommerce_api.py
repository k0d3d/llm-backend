# src/my_dynamic_mas/tools/woocommerce_api.py

import requests
from typing import Dict, Any

# Load configuration (e.g., from config/config.yaml)
# For example:
# import yaml
# with open("src/my_dynamic_mas/config/config.yaml", "r") as f:
#     config = yaml.safe_load(f)
#
# WOOCOMMERCE_BASE_URL = config.get("woocommerce_base_url")
# WOOCOMMERCE_CONSUMER_KEY = config.get("woocommerce_consumer_key")
# WOOCOMMERCE_CONSUMER_SECRET = config.get("woocommerce_consumer_secret")

WOOCOMMERCE_BASE_URL = "YOUR_WOOCOMMERCE_URL"  # Replace with your actual URL
WOOCOMMERCE_CONSUMER_KEY = "YOUR_CONSUMER_KEY"  # Replace with your actual key
WOOCOMMERCE_CONSUMER_SECRET = "YOUR_CONSUMER_SECRET"  # Replace with your actual secret


def call_woocommerce_api(
    path: str,
    method: str = "GET",
    params: Dict[str, Any] = None,
    data: Dict[str, Any] = None,
) -> Dict[Any, Any]:
    """
    A generic tool to interact with the WooCommerce REST API.

    Args:
        path: The API endpoint path (e.g., 'products', 'orders').
        method: The HTTP method (GET, POST, PUT, DELETE).
        params: Dictionary of query parameters.
        data: Dictionary of request body for POST/PUT.

    Returns:
        A dictionary representing the JSON response from the API.
    """
    url = f"{WOOCOMMERCE_BASE_URL}/wp-json/wc/v3/{path}"  # Assuming v3 of the API
    auth = (WOOCOMMERCE_CONSUMER_KEY, WOOCOMMERCE_CONSUMER_SECRET)
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.request(
            method, url, auth=auth, params=params, headers=headers, json=data
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling WooCommerce API: {e}")
        if response is not None:
            print(f"Response status code: {response.status_code}")
            print(f"Response text: {response.text}")
        return {"error": str(e)}


# Example usage within an agent:
if __name__ == "__main__":
    # Example: Get all products
    products = call_woocommerce_api("products")
    print("Products:", products)

    # Example: Create a new product (requires POST and data)
    new_product_data = {
        "name": "Test Product from AI",
        "type": "simple",
        "regular_price": "25.00",
    }
    created_product = call_woocommerce_api(
        "products", method="POST", data=new_product_data
    )
    print("Created Product:", created_product)

    # Example: Get orders with specific parameters
    orders = call_woocommerce_api("orders", params={"status": "processing"})
    print("Processing Orders:", orders)

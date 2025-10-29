"""
Replicate API Error Parser

Parses Replicate API error responses into structured format for error recovery.
"""

from typing import Dict, Optional, List, Any
import re


class ReplicateErrorParser:
    """Parse Replicate API errors into structured format"""

    @staticmethod
    def parse_validation_error(error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse 422 validation error into structured format.

        Example Replicate error:
        {
            "detail": "aspect_ratio: aspect_ratio must be one of the following: '16:9', '1:1', '21:9', ..."
        }

        Returns:
        {
            "error_type": "validation",
            "field": "aspect_ratio",
            "message": "aspect_ratio must be one of ...",
            "current_value": None,  # Extracted from context if available
            "valid_values": ["16:9", "1:1", "21:9", ...],
            "raw_error": {...}
        }
        """
        detail = error_data.get("detail", "")

        # Try to extract field name and validation message
        match = re.match(r"(\w+):\s*(.+)", detail)
        if not match:
            return {
                "error_type": "validation",
                "field": None,
                "message": detail,
                "valid_values": [],
                "raw_error": error_data
            }

        field_name = match.group(1)
        message = match.group(2)

        # Extract valid enum values if present
        valid_values = []
        enum_match = re.search(r"must be one of the following:\s*(.+)", message)
        if enum_match:
            # Parse: '16:9', '1:1', '21:9', ...
            values_str = enum_match.group(1)
            valid_values = re.findall(r"'([^']+)'", values_str)

        return {
            "error_type": "validation",
            "field": field_name,
            "message": message,
            "valid_values": valid_values,
            "raw_error": error_data
        }

    @staticmethod
    def parse_error(status_code: int, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse any Replicate API error.

        Returns:
        {
            "error_type": "validation" | "authentication" | "not_found" | "rate_limit" | "server_error",
            "recoverable": bool,
            "field": str | None,
            "message": str,
            "valid_values": List[str],
            "raw_error": dict
        }
        """
        # Map status codes to error types
        error_type_map = {
            401: "authentication",
            403: "authentication",
            404: "not_found",
            422: "validation",
            429: "rate_limit",
            500: "server_error",
            502: "server_error",
            503: "server_error"
        }

        error_type = error_type_map.get(status_code, "unknown")

        # Validation errors are recoverable
        recoverable = error_type == "validation"

        # Parse based on error type
        if error_type == "validation":
            parsed = ReplicateErrorParser.parse_validation_error(error_data)
            parsed["recoverable"] = True
            return parsed

        # Non-validation errors
        detail = error_data.get("detail", "") or error_data.get("error", "")
        return {
            "error_type": error_type,
            "recoverable": False,
            "field": None,
            "message": detail or f"HTTP {status_code} error",
            "valid_values": [],
            "raw_error": error_data
        }

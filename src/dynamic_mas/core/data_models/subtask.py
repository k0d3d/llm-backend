from typing import Optional, Callable, List, Dict
from pydantic import BaseModel


class SubTask(BaseModel):
    sender: str
    description: str
    assets: Dict = {}
    available_tools: List[str] = []
    callback_function: Optional[Callable] = None
    structured_output_schema: Optional[Dict] = None
    destination: str
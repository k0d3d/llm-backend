from typing import List, Dict, Union, Callable, Optional
from pydantic import BaseModel


class Condition(BaseModel):
    name: str
    function: Optional[Callable] = (
        None  # Or a string reference to a function in the registry
    )
    args: Dict = {}


class Action(BaseModel):
    name: str
    function: Optional[Callable] = None  # Or a string reference
    args: Dict = {}
    next_steps: Optional[List[Union["ControlFlowStep", "TaskStep"]]] = None


class TaskStep(BaseModel):
    type: str = "task"
    agent: str
    inputs: Dict = {}
    callback: Optional[str] = None  # Name of a function to call with the result


class ControlFlowStep(BaseModel):
    type: str = "flow"
    flow_type: str  # e.g., "conditional", "sequential", "iterate"
    steps: List[Union["ControlFlowStep", "TaskStep", "Condition", "Action"]] = []
    condition: Optional[Condition] = None
    on_true: Optional[List[Union["ControlFlowStep", "TaskStep"]]] = None
    on_false: Optional[List[Union["ControlFlowStep", "TaskStep"]]] = None
    iterator: Optional[str] = None  # Key in the context to iterate over
    operation: Optional[Action] = None


# To allow recursive definitions
ControlFlowStep.update_forward_refs()
Action.update_forward_refs()

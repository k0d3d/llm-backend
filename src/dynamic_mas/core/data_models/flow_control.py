from typing import List, Dict, Union, Callable, Optional
from pydantic import BaseModel


class Condition(BaseModel):
    name: str
    function: Optional[str] = None  # String reference to function in registry
    args: Dict = {}


class Action(BaseModel):
    name: str
    function: Optional[str] = None  # String reference
    args: Dict = {}
    # Removed next_steps for clarity, flow is now controlled by ControlFlowStep


class TaskStep(BaseModel):
    type: str = "task"
    agent: str
    inputs: Dict = {}
    callback: Optional[str] = None  # Name of a function to call with the result


class ControlFlowStep(BaseModel):
    type: str = "flow"
    flow_type: str  # "sequential", "conditional", "iterate"
    steps: Optional[List[Union["ControlFlowStep", "TaskStep"]]] = (
        None  # For sequential and potentially others
    )
    condition: Optional[Condition] = None
    on_true: Optional[List[Union["ControlFlowStep", "TaskStep"]]] = None
    on_false: Optional[List[Union["ControlFlowStep", "TaskStep"]]] = None
    iterator: Optional[str] = None  # Key in the context to iterate over
    iteration_variable: Optional[str] = (
        "item"  # Name of the variable for the current item in the loop
    )
    operation: Optional[Union[Action, TaskStep, "ControlFlowStep"]] = (
        None  # Operation to perform in the loop
    )


# To allow recursive definitions
ControlFlowStep.update_forward_refs()

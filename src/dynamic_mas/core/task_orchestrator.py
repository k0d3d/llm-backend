from typing import List, Dict, Callable, Union
from src.my_dynamic_mas.core.data_models.subtask import SubTask
from src.my_dynamic_mas.core.data_models.flow_control import (
    ControlFlowStep,
    TaskStep,
    Condition,
    Action,
)


class TaskOrchestrator:
    def __init__(
        self,
        agents: Dict[str, Callable],
        tools: Dict[str, Callable],
        function_registry: Dict[str, Callable],
    ):
        self.agents = agents
        self.tools = tools
        self.function_registry = function_registry
        self.task_queue: List[Union[SubTask, ControlFlowStep]] = []
        self.results: Dict[str, any] = {}
        self.context: Dict[
            str, any
        ] = {}  # To store intermediate results for flow control

    def submit_task(self, task: Union[SubTask, ControlFlowStep]):
        """Submits a new task or control flow step to the orchestrator."""
        self.task_queue.append(task)
        print(
            f"Submitted: '{task.model_dump_json()[:50]}...' for '{getattr(task, 'destination', 'flow_control')}'"
        )

    def run(self):
        """Processes items in the queue until it's empty."""
        while self.task_queue:
            item = self.task_queue.pop(0)

            if isinstance(item, SubTask):
                self._process_subtask(item)
            elif isinstance(item, ControlFlowStep):
                self._process_control_flow(item)
            else:
                print(f"Error: Unknown item type in task queue: {item}")

        print("Task orchestration complete.")
        return self.results

    def _process_subtask(self, task: SubTask):
        """Processes a SubTask by routing it to the appropriate agent."""
        destination = task.destination
        if destination in self.agents:
            agent = self.agents[destination]
            print(f"Processing task '{task.description}' with agent '{destination}'")
            next_tasks_or_result = agent(task, self.tools, self.agents)
            self._handle_agent_response(task, next_tasks_or_result)
        else:
            print(
                f"Error: Unknown destination agent '{destination}' for task: '{task.description}'"
            )

    def _handle_agent_response(self, current_task: SubTask, response: any):
        """Handles the response from an agent."""
        if isinstance(response, list) and all(
            isinstance(item, (SubTask, ControlFlowStep)) for item in response
        ):
            print(
                f"Agent '{current_task.destination}' delegated {len(response)} new items."
            )
            self.task_queue.extend(response)
        elif isinstance(response, ControlFlowStep):
            print(f"Agent '{current_task.destination}' returned a control flow step.")
            self.task_queue.append(response)
        elif response is not None:
            print(
                f"Agent '{current_task.destination}' returned a result for task '{current_task.description}'."
            )
            self.results[current_task.description] = response
            if current_task.callback_function:
                self.context[current_task.callback_function] = (
                    response  # Store in context for potential later use
                )
        else:
            print(
                f"Agent '{current_task.destination}' returned no further tasks or result for '{current_task.description}'."
            )

    def _process_control_flow(self, flow_step: ControlFlowStep):
        """Processes a ControlFlowStep."""
        flow_type = flow_step.flow_type

        if flow_type == "sequential":
            print(f"Executing sequential flow with {len(flow_step.steps)} steps.")
            self.task_queue.extend(flow_step.steps)
        elif flow_type == "conditional":
            if flow_step.condition:
                condition_func = self.function_registry.get(
                    flow_step.condition.function
                )
                if condition_func:
                    condition_result = condition_func(
                        **flow_step.condition.args, context=self.context
                    )
                    print(
                        f"Evaluating condition '{flow_step.condition.name}': {condition_result}"
                    )
                    if condition_result and flow_step.on_true:
                        self.task_queue.extend(flow_step.on_true)
                    elif not condition_result and flow_step.on_false:
                        self.task_queue.extend(flow_step.on_false)
                else:
                    print(
                        f"Error: Condition function '{flow_step.condition.function}' not found."
                    )
            else:
                print("Error: Conditional flow without a condition.")
        elif flow_type == "iterate":
            if flow_step.iterator and flow_step.operation:
                items = self.context.get(flow_step.iterator, [])
                if isinstance(items, list):
                    print(
                        f"Iterating over '{flow_step.iterator}' with {len(items)} items."
                    )
                    for item in items:
                        action_func = self.function_registry.get(
                            flow_step.operation.function
                        )
                        if action_func:
                            action_result = action_func(
                                **flow_step.operation.args,
                                item=item,
                                context=self.context,
                            )
                            # Handle the result of the action, potentially adding new tasks
                            print(
                                f"Iteration action '{flow_step.operation.name}' result: {action_result}"
                            )
                            # You might want to add new SubTasks or update context based on this result
                        else:
                            print(
                                f"Error: Iteration action function '{flow_step.operation.function}' not found."
                            )
                else:
                    print(
                        f"Error: '{flow_step.iterator}' is not a list in the context."
                    )
            else:
                print("Error: Iterative flow missing iterator or operation.")
        else:
            print(f"Error: Unknown flow type '{flow_type}'.")

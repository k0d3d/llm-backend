from typing import List, Dict, Callable, Union
from dynamic_mas.core.data_models.subtask import SubTask
from dynamic_mas.core.data_models.flow_control import (
    ControlFlowStep,
    TaskStep,
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
                f"Agent '{current_task.destination}' returned a result for task '{current_task.description}'. Storing in context."
            )
            self.context[
                current_task.description.replace(" ", "_").lower() + "_result"
            ] = response  # Store in context
            if current_task.callback_function:
                self.context[current_task.callback_function] = (
                    response  # Also store by callback name if provided
                )
        else:
            print(
                f"Agent '{current_task.destination}' returned no further tasks or result for '{current_task.description}'."
            )

    def _process_control_flow(self, flow_step: ControlFlowStep):
        """Processes a ControlFlowStep."""
        flow_type = flow_step.flow_type

        if flow_type == "sequential" and flow_step.steps:
            print(f"Executing sequential flow with {len(flow_step.steps)} steps.")
            self.task_queue.extend(flow_step.steps)
        elif flow_type == "conditional" and flow_step.condition:
            condition_func = self.function_registry.get(flow_step.condition.function)
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
        elif flow_type == "iterate" and flow_step.iterator and flow_step.operation:
            items = self.context.get(flow_step.iterator, [])
            if isinstance(items, list):
                print(f"Iterating over '{flow_step.iterator}' with {len(items)} items.")
                for item in items:
                    iteration_context = self.context.copy()
                    iteration_context[flow_step.iteration_variable] = item
                    if isinstance(flow_step.operation, TaskStep):
                        task = SubTask(
                            sender="flow_control",
                            description=f"Iteration task: {flow_step.operation.agent}",
                            assets=flow_step.operation.inputs,
                            destination=flow_step.operation.agent,
                            callback_function=flow_step.operation.callback,  # Propagate callback if needed
                        )
                        self.submit_task(task)
                    elif isinstance(flow_step.operation, Action):
                        action_func = self.function_registry.get(
                            flow_step.operation.function
                        )
                        if action_func:
                            action_result = action_func(
                                **flow_step.operation.args, context=iteration_context
                            )
                            print(
                                f"Iteration action '{flow_step.operation.name}' result: {action_result}"
                            )
                            # Store the result in the context if needed
                            self.context[f"{flow_step.operation.name}_result"] = (
                                action_result
                            )
                        else:
                            print(
                                f"Error: Iteration action function '{flow_step.operation.function}' not found."
                            )
                    elif isinstance(flow_step.operation, ControlFlowStep):
                        # Process nested control flow
                        self._process_control_flow(
                            flow_step.operation.copy(update=iteration_context)
                        )  # Pass updated context
                    else:
                        print(
                            f"Error: Unsupported operation type in iteration: {type(flow_step.operation)}"
                        )
            else:
                print(f"Error: '{flow_step.iterator}' is not a list in the context.")
        else:
            print(f"Error: Unknown or incomplete flow type '{flow_type}'.")

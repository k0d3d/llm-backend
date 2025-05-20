
import json

from llm_backend.core.types.common import RunInput
from pydantic_ai import Agent, ModelRetry, RunContext
from llm_backend.core.types.replicate import ExampleInput, AgentPayload, Props
from llm_backend.tools.replicate_tool import run_replicate


class ReplicateTeam:
    def __init__(
      self, 
      prompt,
      tool_config,
      run_input: RunInput,
      ):
        self.prompt = prompt
        self.description = tool_config.get("description")
        self.example_input = tool_config.get("example_input")
        self.latest_version = tool_config.get("latest_version")
        self.run_input = run_input

    def api_interaction_agent(self):
        """
          API Interaction Agent
          Handles authentication with Replicate.com
          Sends requests and receives responses
          Manages retry logic and error handling
        """
        api_interaction_agent = Agent(
            "openai:gpt-4o",
            deps_type=AgentPayload, 
            output_type=str,
            system_prompt=(
                """
                Provided with a payload to send to replicate.com
                
                """
            ),
        )

        @api_interaction_agent.tool
        def use_replicate_tool(ctx: RunContext[AgentPayload]):
            """
            Send the request to replicate.com and receive the response.
            """
            print(ctx.deps.input)
            return run_replicate(
                run_input=self.run_input,
                model_params={
                  "example_input": self.example_input,
                  "latest_version": self.latest_version,
                },
                input=ctx.deps.input,
            )

        return api_interaction_agent


    def replicate_agent(self):
      replicate_agent = Agent(
        "openai:gpt-4o",
        deps_type=ExampleInput,
        output_type=AgentPayload,
        system_prompt=(
            """
            Analyze the example_input. It contains properties that are used to run a model on replicate.com. 
            The props object contains all properties and affected properties.
            The exact prompt string must be the part of the final payload.
            The final output should be a json payload based on the example_input schema to send a request.
            Do not make up properties that are not in example_input.
            DO NOT wrap suggested input in a parent object
            DO NOT make up the input object. Rewrite example_input and use its schema.       

          """
        ),
      )


      @replicate_agent.system_prompt
      def get_example_input(ctx: RunContext[ExampleInput]):
          return f"{ctx.deps.example_input}"


      @replicate_agent.system_prompt
      def get_description(ctx: RunContext[ExampleInput]):
          return f"{ctx.deps.description}"

      @replicate_agent.system_prompt
      def get_prompt(ctx: RunContext[ExampleInput]):
          return f"{ctx.deps.prompt}"

    #   @replicate_agent.tool
    #   def check_payload_is_valid(ctx: RunContext[ExampleInput], payload: PayloadInput):
    #       """
    #       Check if the payload is valid based on the example_input.
    #       """
    #       # get all the keys from ctx.deps.example_input and compare to the keys in ctx.deps.props.all_props and ctx.deps.props.affected_props
          
    #       payload_input_dict = payload.input
          
    #       payload_input_keys = set(payload_input_dict.keys())
    #       all_props_set = set(ctx.deps.props.all_props)
    #       affected_props_set = set(ctx.deps.props.affected_props)

    #       # Check if all payload_input keys are in all_props
    #       if not all_props_set.issubset(payload_input_keys):
    #           invalid_props = all_props_set - payload_input_keys
    #           raise ModelRetry(f"Invalid properties found: {invalid_props}")

    #       # Check if affected_props is a subset of all_props
    #       if not affected_props_set.issubset(payload_input_keys):
    #           invalid_affected = affected_props_set - payload_input_keys
    #           raise ModelRetry(f"Invalid affected properties found: {invalid_affected}")
          

    #       return "True"

      @replicate_agent.tool(retries=3)
      def check_payload_contains_prompt(ctx: RunContext[ExampleInput], payload: AgentPayload) -> AgentPayload:
          """
          Check if the payload contains the prompt string and return True if it does.
          """
        #   print(prompt)

          payload_input_dict = payload.input
          print(payload_input_dict.model_dump().values())
          if ctx.deps.prompt not in payload_input_dict.model_dump().values():
              raise ModelRetry("Prompt not found in payload")

          return payload


      return replicate_agent


    def extract_props_agent(self):
        extract_props_agent = Agent(
            "openai:gpt-4o",
            deps_type=dict,
            output_type=Props,
            system_prompt=(
                """
              Analyze the example_input. It contains properties that are used to run a model on replicate.com. 
              Extract the properties from the example_input. 
              Return a json object with two properties: all_props and affected_props.        
              all_props should contain all properties from the example_input.
              affected_props should contain only properties that are going to change based on the prompt.
            """
                    ),
                )


        @extract_props_agent.tool
        def extract_all_props(ctx: RunContext[dict]) -> str:
            """
            Extract all properties from the example_input.
            """
            input_keys = set(ctx.deps.keys())
            return "\n".join(input_keys)


        @extract_props_agent.tool
        def check_valid_input_props(ctx: RunContext[dict], props: Props) -> str:
            """
            Check if the input props are valid based on the prompt.
            """
            # get all the keys from ctx.deps and compare to the keys in props.all_props and props.affected_props
            input_keys = set(ctx.deps.keys())
            all_props_set = set(props.all_props)
            affected_props_set = set(props.affected_props)

            # Check if all input keys are in all_props
            if not input_keys.issubset(all_props_set):
                invalid_props = input_keys - all_props_set
                raise ModelRetry(f"Invalid properties found: {invalid_props}")

            # Check if affected_props is a subset of all_props
            if not affected_props_set.issubset(all_props_set):
                invalid_affected = affected_props_set - all_props_set
                raise ModelRetry(f"Invalid affected properties found: {invalid_affected}")

            return "True"

        return extract_props_agent


    def run(self):
        props_agent = self.extract_props_agent()
        props = props_agent.run_sync(
            "Extract the properties from the example_input.",
            deps=self.example_input,
        )

        replicate_agent = self.replicate_agent()
        replicate_result = replicate_agent.run_sync(
            "Rewrite the example_input based on the affected properties provided.",
            deps=ExampleInput(
                example_input=self.example_input,
                description=self.description,
                prompt=self.prompt,
                props=props.output,
            ),
        )

        api_interaction_agent = self.api_interaction_agent()
        api_result =  api_interaction_agent.run_sync(
            "Send the request to replicate.com and receive the response.",
            deps=replicate_result.output,
        )

        return api_result.output
        return replicate_result.output

        
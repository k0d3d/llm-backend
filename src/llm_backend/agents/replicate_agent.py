
import json

from llm_backend.core.types.common import RunInput
from pydantic_ai import Agent, ModelRetry, RunContext
from llm_backend.core.types.replicate import ExampleInput, PayloadInput, Props


class ReplicateTeam:
    def __init__(self, run_input: RunInput):
        self.run_input = run_input


    def replicate_agent(self):
      replicate_agent = Agent(
        "openai:gpt-4o",
        deps_type=ExampleInput,
        output_type=PayloadInput,
        system_prompt=(
            """
            Analyze the example_input. It contains properties that are used to run a model on replicate.com. 
            The props object contains all properties and affected properties.
            The prompt must be the same as the value of the affected properties.
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

      @replicate_agent.tool
      def check_payload_is_valid(ctx: RunContext[ExampleInput], payload: PayloadInput):
          """
          Check if the payload is valid based on the example_input.
          """
          # get all the keys from ctx.deps.example_input and compare to the keys in ctx.deps.props.all_props and ctx.deps.props.affected_props
          
          payload_input_dict = json.loads(payload.input)
          
          payload_input_keys = set(payload_input_dict.keys())
          all_props_set = set(ctx.deps.props.all_props)
          affected_props_set = set(ctx.deps.props.affected_props)

          # Check if all payload_input keys are in all_props
          if not all_props_set.issubset(payload_input_keys):
              invalid_props = all_props_set - payload_input_keys
              raise ModelRetry(f"Invalid properties found: {invalid_props}")

          # Check if affected_props is a subset of all_props
          if not affected_props_set.issubset(payload_input_keys):
              invalid_affected = affected_props_set - payload_input_keys
              raise ModelRetry(f"Invalid affected properties found: {invalid_affected}")

          return "True"


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
        props = self.extract_props_agent.run_sync(
            "Extract the properties from the example_input.",
            deps={
                "beta": 0.7,
                "seed": 0,
                "text": "StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis.",
                "alpha": 0.3,
                "diffusion_steps": 10,
                "embedding_scale": 1.5,
            },
        )

        result = self.replicate_agent.run_sync(
            "Rewrite the example_input based on the affected properties provided.",
            deps=ExampleInput(
                example_input={
                  "beta": 0.7,
                  "seed": 0,
                  "text": "StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis.",
                  "alpha": 0.3,
                  "diffusion_steps": 10,
                  "embedding_scale": 1.5,
              },
              description="Generates speech from text",
              prompt=self.run_input.prompt,
              props=props.output,
          ),
        )

        return result.output
        

import os

from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from llm_backend.core.types.common import MessageType
from llm_backend.core.types.replicate import ExampleInput, AgentPayload, InformationInputResponse, InformationInputPayload
from llm_backend.tools.replicate_tool import run_replicate
from llm_backend.core.helpers import send_data_to_url_async

TOHJU_NODE_API = os.getenv("TOHJU_NODE_API", "https://api.tohju.com")
CORE_API_URL = os.getenv("CORE_API_URL", "https://core-api-d1kvr2.asyncdev.workers.dev")

class ReplicateTeam:
    def __init__(self, prompt, tool_config, run_input, hitl_enabled=False):
        self.prompt = prompt
        self.tool_config = tool_config
        self.run_input = run_input
        self.hitl_enabled = hitl_enabled
        self.example_input = tool_config.get("example_input", {})
        self.description = tool_config.get("description", "")
        self.latest_version = tool_config.get("latest_version", "")
        self.model_name = tool_config.get("model_name", "")

    def response_audit_agent(self):
        response_audit_agent = Agent(
            "openai:gpt-4o",
            deps_type=str,
            output_type=str,
            system_prompt=(
                """
                You are an AI assistant that provides feedback to users about their requests, responses, or errors.
                The user does not need to know information like the provider (e.g. Replicate.com).
                Remove all links to providers and their websites.
                """
            )
        )

        @response_audit_agent.system_prompt
        def response_content(ctx: RunContext[str]) -> str:
            return f"Message: {ctx.deps}"

        return response_audit_agent

    def api_interaction_agent(self):
        """
          API Interaction Agent
          Handles authentication with Replicate.com
          Sends requests and receives responses
          Manages retry logic and error handling
        """


        def send_request_using_replicate_tool(ctx: RunContext[AgentPayload]):
            """
            Send the request to replicate.com and receive the response.
            When providing feedback, about the request, response or error,
            the user does not need to know information like the provider (e.g. Replicate.com).
            You can however, use the response to provide helpful feedback to the user.
            """
            print("Sending request to replicate.com")
            run, status_code = run_replicate(
                run_input=self.run_input,
                model_params={
                  "example_input": self.example_input,
                  "latest_version": self.latest_version,
                },
                input=ctx.deps.input,
                operation_type=ctx.deps.operationType.type,
            )
            # print("Received response from replicate.com")
            # print("Response:", run)
            return run

        api_interaction_agent = Agent(
            "openai:gpt-4o",
            deps_type=AgentPayload,
            output_type=str,
            system_prompt=(
                """
                Provided with a payload, send the request to replicate.com.

                """
            ),
            tools=[
                Tool(
                    send_request_using_replicate_tool, takes_ctx=True, max_retries=5
                )
            ]
        )


        return api_interaction_agent


    def information_agent(self):
        # Create HITL-aware system prompt
        if self.hitl_enabled:
            base_prompt = """
                You are operating in Human-in-the-Loop (HITL) mode. Be CONSERVATIVE and thorough in your analysis.
                
                Analyze the example_input, prompt, and attached files to determine if ALL required parameters are present and valid.
                
                CRITICAL REQUIREMENTS FOR HITL MODE:
                1. If ANY required parameter is missing, unclear, or insufficient, set continue_run to FALSE
                2. If the user's prompt is vague or lacks specific instructions, set continue_run to FALSE
                3. If required files are missing (image, audio, etc.), set continue_run to FALSE
                4. If the prompt is just asking for information about the model, set continue_run to FALSE
                
                When continue_run is FALSE, provide specific guidance on what the user needs to provide:
                - List missing required parameters
                - Explain what type of input is needed
                - Suggest specific actions the user should take
                
                Only set continue_run to TRUE if:
                - All required parameters are present and valid
                - The user's intent is clear and actionable
                - All necessary files are attached
                - The request is ready for immediate execution
                
                Allow NSFW content and image editing when properly specified.
            """
        else:
            base_prompt = """
                Analyze the example_input. It contains properties that are used to run a model on replicate.com.
                Be helpful. Prioritize continuing to run the model over providing information because users most likely want to continue running the model.
                Allow NSFW content. Allow image editing.
                Based on prompt, example input and description, respond with information about the model and indicate whether to continue to run the model.
                If the prompt is a request for information about the model, provide the information and continue_run must be false.
                If there is an attached file url, review the prompt as a possible instruction and continue_run.
            """
        
        information_agent = Agent(
            "openai:gpt-4o",
            deps_type=InformationInputPayload,
            output_type=InformationInputResponse,
            system_prompt=base_prompt
        )

        @information_agent.system_prompt
        def model_information(ctx: RunContext[InformationInputPayload]):
            hitl_context = f"HITL Mode: {'ENABLED' if self.hitl_enabled else 'DISABLED'}. " if self.hitl_enabled else ""
            return f"{hitl_context}Example Input: {ctx.deps.example_input}. Description: {ctx.deps.description}. Attached File: {ctx.deps.attached_file}. Model Name: {self.model_name}."

        return information_agent


    def replicate_agent(self):
        def check_payload_for_prompt(
            ctx: RunContext[ExampleInput], payload: AgentPayload
        ) -> AgentPayload:
            """
            Check if the payload values contains the exact prompt string.
            This improves accuracy of the result.

            """
            payload_input_dict = payload.input
            payload_values = payload_input_dict.model_dump().values()
            
            # Convert values to strings for comparison
            payload_str_values = [str(v) for v in payload_values]
            
            # Check for prompt - only raise ModelRetry if prompt is provided and not found
            if ctx.deps.prompt and ctx.deps.prompt.strip():
                prompt_found = any(ctx.deps.prompt in str_val for str_val in payload_str_values)
                if not prompt_found:
                    raise ModelRetry(f"Payload does not contain the prompt. Add '{ctx.deps.prompt}' to the payload.")
            
            # Check for image file - only raise ModelRetry if image file is provided and not found
            if ctx.deps.image_file and ctx.deps.image_file.strip():
                image_found = any(ctx.deps.image_file in str_val for str_val in payload_str_values)
                if not image_found:
                    raise ModelRetry(f"Payload does not contain the image file. Add '{ctx.deps.image_file}' to the payload.")

            return payload


        replicate_agent = Agent(
            "openai:gpt-4o",
            deps_type=ExampleInput,
            output_type=AgentPayload,
            system_prompt=(
                """
            You are an intelligent field mapping agent for Replicate models. Your job is to create optimal payloads by understanding model schemas and user intent.
            
            CORE RESPONSIBILITIES:
            1. Analyze the example_input schema to understand field relationships and priorities
            2. Map user inputs (prompt, attachments, HITL edits) to the correct model fields
            3. Create a valid JSON payload that follows the example_input schema exactly
            4. Handle semantic field mapping (e.g., input_image → image for background removal models)
            
            INTELLIGENT FIELD MAPPING RULES:
            - For TEXT fields: Map prompt to the most appropriate field (prompt > text > input > instruction)
            - For IMAGE fields: Prioritize based on model purpose:
              * Background removal models: input_image/source_image → image
              * Image generation: prompt → prompt, reference images → image/input_image
              * Image editing: source → image/input_image, instructions → prompt
            - For AUDIO fields: Map to audio/file/input based on schema
            
            HITL EDIT INTEGRATION:
            - When HITL edits are provided (e.g., {"input_image": "url", "source_image": "url"}), intelligently map them:
              * If example_input has "image" but edits have "input_image", map input_image → image
              * If example_input has both, use the most specific field for the model type
              * Preserve user intent while respecting model schema
            
            SCHEMA ANALYSIS:
            - Identify primary vs secondary fields (image vs image_url)
            - Understand field purposes from names and model description
            - Detect required vs optional parameters
            
            OUTPUT REQUIREMENTS:
            - Return exact example_input structure with mapped values
            - Include operationType based on model description analysis
            - Ensure all user inputs are represented in the final payload
            - DO NOT create fields not in example_input
            - DO NOT wrap in parent objects
            
            VALIDATION:
            - Verify prompt appears in final payload if provided
            - Verify image files appear in appropriate fields if provided
            - Ensure HITL edits are properly integrated
          """
            ),
            tools=[Tool(check_payload_for_prompt, takes_ctx=True, max_retries=5, description="Check if the payload contains the prompt string and image file.")],
        )

        @replicate_agent.system_prompt
        def get_example_input(ctx: RunContext[ExampleInput]):
            return f"Example input: {ctx.deps.example_input}"

        @replicate_agent.system_prompt
        def get_description(ctx: RunContext[ExampleInput]):
            return f"Model description: {ctx.deps.description}"

        @replicate_agent.system_prompt
        def get_prompt(ctx: RunContext[ExampleInput]):
            return f"Prompt: {ctx.deps.prompt}"

        @replicate_agent.system_prompt
        def get_image_file(ctx: RunContext[ExampleInput]):
            return f"Image file url: {ctx.deps.image_file}"
        
        @replicate_agent.system_prompt
        def get_hitl_edits(ctx: RunContext[ExampleInput]):
            if ctx.deps.hitl_edits:
                return f"HITL Human Edits (integrate these into the payload): {ctx.deps.hitl_edits}"
            return "No HITL edits provided."


        return replicate_agent

    async def run(self):

        information_agent = self.information_agent()
        information = await information_agent.run(
            self.prompt,
            deps=InformationInputPayload(
                example_input=self.example_input,
                description=self.description,
                attached_file=None
            ),
        )

        if not information.output.continue_run:

            message_type = MessageType["REPLICATE_PREDICTION"]

            await send_data_to_url_async(
                data=information.output.response_information,
                url=f"{CORE_API_URL}/from-llm",
                crew_input=self.run_input,
                message_type=message_type,
            )
            return information.output.response_information

        replicate_agent = self.replicate_agent()
        replicate_result = await replicate_agent.run(
            "Rewrite the example_input based on the affected properties provided.",
            deps=ExampleInput(
                example_input=self.example_input,
                description=information.output.response_information,
                prompt=self.prompt,
                image_file=None,
            ),
        )

        api_interaction_agent = self.api_interaction_agent()
        api_result = await api_interaction_agent.run(
            "Send the request to replicate.com and receive the response.",
            deps=replicate_result.output,
        )

        response_audit_agent = self.response_audit_agent()
        response_audit_result = await response_audit_agent.run(
            "Audit the response from the request.",
            deps=api_result.output,
        )

        await send_data_to_url_async(
            data=response_audit_result.output,
            url=f"{CORE_API_URL}/from-llm",
            crew_input=self.run_input,
            message_type=MessageType["REPLICATE_PREDICTION"],
        )

        return response_audit_result.output


import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from llm_backend.core.types.common import MessageType
from llm_backend.core.types.replicate import (
    AgentPayload,
    AttachmentDiscoveryContext,
    AttachmentDiscoveryResult,
    ExampleInput,
    FinalGuardContext,
    FinalGuardDecision,
    FileRequirementAnalysis,
    FileRequirementContext,
    InformationInputPayload,
    InformationInputResponse,
    PayloadValidationContext,
    PayloadValidationOutput,
    ValidationIssueDetail,
)
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
        self.field_metadata = tool_config.get("field_metadata", {}) or {}
        self.hitl_alias_metadata = tool_config.get("hitl_alias_metadata", {}) or {}
        self.hitl_edits: Dict[str, Any] = {}
        self.attachments = self._collect_attachments()
        resolved_edits = self._resolve_hitl_edits()
        if resolved_edits:
            self.hitl_edits.update(resolved_edits)
        self.operation_type = self._resolve_operation_type()

    def _build_text_context(self) -> List[str]:
        """Gather free-form text sources that might reference attachments."""
        context_fragments: List[str] = []

        if isinstance(self.prompt, str) and self.prompt:
            context_fragments.append(self.prompt)

        prompt_doc = getattr(self.run_input, "prompt_document", None)
        if isinstance(prompt_doc, str) and prompt_doc:
            context_fragments.append(prompt_doc)

        # Include recent conversation snippets if available
        conversation = getattr(self.run_input, "conversation", None)
        if isinstance(conversation, list):
            for message in conversation[-10:]:
                if isinstance(message, dict):
                    text = message.get("content") or message.get("text")
                    if isinstance(text, str) and text:
                        context_fragments.append(text)

        return context_fragments

    def _collect_attachments(self) -> List[str]:
        """Gather candidate attachment URLs from run input and tool config, with AI fallback."""
        attachments: List[str] = []
        sources: List[Dict[str, Any]] = []

        agent_tool_config = getattr(self.run_input, "agent_tool_config", None)
        if isinstance(agent_tool_config, dict):
            replicate_entry = agent_tool_config.get("replicate-agent-tool") or agent_tool_config.get("replicate_agent_tool")
            if isinstance(replicate_entry, dict):
                sources.append(replicate_entry)
                data = replicate_entry.get("data")
                if isinstance(data, dict):
                    sources.append(data)

        sources.append(self.tool_config)

        for source in sources:
            if not isinstance(source, dict):
                continue
            raw_attachments = source.get("attachments")
            if isinstance(raw_attachments, list):
                for item in raw_attachments:
                    if isinstance(item, str) and item and item not in attachments:
                        attachments.append(item)

        for source in sources:
            if not isinstance(source, dict):
                continue
            last_asset = source.get("last_uploaded_asset")
            if isinstance(last_asset, str) and last_asset and last_asset not in attachments:
                attachments.append(last_asset)

        # If deterministic harvesting failed, use AI assistant
        if not attachments and self.hitl_enabled:
            triage_agent = self._attachment_triage_agent()
            try:
                async def run_triage() -> AttachmentDiscoveryResult:
                    discovery = await triage_agent.run(
                        "Analyze the provided context and identify any media URLs relevant to the model schema.",
                        deps=AttachmentDiscoveryContext(
                            prompt=self.prompt,
                            text_context=self._build_text_context(),
                            candidate_urls=[],
                            schema_metadata=self.field_metadata,
                            hitl_field_metadata=self.hitl_alias_metadata,
                            expected_media_fields=[field for field, meta in self.field_metadata.items() if meta.get("collection") or "image" in field.lower() or "audio" in field.lower()],
                        ),
                    )
                    return getattr(discovery, "output", None)

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(lambda: asyncio.run(run_triage()))
                        result = future.result()
                else:
                    result = asyncio.run(run_triage())

                if isinstance(result, AttachmentDiscoveryResult):
                    attachments = list(dict.fromkeys(result.attachments or []))
                    if result.mapping:
                        # Merge mapping into hitl edits so downstream consumers see structured assignments
                        for target, url in result.mapping.items():
                            if isinstance(target, str) and url:
                                self.hitl_edits[target] = url
            except Exception as triage_error:
                print(f"⚠️ Attachment triage agent failed: {triage_error}")

        return attachments

    def _attachment_triage_agent(self) -> Agent:
        """Agent that inspects prompt/context to discover potential attachments."""

        def gather_context(ctx: RunContext[AttachmentDiscoveryContext]) -> Dict[str, Any]:
            return {
                "prompt": ctx.deps.prompt,
                "text_context": ctx.deps.text_context,
                "candidate_urls": ctx.deps.candidate_urls,
                "schema_metadata": ctx.deps.schema_metadata,
                "hitl_field_metadata": ctx.deps.hitl_field_metadata,
                "expected_media_fields": ctx.deps.expected_media_fields,
            }

        agent = Agent(
            "openai:gpt-4.1-mini",
            deps_type=AttachmentDiscoveryContext,
            output_type=AttachmentDiscoveryResult,
            system_prompt=(
                """
                You are an Attachment Discovery Agent. Review the provided prompt and surrounding text context
                to find URLs or asset references that should be used as model inputs. Determine the most relevant
                assets for the model, considering the schema metadata and expected media fields.

                Respond strictly with the AttachmentDiscoveryResult schema, providing a list of attachment URLs.
                Include reasoning when helpful.
                """
            ),
            tools=[
                Tool(
                    gather_context,
                    takes_ctx=True,
                    description="Inspect prompt and context snippets to assist in attachment discovery."
                )
            ],
        )

        return agent

    def _resolve_hitl_edits(self) -> Dict[str, Any]:
        """Merge human edits from tool config, run input metadata, and orchestrator state."""
        edits: Dict[str, Any] = {}

        agent_tool_config = getattr(self.run_input, "agent_tool_config", None)
        if isinstance(agent_tool_config, dict):
            replicate_entry = agent_tool_config.get("replicate-agent-tool") or agent_tool_config.get("replicate_agent_tool")
            if isinstance(replicate_entry, dict):
                entry_edits = replicate_entry.get("hitl_edits")
                if isinstance(entry_edits, dict):
                    edits.update(entry_edits)
                data = replicate_entry.get("data")
                if isinstance(data, dict):
                    data_edits = data.get("hitl_edits")
                    if isinstance(data_edits, dict):
                        edits.update(data_edits)

        config_edits = self.tool_config.get("hitl_edits")
        if isinstance(config_edits, dict):
            edits.update(config_edits)

        runtime_edits = getattr(self.run_input, "human_edits", None)
        if isinstance(runtime_edits, dict):
            edits.update(runtime_edits)

        return edits

    def _resolve_operation_type(self) -> str:
        """Infer operation type for downstream validation."""
        candidate = (
            self.tool_config.get("operation_type")
            or self.tool_config.get("operationType")
            or "image"
        )
        if candidate not in {"image", "video", "text", "audio"}:
            return "image"
        return candidate

    def file_requirement_agent(self) -> Agent:
        """Agent responsible for determining required file inputs."""

        def schema_inspector(ctx: RunContext[FileRequirementContext]) -> Dict[str, Any]:
            schema = ctx.deps.example_input or {}
            schema_keys = list(schema.keys()) if isinstance(schema, dict) else []

            image_like = [key for key in schema_keys if "image" in key.lower() or "frame" in key.lower()]
            audio_like = [key for key in schema_keys if "audio" in key.lower() or "voice" in key.lower()]
            video_like = [key for key in schema_keys if "video" in key.lower()]

            has_prompt = any("prompt" in key.lower() for key in schema_keys)

            return {
                "schema_keys": schema_keys,
                "image_like": image_like,
                "audio_like": audio_like,
                "video_like": video_like,
                "has_prompt": has_prompt,
                "hitl_edits": ctx.deps.hitl_edits,
                "attachments": ctx.deps.attachments,
                "model_name": ctx.deps.model_name,
                "model_description": ctx.deps.model_description,
                "existing_payload": ctx.deps.existing_payload,
            }

        agent = Agent(
            "openai:gpt-4.1-mini",
            deps_type=FileRequirementContext,
            output_type=FileRequirementAnalysis,
            system_prompt=(
                """
                You are a Guardrail Agent that determines whether running the current model requires
                specific file inputs. Use the available schema inspection tool to understand the
                example_input, existing payload, model name, and description before responding.

                When you respond:
                - Populate required_files as a list of normalized types (e.g. "image", "audio").
                - Populate blocking_issues with ValidationIssueDetail entries when a required file is missing.
                - Set ready to False if blocking issues are present.
                - Add human friendly suggestions when action is required.

                Respond strictly with the FileRequirementAnalysis schema.
                """
            ),
            tools=[
                Tool(
                    schema_inspector,
                    takes_ctx=True,
                    description="Inspect the example_input schema, model description, and gather heuristic hints for required files."
                )
            ],
        )

        return agent

    def payload_validation_agent(self) -> Agent:
        """Agent that validates the candidate payload against requirements."""

        def payload_snapshot(ctx: RunContext[PayloadValidationContext]) -> Dict[str, Any]:
            payload_dict = ctx.deps.candidate_payload.input.model_dump() if ctx.deps.candidate_payload else {}
            return {
                "payload": payload_dict,
                "required_files": ctx.deps.required_files,
                "attachments": ctx.deps.attachments,
                "hitl_edits": ctx.deps.hitl_edits,
                "operation_type": ctx.deps.operation_type,
            }

        agent = Agent(
            "openai:gpt-4.1-mini",
            deps_type=PayloadValidationContext,
            output_type=PayloadValidationOutput,
            system_prompt=(
                """
                You are a Payload Validation Agent. Review the candidate payload and ensure that it
                contains all required parameters, matches the example_input schema semantics, and
                integrates human edits or attachments when necessary.

                Guidance:
                - Leverage the payload snapshot tool before responding.
                - If required files are missing, add blocking issues with severity "error".
                - If you can auto-resolve a missing value by applying HITL edits or attachments, note it in auto_fixes
                  and update the payload accordingly before returning it.
                - Provide warnings for non-blocking improvements.
                - Set ready to False whenever blocking issues exist.

                Respond strictly with the PayloadValidationOutput schema.
                """
            ),
            tools=[
                Tool(
                    payload_snapshot,
                    takes_ctx=True,
                    description="Review the candidate payload, required files, and available human edits/attachments."
                )
            ],
        )

        return agent

    def replicate_agent(self) -> Agent:
        """Agent that generates the Replicate API payload from example input and prompt."""

        agent = Agent(
            "openai:gpt-4.1-mini",
            deps_type=ExampleInput,
            output_type=AgentPayload,
            system_prompt=(
                """
                You are a Replicate Payload Generation Agent. Your job is to transform the user's prompt
                and any provided files into a valid Replicate API payload that matches the example_input schema.

                Guidelines:
                - Use the example_input as your schema template
                - Integrate the user's prompt into appropriate text/prompt fields
                - Apply any image_file or attachments to relevant image/file fields
                - Honor any hitl_edits by overriding corresponding fields
                - Review schema_metadata and hitl_field_metadata to understand which fields are collections
                  (e.g., image_input arrays) or nested dictionaries, and ensure your payload respects those
                  data shapes when applying edits
                - Only include fields that exist in the example_input schema

                **Conversation History Context**
                - If conversation history is provided (last 10 messages), analyze it for context
                - User prompts may reference previous messages (e.g., "use the settings from before", "like I said earlier")
                - Look for configuration values mentioned in previous messages:
                  * Image generation settings (aspect_ratio, output_format, prompt_upsampling, etc.)
                  * Prompts or descriptions from earlier in the conversation
                  * File URLs or attachments shared previously
                - When the current prompt references "history", "previous", "before", "earlier", or "that":
                  * Search conversation for relevant settings/values
                  * Apply those historical values to the current payload
                - Prioritize: current prompt > conversation history > example_input defaults

                **NEW: Structured Form Values Support**
                - If structured_form_values are provided, these are authoritative user-provided values from a form
                - You MUST intelligently map structured_form_values field names to the correct example_input field names
                - Common mappings:
                  * "prompt" → "input" (for text input fields)
                  * "image_input" → "file_input" (for image file fields)
                  * "image" → "input_image" or "image_url" or "file_input"
                - Use schema_metadata to understand the actual API field names
                - When structured_form_values["prompt"] exists, find the correct text input field in example_input
                  (could be "input", "prompt", "text", "instruction", etc.) and use that value
                - When structured_form_values contains arrays (like image_input=[]), map to the correct array field
                  in example_input (could be "file_input", "images", "input_images", etc.)

                Respond strictly with the AgentPayload schema containing the transformed input.
                """
            ),
        )

        return agent

    def information_agent(self) -> Agent:
        """Agent that analyzes if the request can proceed or needs more information."""
        
        agent = Agent(
            "openai:gpt-4.1-mini",
            deps_type=InformationInputPayload,
            output_type=InformationInputResponse,
            system_prompt=(
                """
                You are an Information Gathering Agent. Analyze the user's prompt and determine if
                you have enough information to proceed with the Replicate model execution.
                
                Guidelines:
                - Check if the prompt is clear and actionable
                - Verify if required files are present when needed
                - Set continue_run to True if you can proceed
                - Set continue_run to False if you need clarification or more information
                - Provide helpful response_information explaining what's needed or confirming readiness
                
                Respond strictly with the InformationInputResponse schema.
                """
            ),
        )
        
        return agent

    def api_interaction_agent(self) -> Agent:
        """Agent that executes the Replicate API call."""
        
        async def execute_replicate(ctx: RunContext[AgentPayload]) -> Dict[str, Any]:
            """Execute the Replicate API call with the provided payload."""
            payload_dict = ctx.deps.input.model_dump()
            result = await run_replicate(
                version=ctx.deps.version or self.latest_version,
                input_data=payload_dict,
            )
            return result
        
        agent = Agent(
            "openai:gpt-4.1-mini",
            deps_type=AgentPayload,
            system_prompt=(
                """
                You are an API Interaction Agent. Execute the Replicate API call and return the result.
                Use the execute_replicate tool to make the actual API call.
                """
            ),
            tools=[
                Tool(
                    execute_replicate,
                    takes_ctx=True,
                    description="Execute the Replicate API call with the validated payload."
                )
            ],
        )
        
        return agent

    def response_audit_agent(self) -> Agent:
        """Agent that audits and formats the Replicate API response."""
        
        agent = Agent(
            "openai:gpt-4.1-mini",
            system_prompt=(
                """
                You are a Response Audit Agent. Review the Replicate API response and format it
                for the end user. Extract relevant output URLs, status information, and any errors.
                Provide a clean, user-friendly summary of the results.
                """
            ),
        )
        
        return agent

    def final_guard_agent(self) -> Agent:
        """Agent that performs a final schema compliance check before execution."""

        def compute_schema_diff(ctx: RunContext[FinalGuardContext]) -> Dict[str, Any]:
            candidate_payload = ctx.deps.candidate_payload.input.model_dump() if ctx.deps.candidate_payload else {}
            return {
                "example_input": ctx.deps.example_input,
                "candidate_payload": candidate_payload,
                "model_name": ctx.deps.model_name,
                "model_description": ctx.deps.model_description,
                "operation_type": ctx.deps.operation_type,
                "diff_preview": json.dumps({
                    "example_input": ctx.deps.example_input,
                    "candidate_payload": candidate_payload,
                }, indent=2),
            }

        agent = Agent(
            "openai:gpt-4.1-mini",
            deps_type=FinalGuardContext,
            output_type=FinalGuardDecision,
            system_prompt=(
                """
                You are the Final Guard Agent. Confirm that the payload about to be sent to the provider
                follows the example_input schema exactly (field names, types, and presence) and honours
                human edits. You must:
                - Use the compute_schema_diff tool to compare candidate payload against example_input.
                - Reject (approved = False) when required fields are missing, wrong, or incompatible.
                - Provide blocking issues containing actionable feedback for any mismatches.
                - Include a concise diff_summary highlighting important differences when available.

                Respond strictly with the FinalGuardDecision schema.
                """
            ),
            tools=[
                Tool(
                    compute_schema_diff,
                    takes_ctx=True,
                    description="Produce a JSON diff preview between example_input and the candidate payload."
                )
            ],
        )

        return agent

    def _format_issue_payload(
        self,
        stage: str,
        issues: List[ValidationIssueDetail],
        suggestions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        issue_dicts: List[Dict[str, Any]] = []
        message_lines = [f"{stage} requires attention:"]

        for issue in issues:
            issue_dict = {
                "field": issue.field,
                "issue": issue.issue,
                "severity": issue.severity,
                "suggested_fix": issue.suggested_fix,
                "auto_fixable": issue.auto_fixable,
            }
            issue_dicts.append(issue_dict)

            detail_line = f"- [{issue.severity.upper()}] {issue.field}: {issue.issue}"
            message_lines.append(detail_line)
            if issue.suggested_fix:
                message_lines.append(f"  Suggestion: {issue.suggested_fix}")

        if suggestions:
            message_lines.append("Recommended next steps:")
            for item in suggestions:
                if item:
                    message_lines.append(f"* {item}")

        return {
            "stage": stage,
            "issues": issue_dicts,
            "suggestions": suggestions or [],
            "message": "\n".join(message_lines),
        }

    async def _notify_blocking(
        self,
        stage: str,
        issues: List[ValidationIssueDetail],
        suggestions: Optional[List[str]] = None,
    ) -> str:
        payload = self._format_issue_payload(stage, issues, suggestions)
        message_type = MessageType["REPLICATE_PREDICTION"]
        await send_data_to_url_async(
            data=payload,
            url=f"{CORE_API_URL}/from-llm",
            crew_input=self.run_input,
            message_type=message_type,
        )
        return payload["message"]

    async def run(self):
        primary_image = (
            self.hitl_edits.get("input_image")
            or self.hitl_edits.get("source_image")
            or self.hitl_edits.get("image")
            or (self.attachments[0] if self.attachments else None)
        )

        information_agent = self.information_agent()
        information = await information_agent.run(
            self.prompt,
            deps=InformationInputPayload(
                example_input=self.example_input,
                description=self.description,
                attached_file=primary_image,
            ),
        )

        if not information.output.continue_run:
            await send_data_to_url_async(
                data=information.output.response_information,
                url=f"{CORE_API_URL}/from-llm",
                crew_input=self.run_input,
                message_type=MessageType["REPLICATE_PREDICTION"],
            )
            return information.output.response_information

        # Stage 1: File requirement detection
        file_requirement_agent = self.file_requirement_agent()
        file_requirement_context = FileRequirementContext(
            prompt=self.prompt,
            example_input=self.example_input,
            model_name=self.model_name,
            model_description=self.description,
            existing_payload=self.example_input if isinstance(self.example_input, dict) else {},
            hitl_edits=self.hitl_edits,
            attachments=self.attachments,
        )
        file_analysis_result = await file_requirement_agent.run(
            "Determine if additional file inputs are required before executing the model.",
            deps=file_requirement_context,
        )
        file_analysis = file_analysis_result.output

        if not file_analysis.ready:
            return await self._notify_blocking(
                "File Requirement Review",
                file_analysis.blocking_issues,
                file_analysis.suggestions,
            )

        replicate_agent = self.replicate_agent()
        replicate_result = await replicate_agent.run(
            "Rewrite the example_input based on the affected properties provided.",
            deps=ExampleInput(
                example_input=self.example_input,
                description=information.output.response_information,
                prompt=self.prompt,
                image_file=primary_image,
                attachments=self.attachments,
                hitl_edits=self.hitl_edits,
            ),
        )

        # Stage 2: Payload validation
        payload_validation_agent = self.payload_validation_agent()
        payload_validation_context = PayloadValidationContext(
            prompt=self.prompt,
            example_input=self.example_input,
            candidate_payload=replicate_result.output,
            required_files=file_analysis.required_files,
            hitl_edits=self.hitl_edits,
            attachments=self.attachments,
            operation_type=self.operation_type,
        )
        payload_validation_result = await payload_validation_agent.run(
            "Validate the candidate payload before execution.",
            deps=payload_validation_context,
        )
        payload_validation_output = payload_validation_result.output

        if not payload_validation_output.ready:
            suggestions = []
            if payload_validation_output.summary:
                suggestions.append(payload_validation_output.summary)
            return await self._notify_blocking(
                "Payload Validation",
                payload_validation_output.blocking_issues,
                suggestions or None,
            )

        validated_payload = payload_validation_output.payload

        # Stage 3: Final guard check
        final_guard_agent = self.final_guard_agent()
        final_guard_context = FinalGuardContext(
            prompt=self.prompt,
            example_input=self.example_input,
            candidate_payload=validated_payload,
            model_name=self.model_name,
            model_description=self.description,
            operation_type=self.operation_type,
        )
        final_guard_result = await final_guard_agent.run(
            "Perform a final schema compliance guard before execution.",
            deps=final_guard_context,
        )
        final_guard_output = final_guard_result.output

        if not final_guard_output.approved:
            suggestions = []
            if final_guard_output.diff_summary:
                suggestions.append(final_guard_output.diff_summary)
            return await self._notify_blocking(
                "Final Guard",
                final_guard_output.blocking_issues,
                suggestions or None,
            )

        approved_payload = final_guard_output.payload

        api_interaction_agent = self.api_interaction_agent()
        api_result = await api_interaction_agent.run(
            "Send the request to replicate.com and receive the response.",
            deps=approved_payload,
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

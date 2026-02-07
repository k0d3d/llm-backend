"""
Replicate provider implementation wrapping existing ReplicateTeam logic
"""

import time
import re
from copy import deepcopy
from typing import Dict, Any, List, Optional

from llm_backend.core.providers.base import (
    AIProvider, ProviderPayload, ProviderResponse, 
    ProviderCapabilities, ValidationIssue, OperationType
)
from llm_backend.tools.replicate_tool import run_replicate
from llm_backend.providers.replicate_error_parser import ReplicateErrorParser


class ReplicatePayload(ProviderPayload):
    """Replicate-specific payload"""
    input: Dict[str, Any]
    operation_type: OperationType
    model_version: str
    webhook_url: str = None


class ReplicateProvider(AIProvider):
    """Provider implementation for Replicate models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.example_input = config.get("example_input", {})
        self.description = config.get("description", "")
        self.latest_version = config.get("latest_version", "")
        self.model_name = config.get("name", "")
        self.field_metadata = self._build_field_metadata()
        self.hitl_alias_metadata = self._build_hitl_alias_metadata(self.field_metadata)
        self._orchestrator = None  # Will be set by orchestrator if using form-based workflow
    
    def _build_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Derive lightweight schema metadata from the example input."""
        metadata: Dict[str, Dict[str, Any]] = {}
        if not isinstance(self.example_input, dict):
            return metadata

        for field, value in self.example_input.items():
            field_meta: Dict[str, Any] = {
                "type": type(value).__name__,
                "collection": isinstance(value, (list, tuple)),
            }

            if isinstance(value, (list, tuple)) and value:
                field_meta["item_type"] = type(value[0]).__name__
            if isinstance(value, dict):
                field_meta["nested_keys"] = list(value.keys())

            metadata[field] = field_meta

        return metadata

    def _build_hitl_alias_metadata(self, field_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create alias metadata that captures collection/dict hints for HITL edits."""
        base_aliases: Dict[str, List[str]] = {
            "input_image": ["input_image", "source_image", "image", "image_url", "image_input"],
            "source_image": ["source_image", "input_image", "image", "image_url"],
            "image": ["image", "input_image", "source_image", "image_url", "image_input"],
            "driven_audio": ["driven_audio", "audio_file", "audio"],
            "audio_file": ["audio_file", "driven_audio", "audio"],
            "prompt": ["prompt", "text", "instruction", "input"],
            "instruction": ["instruction", "prompt", "text"],
        }

        alias_metadata: Dict[str, Dict[str, Any]] = {}

        def collection_hint(target: str) -> bool:
            return field_metadata.get(target, {}).get("collection", False)

        def dict_hint(target: str) -> bool:
            return field_metadata.get(target, {}).get("type") == "dict"

        for alias_key, targets in base_aliases.items():
            alias_metadata[alias_key] = {
                "targets": targets,
                "collection": any(collection_hint(t) for t in targets),
                "dict": any(dict_hint(t) for t in targets),
            }

        for field, meta in field_metadata.items():
            alias_metadata.setdefault(field, {
                "targets": [field],
                "collection": meta.get("collection", False),
                "dict": meta.get("type") == "dict",
            })

            # Provide generic aliases for multi-valued media fields
            if meta.get("collection") and "image" in field.lower():
                alias_metadata.setdefault("image_input", {
                    "targets": [field, "image_input"],
                    "collection": True,
                    "dict": False,
                })

        return alias_metadata

    def _coerce_payload_to_schema(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure payload fields align with the inferred schema metadata."""
        if not payload:
            return payload

        coerced = payload.copy()
        for field, meta in self.field_metadata.items():
            if field not in coerced:
                continue

            value = coerced[field]

            if meta.get("collection") and not isinstance(value, (list, tuple)):
                coerced[field] = [value]
                continue

            if meta.get("type") == "dict" and isinstance(value, dict):
                continue

            if meta.get("type") == "dict" and not isinstance(value, dict):
                coerced[field] = {"value": value}

        return coerced

    def _filter_payload_to_schema(
        self,
        payload: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Strip payload fields that are not present in the example_input schema."""

        if not isinstance(payload, dict):
            return payload

        schema = schema if schema is not None else self.example_input
        if not isinstance(schema, dict):
            return payload

        # Extract example URLs to filter them out from the payload if they are leaking
        example_urls = set()
        for value in schema.values():
            if isinstance(value, str) and value.startswith("http"):
                example_urls.add(value)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, str) and v.startswith("http"):
                        example_urls.add(v)

        filtered: Dict[str, Any] = {}

        for field, schema_value in schema.items():
            if field not in payload:
                continue

            value = payload[field]

            if isinstance(schema_value, dict) and isinstance(value, dict):
                filtered[field] = self._filter_payload_to_schema(value, schema_value)
            elif isinstance(schema_value, list):
                normalized_list = value if isinstance(value, list) else [value]

                # Filter out example URLs from the list
                filtered_list = [v for v in normalized_list if v not in example_urls]

                if schema_value and isinstance(schema_value[0], dict):
                    filtered[field] = [
                        self._filter_payload_to_schema(item, schema_value[0])
                        if isinstance(item, dict)
                        else item
                        for item in filtered_list
                    ]
                else:
                    filtered[field] = filtered_list
            else:
                # Filter out example URLs from single values
                if isinstance(value, str) and value in example_urls:
                    print(f"🧹 _filter_payload_to_schema: Stripped example URL from field '{field}'")
                    continue
                filtered[field] = value

        return filtered
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Return Replicate model capabilities"""
        operation_type = self._infer_operation_type()
        
        return ProviderCapabilities(
            name=self.model_name,
            description=self.description,
            version=self.latest_version,
            input_schema=self.example_input,
            supported_operations=[operation_type],
            safety_features=["content_filter", "nsfw_detection"],
            rate_limits={"requests_per_minute": 60},
            max_input_size=10 * 1024 * 1024,  # 10MB
            cost_per_request=0.01  # Estimate
        )
    
    def _normalize_attachments(self, attachments: List[str]) -> List[str]:
        """Normalize attachment URLs to ensure they're accessible to Replicate"""
        if not attachments:
            return []

        normalized = []
        for url in attachments:
            # Handle local dev server URLs
            if 'serve-dev.tohju.com' in url:
                normalized.append(url)
            # Handle replicate URLs
            elif 'replicate.delivery' in url:
                normalized.append(url)
            # Handle other URLs (could add S3, etc)
            else:
                normalized.append(url)
        return normalized

    def _strip_attachment_mentions(self, prompt: str, attachments: List[str]) -> tuple[str, List[str]]:
        """Remove attachment markers from prompt and extract embedded URLs

        Returns:
            tuple: (cleaned_prompt, extracted_urls)
        """
        if not prompt:
            return prompt, list(attachments) if attachments else []

        clean_prompt = prompt
        extracted_urls = list(attachments) if attachments else []

        # NEW: Extract URLs from ---BEGIN_ATTACHMENTS--- delimiter block
        attachment_block = re.search(r'---BEGIN_ATTACHMENTS---(.*?)---END_ATTACHMENTS---', clean_prompt, re.DOTALL)
        if attachment_block:
            embedded_urls = [url.strip() for url in attachment_block.group(1).strip().split('\n') if url.strip()]
            extracted_urls.extend(embedded_urls)
            clean_prompt = re.sub(r'\n*---BEGIN_ATTACHMENTS---.*?---END_ATTACHMENTS---\n*', '', clean_prompt, re.DOTALL)
            print(f"📦 Extracted {len(embedded_urls)} URLs from delimiter block: {embedded_urls}")

        # LEGACY: Remove "📎 Attachments:" section (Telegram old format)
        legacy_emoji_match = re.search(r'📎\s*Attachments?:(.*?)(?=\n\n|$)', clean_prompt, re.DOTALL | re.IGNORECASE)
        if legacy_emoji_match:
            # Try to extract URLs from legacy format
            legacy_text = legacy_emoji_match.group(1)
            legacy_urls = re.findall(r'https?://[^\s]+', legacy_text)
            if legacy_urls:
                extracted_urls.extend(legacy_urls)
                print(f"📎 Extracted {len(legacy_urls)} URLs from legacy emoji format")
        clean_prompt = re.sub(r'\n\n📎\s*Attachments?:.*$', '', clean_prompt, flags=re.IGNORECASE | re.DOTALL)
        clean_prompt = re.sub(r'📎\s*Attachments?:.*$', '', clean_prompt, flags=re.IGNORECASE | re.DOTALL)

        # LEGACY: Strip ":-> attached document:" markers (PWA old format)
        arrow_matches = re.findall(r':->\\s*attached\\s+document:?\\s*(https?://[^\\s]*)', clean_prompt, re.IGNORECASE)
        if arrow_matches:
            extracted_urls.extend(arrow_matches)
            print(f"📌 Extracted {len(arrow_matches)} URLs from :-> format")
        clean_prompt = re.sub(r'\s*:->\s*attached\s+document:?\s*', '', clean_prompt, flags=re.IGNORECASE)
        clean_prompt = re.sub(r'\n\s*:->\s*attached\s+document:?\s*', '', clean_prompt, flags=re.IGNORECASE)

        # Remove specific attachment URLs from the cleaned prompt
        for url in extracted_urls:
            clean_prompt = clean_prompt.replace(url, '')
            # Also remove markdown-style links
            clean_prompt = re.sub(r'\[.*?\]\([^)]*\)', '', clean_prompt)

        # Remove any remaining Telegram API URLs
        clean_prompt = re.sub(r'https?://api\.telegram\.org/[^\s]*', '', clean_prompt)

        # Clean up orphaned colons
        clean_prompt = re.sub(r':\s*$', '', clean_prompt)

        # Clean up artifacts and excessive whitespace
        clean_prompt = re.sub(r'""', '"', clean_prompt)
        clean_prompt = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_prompt)  # Max 2 newlines
        clean_prompt = re.sub(r' +', ' ', clean_prompt)  # Single spaces only
        clean_prompt = re.sub(r'\n ', '\n', clean_prompt)  # Remove leading spaces after newlines

        # Deduplicate extracted URLs
        extracted_urls = list(dict.fromkeys(extracted_urls))  # Preserve order, remove duplicates

        return clean_prompt.strip(), extracted_urls

    async def _resolve_attachment_conflicts(self, user_attachments: List[str], payload: Dict[str, Any], prompt: str, example_urls: List[str]) -> Dict[str, Any]:
        """Use AI agent to resolve conflicts between user attachments and example URLs"""
        from llm_backend.agents.attachment_resolver import resolve_attachment_conflicts
        
        try:
            result = await resolve_attachment_conflicts(
                user_attachments=user_attachments,
                example_input=self.example_input,
                current_payload=payload,
                prompt=prompt,
                example_urls=example_urls
            )
            
            print(f"🤖 Attachment resolver: {result.reasoning}")
            for change in result.changes_made:
                print(f"🔧 Attachment change: {change}")
            
            return result.resolved_payload
        except Exception as e:
            print(f"⚠️ Attachment resolution failed: {e}")
            # Fallback: manually replace example URLs with user attachments
            return self._manual_attachment_resolution(user_attachments, payload)
    
    def _manual_attachment_resolution(self, user_attachments: List[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Manual fallback for attachment resolution"""
        if not user_attachments:
            return payload

        resolved = payload.copy()
        primary_attachment = user_attachments[0]

        # Replace any example URLs with user's primary attachment
        replaced = False
        for key, value in resolved.items():
            if isinstance(value, str) and "replicate.delivery" in value:
                resolved[key] = primary_attachment
                print(f"🔧 Manual replacement: {key} -> {primary_attachment}")
                replaced = True

        # If no field was replaced, ADD the attachment to a schema field
        if not replaced and primary_attachment:
            attachment_fields = ["input_image", "image", "image_input", "source_image", "image_url"]
            example_input = getattr(self, 'example_input', {}) or {}

            # First try to find a common attachment field that exists in the schema
            for field in attachment_fields:
                if field in example_input:
                    resolved[field] = primary_attachment
                    print(f"🔧 Manual addition (schema match): {field} = {primary_attachment}")
                    replaced = True
                    break

            # Second: look for ANY field in schema that looks like an attachment field
            if not replaced:
                attachment_indicators = ["image", "img", "photo", "picture", "file", "input", "source", "media"]
                for field, value in example_input.items():
                    field_lower = field.lower()
                    if any(ind in field_lower for ind in attachment_indicators):
                        resolved[field] = primary_attachment
                        print(f"🔧 Manual addition (schema heuristic): {field} = {primary_attachment}")
                        replaced = True
                        break

            # Third: look for any field that has a URL value in schema (likely an attachment)
            if not replaced:
                for field, value in example_input.items():
                    if isinstance(value, str) and value.startswith("http"):
                        resolved[field] = primary_attachment
                        print(f"🔧 Manual addition (URL field): {field} = {primary_attachment}")
                        replaced = True
                        break

            # Last resort: if we still couldn't find a match, log warning
            if not replaced:
                print(f"⚠️ Manual fallback: Could not find attachment field in schema. Schema fields: {list(example_input.keys())}")

        return resolved

    def set_orchestrator(self, orchestrator):
        """Set the orchestrator reference to access form data"""
        self._orchestrator = orchestrator
        print(f"📋 Orchestrator linked to provider for form-based payload creation")

    def _map_form_fields_to_api_fields(self, form_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministically map form field names to API field names.

        Uses hitl_alias_metadata and example_input to find correct mappings.
        This replaces the AI agent for form-based workflows.
        """
        api_payload = {}

        print(f"🗺️ Mapping {len(form_values)} form fields to API fields")

        for form_field, form_value in form_values.items():
            # Default: try to find mapping in example_input or use same name
            api_field = None

            # Strategy 1: Check if form field exists directly in example_input
            if form_field in self.example_input:
                api_field = form_field
                print(f"   ✅ Direct match: '{form_field}' → '{api_field}'")

            # Strategy 2: Check hitl_alias_metadata for reverse mapping
            if not api_field and self.hitl_alias_metadata:
                for api_name, aliases in self.hitl_alias_metadata.items():
                    if form_field in aliases:
                        api_field = api_name
                        print(f"   ✅ Alias match: '{form_field}' → '{api_field}' (via {api_name})")
                        break

            # Strategy 3: Common pattern mappings
            if not api_field:
                # prompt → input (most common text field mapping)
                if form_field == "prompt" and "input" in self.example_input:
                    api_field = "input"
                    print(f"   ✅ Pattern match: 'prompt' → 'input'")
                # image_input → file_input (common image field mapping)
                elif form_field == "image_input" and "file_input" in self.example_input:
                    api_field = "file_input"
                    print(f"   ✅ Pattern match: 'image_input' → 'file_input'")
                # image → input_image (another common pattern)
                elif form_field == "image" and "input_image" in self.example_input:
                    api_field = "input_image"
                    print(f"   ✅ Pattern match: 'image' → 'input_image'")
                # Fallback: keep original name
                else:
                    api_field = form_field
                    print(f"   ⚠️ No mapping found, using original: '{form_field}'")

            api_payload[api_field] = form_value

        return api_payload

    async def _create_payload_from_form(self, form_data: Dict[str, Any], operation_type: OperationType, config: Dict) -> ReplicatePayload:
        """Create payload directly from form data using deterministic field mapping"""
        current_values = form_data.get("current_values", {})

        print(f"📋 Creating payload from form with {len(current_values)} fields")

        # Clean prompt field if it exists - remove attachment mentions
        if "prompt" in current_values and isinstance(current_values["prompt"], str):
            # Get list of attachments from form data or orchestrator state
            attachments = []
            if hasattr(self, '_orchestrator') and self._orchestrator:
                attachments = getattr(self._orchestrator.state, 'attachments', [])

            original_prompt = current_values["prompt"]
            cleaned_prompt, extracted_attachments = self._strip_attachment_mentions(original_prompt, attachments)

            if cleaned_prompt != original_prompt:
                print(f"🧹 Cleaned prompt: '{original_prompt[:50]}...' -> '{cleaned_prompt[:50]}...'")
                print(f"📎 Extracted {len(extracted_attachments)} total attachments")
                current_values = {**current_values, "prompt": cleaned_prompt}

            # Update attachments list with extracted URLs
            if extracted_attachments:
                attachments = extracted_attachments

        # NEW: Use deterministic field name mapping
        payload_input = self._map_form_fields_to_api_fields(current_values)

        # Apply schema coercion to ensure proper types
        payload_input = self._coerce_payload_to_schema(payload_input)

        # Filter to only include fields from example_input schema
        payload_input = self._filter_payload_to_schema(payload_input)

        print(f"📦 Final payload input: {list(payload_input.keys())}")

        # Build webhook URL - handle None case
        webhook_url = config.get("webhook_url")
        if webhook_url is None:
            webhook_url = ""  # Use empty string instead of None

        return ReplicatePayload(
            input=payload_input,
            operation_type=operation_type,
            model_version=self.latest_version,
            webhook_url=webhook_url,
            provider_name=self.model_name
        )

    async def create_payload(self, prompt: str, attachments: List[str], operation_type: OperationType, config: Dict, conversation: Optional[List[Dict[str, str]]] = None, hitl_edits: Dict = None) -> ReplicatePayload:
        """Create Replicate-specific payload using intelligent Pydantic AI agent"""
        from llm_backend.agents.replicate_team import ReplicateTeam
        from llm_backend.core.types.replicate import ExampleInput
        import asyncio

        if self.run_input is None:
            raise ValueError("ReplicateProvider requires run_input to be set before creating payloads")

        print("🤖 Using intelligent Pydantic AI agent for payload creation")
        print(f"📥 Raw attachments received: {attachments}")
        print(f"📝 Original prompt: '{prompt[:100]}...'")

        normalized_attachments = self._normalize_attachments(attachments or [])
        print(f"🔄 Normalized attachments: {normalized_attachments}")

        # Extract example URLs to filter them out from user attachments
        example_urls = set()
        if isinstance(self.example_input, dict):
            for value in self.example_input.values():
                if isinstance(value, str) and value.startswith("http"):
                    example_urls.add(value)
        print(f"📋 Example URLs from schema: {example_urls}")

        # Filter out example URLs to get actual user attachments
        actual_user_attachments = [url for url in normalized_attachments if url not in example_urls]
        print(f"✅ Actual user attachments (after filtering): {actual_user_attachments}")

        # Extract attachments from prompt and clean it
        clean_prompt, extracted_attachments = self._strip_attachment_mentions(prompt, normalized_attachments)
        print(f"🧹 Cleaned prompt: '{clean_prompt}'")
        if clean_prompt != prompt:
            print(f"   ✂️ Removed {len(prompt) - len(clean_prompt)} characters from prompt")

        # Update attachments list with extracted URLs
        if extracted_attachments:
            normalized_attachments = extracted_attachments
            print(f"📎 Using {len(extracted_attachments)} extracted attachments")

            # Re-filter to get actual user attachments after extraction
            actual_user_attachments = [url for url in normalized_attachments if url not in example_urls]
            print(f"✅ Actual user attachments after extraction: {actual_user_attachments}")

        # Use actual user attachment as primary, fallback to first if none found
        primary_attachment = (actual_user_attachments[0] if actual_user_attachments
                            else None)
        print(f"🎯 Primary attachment selected: {primary_attachment}")

        example_input_data = deepcopy(self.example_input) if isinstance(self.example_input, dict) else {}

        # Strip example URLs from example_input_data to avoid leaking them into payload
        if isinstance(example_input_data, dict):
            for field, value in list(example_input_data.items()):
                if isinstance(value, str) and value in example_urls:
                    example_input_data[field] = None
                    print(f"🧹 Stripped example URL from field '{field}'")
                elif isinstance(value, list):
                    filtered_list = [v for v in value if v not in example_urls]
                    if filtered_list != value:
                        example_input_data[field] = filtered_list
                        print(f"🧹 Stripped example URLs from list field '{field}'")
        if primary_attachment:
            image_fields = [
                "input_image",
                "image",
                "image_input",
                "source_image",
                "image_url",
            ]

            assigned = False
            for field in image_fields:
                if field in example_input_data:
                    old_value = example_input_data[field]
                    example_input_data[field] = primary_attachment
                    assigned = True
                    print(f"   ✅ Assigned '{field}': {old_value} -> {primary_attachment}")
                    break

            if not assigned:
                example_input_data.setdefault("input_image", primary_attachment)
                print(f"   ⚠️ No existing image field found, created 'input_image': {primary_attachment}")
        else:
            print(f"⚠️ No primary_attachment to assign")

        replicate_team = ReplicateTeam(
            prompt=clean_prompt,
            tool_config={
                "example_input": example_input_data,
                "description": self.description,
                "latest_version": self.latest_version,
                "model_name": self.model_name,
                "field_metadata": self.field_metadata,
                "hitl_alias_metadata": self.hitl_alias_metadata,
            },
            run_input=self.run_input,
            hitl_enabled=True  # Enable intelligent mapping
        )

        agent_input = ExampleInput(
            prompt=clean_prompt,
            example_input=example_input_data,
            description=self.description,
            attachments=normalized_attachments or None,
            image_file=primary_attachment,
            hitl_edits=hitl_edits,
            schema_metadata=self.field_metadata,
            hitl_field_metadata=self.hitl_alias_metadata,
            structured_form_values=None,  # Not used - form data now uses deterministic mapping
            conversation=conversation,  # Chat history for understanding references to previous prompts/settings
        )

        try:
            replicate_agent = replicate_team.replicate_agent()

            async def run_agent() -> Any:
                return await replicate_agent.run(
                    "Generate an optimal Replicate payload using the provided inputs.",
                    deps=agent_input,
                )

            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(run_agent()))
                    agent_payload = future.result()
            else:
                agent_payload = asyncio.run(run_agent())

            agent_output = getattr(agent_payload, "output", None)
            if agent_output is None:
                raise ValueError("Agent did not return an output payload")

            agent_input_payload = agent_output.input.model_dump()
            
            # Set initial prompt if needed
            prompt_targets = ["prompt", "text", "input", "instruction", "query"]
            for target in prompt_targets:
                if target in agent_input_payload or target in self.example_input:
                    agent_input_payload[target] = clean_prompt
                    break

            # Apply HITL edits after setting initial values
            if hitl_edits:
                agent_input_payload = self._apply_hitl_edits(agent_input_payload, hitl_edits)

            # Extract example URLs to filter them out from user attachments
            example_urls = set()
            if isinstance(self.example_input, dict):
                for value in self.example_input.values():
                    if isinstance(value, str) and value.startswith("http"):
                        example_urls.add(value)

            # Resolve attachment conflicts using AI agent
            if actual_user_attachments:
                agent_input_payload = await self._resolve_attachment_conflicts(
                    actual_user_attachments, agent_input_payload, clean_prompt, list(example_urls)
                )

            # Final safety net: ensure attachment is added if still missing
            if primary_attachment:
                attachment_fields = [
                    "input_image",
                    "image",
                    "image_input",
                    "source_image",
                    "image_url",
                ]
                if not any(agent_input_payload.get(field) for field in attachment_fields):
                    # Find an attachment field that exists in the schema
                    example_input = self.example_input or {}
                    added = False
                    for field in attachment_fields:
                        if field in example_input:
                            agent_input_payload[field] = primary_attachment
                            print(f"🔧 Safety net: added {field} = {primary_attachment}")
                            added = True
                            break
                    # Fallback: look for any field with URL value in schema
                    if not added:
                        for field, value in example_input.items():
                            if isinstance(value, str) and value.startswith("http"):
                                agent_input_payload[field] = primary_attachment
                                print(f"🔧 Safety net (URL match): added {field} = {primary_attachment}")
                                added = True
                                break
                    if not added:
                        print(f"⚠️ Safety net: could not find schema field for attachment")

            agent_input_payload = self._filter_payload_to_schema(agent_input_payload)

            print(f"🤖 Agent created payload: {agent_input_payload}")

            metadata = {
                "original_example": self.example_input,
                "agent_generated": True,
                "agent_messages": getattr(agent_payload, "messages", None),
                "hitl_applied": bool(hitl_edits),
                "schema_metadata": self.field_metadata,
            }

            return ReplicatePayload(
                provider_name="replicate",
                input=self._coerce_payload_to_schema(agent_input_payload),
                operation_type=operation_type,
                model_version=self.latest_version,
                metadata=metadata,
            )

        except Exception as e:
            print(f"⚠️ Agent payload creation failed: {e}")
            print(f"🔍 Agent input details: prompt='{prompt}', hitl_edits={hitl_edits}, example_input={self.example_input}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            print("🔧 Falling back to static mapping")

            # Start with example input and set initial prompt
            fallback_input = deepcopy(example_input_data)
            prompt_targets = ["prompt", "text", "input", "instruction", "query"]
            for target in prompt_targets:
                if target in fallback_input:
                    fallback_input[target] = clean_prompt
                    break

            # Apply HITL edits after setting initial values
            if hitl_edits:
                fallback_input = self._apply_hitl_edits(fallback_input, hitl_edits)

            # Handle attachments in fallback
            if primary_attachment:
                fallback_attachment_fields = [
                    "image", "image_url", "input_image", "image_input",
                    "source_image", "first_frame_image", "subject_reference",
                    "start_image", "init_image"
                ]
                for field in fallback_attachment_fields:
                    if field in fallback_input:
                        fallback_input[field] = primary_attachment
                        break

            fallback_input = self._filter_payload_to_schema(fallback_input)

        return ReplicatePayload(
            provider_name="replicate",
            input=self._coerce_payload_to_schema(fallback_input),
            operation_type=operation_type,
            model_version=self.latest_version,
            metadata={
                "original_example": self.example_input,
                "fallback_used": True,
                "hitl_mapped": bool(hitl_edits),
            },
        )

    def _apply_hitl_edits(self, base_input: Dict[str, Any], hitl_edits: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply human edits onto a payload input dict with smart alias handling."""
        if not hitl_edits:
            return base_input

        updated_input = base_input.copy()

        example_input = getattr(self, "example_input", {}) or {}
        alias_metadata = self.hitl_alias_metadata or {}

        def _coerce_collection(target_key: str, new_value: Any) -> Any:
            """Ensure new_value matches the collection type of the target."""
            current_value = updated_input.get(target_key, example_input.get(target_key))

            if isinstance(current_value, (list, tuple)):
                if isinstance(new_value, (list, tuple)):
                    return list(new_value)
                return [new_value]

            if isinstance(current_value, dict) and isinstance(new_value, dict):
                merged = current_value.copy()
                merged.update(new_value)
                return merged

            return new_value

        for key, value in hitl_edits.items():
            if value in (None, ""):
                continue

            alias_info = alias_metadata.get(key, {"targets": [key], "collection": False, "dict": False})
            targets = alias_info.get("targets", [key])
            applied = False

            for target in targets:
                if target in updated_input or target in example_input:
                    updated_input[target] = _coerce_collection(target, value)
                    applied = True

            if not applied and targets:
                primary_target = targets[0]
                updated_input[primary_target] = _coerce_collection(primary_target, value)

            print(f"🔧 Applied HITL overlay: {key} -> {value}")

        return updated_input
    
    def validate_payload(self, payload: ReplicatePayload, prompt: str, attachments: List[str]) -> List[ValidationIssue]:
        """
        Simplified validation that trusts form field classification.
        Also verifies attachment counts.
        """
        issues = []

        # Count expected vs actual images in payload
        # Extract example URLs to filter them out
        example_urls = set()
        if isinstance(self.example_input, dict):
            for value in self.example_input.values():
                if isinstance(value, str) and value.startswith("http"):
                    example_urls.add(value)
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, str) and v.startswith("http"):
                            example_urls.add(v)

        image_fields = ["input_image", "image", "image_input", "source_image", "image_url"]
        payload_attachments = []
        for field in image_fields:
            val = payload.input.get(field)
            if isinstance(val, list):
                payload_attachments.extend([v for v in val if isinstance(v, str) and v.startswith("http") and v not in example_urls])
            elif isinstance(val, str) and val.startswith("http") and val not in example_urls:
                payload_attachments.append(val)
        
        # Deduplicate
        payload_attachments = list(dict.fromkeys(payload_attachments))
        
        if len(payload_attachments) != len(attachments):
            issues.append(ValidationIssue(
                field="attachments",
                issue=f"Attachment count mismatch: payload has {len(payload_attachments)}, but request has {len(attachments)}",
                severity="warning",
                suggested_fix=f"Ensure all {len(attachments)} attachments are mapped to appropriate fields",
                auto_fixable=False
            ))

        # Check for empty values in fields that are present in the payload
        # (Form classifier already determined which fields are required)
        for field_name, field_value in payload.input.items():
            if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
                issues.append(ValidationIssue(
                    field=field_name,
                    issue=f"Field '{field_name}' is empty",
                    severity="warning",  # Warning, not error - let Replicate API decide
                    suggested_fix=f"Provide a value for {field_name}",
                    auto_fixable=False
                ))
            elif isinstance(field_value, list) and len(field_value) == 0:
                # Empty arrays might be intentional (optional fields)
                issues.append(ValidationIssue(
                    field=field_name,
                    issue=f"Field '{field_name}' is an empty array",
                    severity="info",  # Just informational
                    suggested_fix=f"Add items to {field_name} if needed",
                    auto_fixable=False
                ))

        return issues

    def _infer_field_type(self, key: str, value: any) -> str:
        """Infer field type from key name and value"""
        key_lower = key.lower()
        if any(word in key_lower for word in ["image", "img", "photo"]):
            return "image"
        elif any(word in key_lower for word in ["audio", "sound", "speech"]):
            return "audio"
        elif any(word in key_lower for word in ["video", "clip"]):
            return "video"
        elif any(word in key_lower for word in ["prompt", "text", "instruction"]):
            return "text"
        elif isinstance(value, (int, float)):
            return "number"
        elif isinstance(value, bool):
            return "boolean"
        else:
            return "text"
    
    def _get_field_description(self, key: str) -> str:
        """Get human-readable description for field"""
        descriptions = {
            "prompt": "Describe what you want to create or do",
            "text": "Enter the text content",
            "instruction": "Provide instructions for the AI",
            "image": "Upload an image file",
            "input_image": "Upload the source image",
            "audio": "Upload an audio file",
            "file": "Upload a file",
            "strength": "Adjustment strength (0.0 to 1.0)",
            "steps": "Number of processing steps",
            "guidance_scale": "How closely to follow the prompt"
        }
        return descriptions.get(key.lower(), f"Set the {key} parameter")
    
    def execute(self, payload: ReplicatePayload) -> ProviderResponse:
        """Execute Replicate model request"""
        start_time = time.time()

        # Prepare model parameters for execution
        # We need to ensure example_input URLs are stripped here too
        exec_example_input = deepcopy(self.example_input)
        if isinstance(exec_example_input, dict):
            # Extract example URLs to filter them out
            example_urls = set()
            for value in exec_example_input.values():
                if isinstance(value, str) and value.startswith("http"):
                    example_urls.add(value)
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, str) and v.startswith("http"):
                            example_urls.add(v)
            
            # Strip them
            for field, value in list(exec_example_input.items()):
                if isinstance(value, str) and value in example_urls:
                    exec_example_input[field] = None
                    print(f"🧹 execute: Stripped example URL from field '{field}'")
                elif isinstance(value, list):
                    filtered_list = [v for v in value if v not in example_urls]
                    if filtered_list != value:
                        exec_example_input[field] = filtered_list
                        print(f"🧹 execute: Stripped example URLs from list field '{field}'")

        try:
            # Use existing run_replicate function
            run, status_code = run_replicate(
                run_input=self.run_input,
                model_params={
                    "example_input": exec_example_input,
                    "latest_version": self.latest_version,
                },
                input=payload.input,
                operation_type=payload.operation_type.value,
            )

            execution_time = int((time.time() - start_time) * 1000)

            # Check for errors in response
            if isinstance(run, dict) and run.get("error"):
                print(f"❌ Replicate API error: {run.get('error_message')}")

                # Parse error into structured format
                error_details = ReplicateErrorParser.parse_error(
                    status_code=run.get("status_code", 500),
                    error_data=run.get("raw_error", {})
                )

                return ProviderResponse(
                    raw_response=run,
                    processed_response="",
                    metadata={
                        "error_details": error_details,
                        "recoverable": error_details["recoverable"],
                        "execution_time_ms": execution_time
                    },
                    execution_time_ms=execution_time,
                    status_code=run.get("status_code"),
                    error=error_details["message"]
                )

            # Success path
            return ProviderResponse(
                raw_response=run,
                processed_response=str(run),
                metadata={
                    "model_version": payload.model_version,
                    "operation_type": payload.operation_type.value,
                    "model_name": self.model_name,
                    "execution_time_ms": execution_time
                },
                execution_time_ms=execution_time,
                status_code=status_code
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            print(f"❌ Unexpected error in execute: {e}")
            return ProviderResponse(
                raw_response=None,
                processed_response="",
                metadata={
                    "error_details": {
                        "error_type": "unknown",
                        "recoverable": False,
                        "message": str(e)
                    },
                    "execution_time_ms": execution_time
                },
                execution_time_ms=execution_time,
                error=str(e),
                status_code=500
            )
    
    def audit_response(self, response: ProviderResponse) -> str:
        """Clean and audit Replicate response for user consumption"""
        if response.error:
            return f"Error: {response.error}"
        
        # Start with processed response
        cleaned_response = str(response.processed_response)
        
        # Remove provider-specific information
        cleaned_response = re.sub(r'https?://replicate\.com[^\s]*', '', cleaned_response)
        cleaned_response = re.sub(r'https?://[^\s]*\.replicate\.delivery[^\s]*', '[Generated Content]', cleaned_response)
        
        # Remove technical metadata
        cleaned_response = re.sub(r'\b(prediction|replicate|model)\b', '', cleaned_response, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()
        
        return cleaned_response
    
    def estimate_cost(self, payload: ReplicatePayload) -> float:
        """Estimate cost of executing this payload"""
        base_cost = 0.01  # $0.01 base
        
        # Adjust based on operation type
        if payload.operation_type == OperationType.IMAGE_GENERATION:
            return base_cost * 2
        elif payload.operation_type == OperationType.VIDEO_GENERATION:
            return base_cost * 10
        elif payload.operation_type == OperationType.AUDIO_GENERATION:
            return base_cost * 3
        else:
            return base_cost
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for Replicate"""
        # This would ideally query Replicate API for actual limits
        return {
            "requests_remaining": 100,  # Placeholder
            "reset_time": time.time() + 3600,
            "current_usage": 0,
            "limit_per_minute": 60
        }
    
    def _infer_operation_type(self) -> OperationType:
        """Infer operation type from description and example input"""
        description_lower = self.description.lower()
        
        # Check description for keywords
        if any(word in description_lower for word in ["image", "picture", "photo", "visual", "generate image"]):
            if any(word in description_lower for word in ["edit", "modify", "change", "inpaint"]):
                return OperationType.IMAGE_EDITING
            return OperationType.IMAGE_GENERATION
        elif any(word in description_lower for word in ["video", "movie", "animation", "clip"]):
            return OperationType.VIDEO_GENERATION
        elif any(word in description_lower for word in ["audio", "sound", "music", "speech", "voice"]):
            return OperationType.AUDIO_GENERATION
        elif any(word in description_lower for word in ["embed", "vector", "similarity"]):
            return OperationType.EMBEDDING
        elif any(word in description_lower for word in ["classify", "category", "label"]):
            return OperationType.CLASSIFICATION
        else:
            return OperationType.TEXT_GENERATION
    
    def supports_streaming(self) -> bool:
        """Replicate doesn't typically support streaming"""
        return False
    
    def supports_cancellation(self) -> bool:
        """Replicate supports prediction cancellation"""
        return True

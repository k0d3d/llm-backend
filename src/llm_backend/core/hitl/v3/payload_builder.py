from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from .schema_extractor import ModelSchema
from .context_assembler import RequestContext
import json

class CandidatePayload(BaseModel):
    parameters: Dict[str, Any] = Field(..., description="The constructed API parameters")
    reasoning: str = Field(..., description="Explanation of the construction logic")

    @model_validator(mode='before')
    @classmethod
    def handle_hallucinations(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. Map common variations back to 'parameters'
            for alt in ['payload', 'input', 'data', 'fields', 'constructed_payload']:
                if alt in data and 'parameters' not in data:
                    data['parameters'] = data.pop(alt)
            
            # 2. If 'parameters' is still missing, check for root-level fields
            if 'parameters' not in data:
                # Assume anything that isn't 'reasoning' is a parameter
                reasoning = data.get('reasoning', "No reasoning provided.")
                params = {k: v for k, v in data.items() if k != 'reasoning'}
                data = {'parameters': params, 'reasoning': reasoning}
            
            # 3. INTERNAL ALIAS MAPPING: Fix field names inside parameters
            # Map common LLM hallucinations back to standard schema names
            params = data.get('parameters', {})
            if isinstance(params, dict):
                field_aliases = {
                    "image_input": ["image", "input_image", "source_image", "image_url"],
                    "prompt": ["input", "text", "instruction", "query"],
                    "audio_file": ["audio", "driven_audio", "voice"]
                }
                
                # Check for each known alias group
                for canonical, alts in field_aliases.items():
                    # If canonical is missing, but an alias exists, rename it
                    if canonical not in params:
                        for alt in alts:
                            if alt in params:
                                params[canonical] = params.pop(alt)
                                break
        return data

class PayloadBuilder:
    """
    AI Agent that constructs the payload based strictly on the schema and user context.
    Zero-Trust: It does NOT see the original example_input values.
    """
    
    @staticmethod
    def create_agent() -> Agent:
        return Agent(
            "openai:gpt-4o-mini",
            result_type=CandidatePayload,
            retries=3,
            system_prompt="""
            You are a strict API Parameter Constructor.
            Your goal is to map a User's Request (Natural Language + Attachments) into a JSON object that matches a specific Schema.
            
            RULES:
            1. **Schema Adherence**: Use ONLY fields defined in the schema.
            2. **Partial Fulfillment**: Fill every field you can identify from the User Request. If a required field is missing (like an image), still fill the other fields (like the prompt).
            3. **Prompt Mapping**: The 'User Prompt' in the context is the primary instruction. Map it to the schema's 'CONTENT' field that represents the prompt, input, or text.
            4. **Zero Hallucination**: If a value is absolutely not in the User Request and has no default, leave it null.
            5. **Attachment Mapping**: Map media URLs to appropriate fields (e.g. image_input, audio_file).
            6. **Explicit Edits**: Values in 'explicit_edits' are AUTHORITATIVE. Use them exactly.
            7. **Config vs Content**: 
               - Config (numbers/toggles): Use schema defaults unless the user requests changes.
               - Content (prompts/images): MUST come from the User Request. NEVER use demo values.
            
            OUTPUT REQUIREMENT:
            Return a JSON object with:
            - "parameters": The dictionary of API fields. MUST include the prompt/text if provided.
            - "reasoning": Your explanation.
            """
        )

    @staticmethod
    async def build(context: RequestContext, schema: ModelSchema) -> CandidatePayload:
        agent = PayloadBuilder.create_agent()
        
        # Prepare the input for the agent
        schema_desc = {
            "fields": {
                name: {
                    "type": f.type.value,
                    "description": f.description or "",
                    "default": f.default_value,
                    "is_content_field": f.is_content,
                    "is_required": f.is_required
                }
                for name, f in schema.fields.items()
            },
            "description": schema.description
        }
        
        prompt = f"""
        TASK: 
        1. Read the 'USER REQUEST CONTEXT' below.
        2. Identify the 'USER PROMPT' (the text instruction).
        3. Map that instruction to the appropriate field in the 'SCHEMA DEFINITION' (usually named 'prompt', 'text', or 'input').
        4. Check for any attachments (URLs) and map them to media fields (like 'image', 'audio', 'image_input').
        5. If a field is 'REQUIRED' but you cannot find a value, leave it null (do NOT invent URLs).
        
        USER REQUEST CONTEXT:
        {context.get_llm_view()}
        
        SCHEMA DEFINITION:
        {json.dumps(schema_desc, indent=2)}
        
        Construct the 'parameters' dictionary and provide your reasoning.
        """
        
        result = await agent.run(prompt)
        print(f"🤖 Raw LLM Response: {result.data.model_dump_json()}")
        return result.data
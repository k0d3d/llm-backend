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
            2. **Zero Hallucination**: If a value is not in the User Request and has no default, leave it null.
            3. **Attachment Mapping**: Map media URLs to appropriate fields (e.g. image_input, audio_file).
            4. **Explicit Edits**: Values in 'explicit_edits' are AUTHORITATIVE. Use them exactly.
            5. **Config vs Content**: 
               - Config (numbers/toggles): Use schema defaults unless the user requests changes.
               - Content (prompts/images): MUST come from the User Request. NEVER use demo values.
            
            OUTPUT REQUIREMENT:
            Return a JSON object with:
            - "parameters": The dictionary of API fields.
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
                    "is_content_field": f.is_content
                }
                for name, f in schema.fields.items()
            },
            "description": schema.description
        }
        
        prompt = f"""
        TASK: Construct the 'parameters' for the API call based on the schema below.
        
        SCHEMA DEFINITION:
        {json.dumps(schema_desc, indent=2)}
        
        USER REQUEST CONTEXT:
        {context.get_llm_view()}
        
        Construct the response now.
        """
        
        result = await agent.run(prompt)
        print(f"🤖 Raw LLM Response: {result.data.model_dump_json()}")
        return result.data
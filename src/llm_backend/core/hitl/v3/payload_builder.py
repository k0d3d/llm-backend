from typing import Dict, Any, Optional
from pydantic import BaseModel
from pydantic_ai import Agent
from .schema_extractor import ModelSchema
from .context_assembler import RequestContext
import json

class CandidatePayload(BaseModel):
    payload: Dict[str, Any]
    reasoning: str

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
            You are a strict API Payload Constructor.
            Your goal is to map a User's Request (Natural Language + Attachments) into a JSON object that matches a specific Schema.
            
            RULES:
            1. **Schema Adherence**: The user provides a Schema Definition (field names, types, descriptions). You must ONLY use fields defined in the schema.
            2. **Zero Hallucination**: If the user has not specified a value for a field, and the schema has no default, leave it null/None. Do NOT invent values.
            3. **Attachment Mapping**: If the schema expects an image/audio/video, and the context contains an attachment URL, map it to the most likely field.
            4. **Explicit Edits**: If the context contains 'explicit_edits', those values are AUTHORITATIVE. Use them exactly.
            5. **Config vs Content**: 
               - For 'Config' fields (sliders, numbers), use the schema's default unless the user explicitly asks to change it (e.g., "set width to 512").
               - For 'Content' fields (prompts, images), you MUST get the value from the User Request. Do not use defaults for content.
            
            CRITICAL OUTPUT REQUIREMENT:
            You MUST return a JSON object with EXACTLY these keys:
            - "payload": A dictionary containing the constructed payload fields. THIS IS MANDATORY.
            - "reasoning": A string explaining your decisions.
            
            Example:
            {
                "payload": {"prompt": "a dog", "width": 512},
                "reasoning": "Mapped user prompt to 'prompt' and used default width."
            }
            """
        )

    @staticmethod
    async def build(context: RequestContext, schema: ModelSchema) -> CandidatePayload:
        agent = PayloadBuilder.create_agent()
        
        # Prepare the input for the agent
        # We convert schema to a simplified dict representation for the prompt
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
        SCHEMA DEFINITION:
        {json.dumps(schema_desc, indent=2)}
        
        USER REQUEST CONTEXT:
        {context.get_llm_view()}
        
        Construct the payload now.
        """
        
        result = await agent.run(prompt)
        return result.data

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class RequestContext(BaseModel):
    user_prompt: str
    conversation_history: List[Dict[str, Any]] = []
    attachments: List[str] = []
    user_id: Optional[str] = None
    explicit_edits: Dict[str, Any] = {}
    
    def get_llm_view(self) -> str:
        """Returns a formatted string of the context for the LLM"""
        parts = []
        if self.user_prompt:
            parts.append(f"USER PROMPT: {self.user_prompt}")
        
        if self.attachments:
            parts.append(f"ATTACHMENTS: {', '.join(self.attachments)}")
            
        if self.conversation_history:
            parts.append("RECENT CONVERSATION:")
            # Take last 5 relevant messages
            for msg in self.conversation_history[-5:]:
                role = msg.get("role", "unknown")
                content = msg.get("content") or msg.get("message") or ""
                if content:
                    parts.append(f"- {role.upper()}: {content[:200]}...") # Truncate for brevity
        
        if self.explicit_edits:
            parts.append(f"USER EXPLICIT OVERRIDES: {self.explicit_edits}")
            
        return "

".join(parts)

class ContextAssembler:
    @staticmethod
    def build(run_input: Any, hitl_state_edits: Dict[str, Any] = {}) -> RequestContext:
        """
        Assemble all user context into a single bundle.
        Handles RunInput as object or dict.
        """
        # Normalize run_input
        ri_dict = run_input.dict() if hasattr(run_input, "dict") else (
            run_input.model_dump() if hasattr(run_input, "model_dump") else 
            (run_input if isinstance(run_input, dict) else {})
        )
        
        prompt = ri_dict.get("prompt", "")
        conversation = ri_dict.get("conversation", []) or []
        user_id = ri_dict.get("user_id")
        
        # Attachments: Check multiple places
        attachments = []
        
        # 1. Direct list
        if ri_dict.get("attachments"):
            attachments.extend(ri_dict["attachments"])
            
        # 2. Extract from prompt (simple heuristic)
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', prompt)
        for url in urls:
            if url not in attachments:
                attachments.append(url)
        
        return RequestContext(
            user_prompt=prompt,
            conversation_history=conversation,
            attachments=attachments,
            user_id=user_id,
            explicit_edits=hitl_state_edits
        )

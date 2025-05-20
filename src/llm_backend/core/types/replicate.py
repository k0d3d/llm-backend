from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Dict


class Props(BaseModel):
    all_props: List[str]
    affected_props: List[str]

class PayloadInput(BaseModel):
    # allow any property
    model_config = ConfigDict(extra="allow")


class AgentPayload(BaseModel):
    input: PayloadInput

    @field_validator('input')
    @classmethod
    def input_not_empty(cls, v: Dict) -> Dict:
        if not v:
            raise ValueError('input dictionary cannot be empty')
        return v


class ExampleInput(BaseModel):
    prompt: str
    example_input: dict
    description: str
    props: Props

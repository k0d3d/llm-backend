from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Dict, Literal
from typing import Optional


class Props(BaseModel):
    all_props: List[str]
    affected_props: List[str]

class PayloadInput(BaseModel):
    # allow any property
    model_config = ConfigDict(extra="allow")

class OperationType(BaseModel):
    type: Literal['image', 'video', 'text', 'audio']


class AgentPayload(BaseModel):
    input: PayloadInput
    operationType: OperationType
    
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
    props: Optional[Props] = None
    image_file: Optional[str] = None
    video_file: Optional[str] = None

class InformationInputResponse(BaseModel):
    continue_run: bool
    response_information: str

class InformationInputPayload(BaseModel):
    example_input: dict
    description: str
    attached_file: Optional[str] = None

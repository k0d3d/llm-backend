from pydantic import BaseModel
from typing import List

class Props(BaseModel):
    all_props: List[str]
    affected_props: List[str]


class PayloadInput(BaseModel):
    input: str


class ExampleInput(BaseModel):
    prompt: str
    example_input: dict
    description: str
    props: Props

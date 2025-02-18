from typing import List, Union, Optional
from pydantic import BaseModel, Field
import datetime as dt
import json
from pydantic import BaseModel, field_validator

from llmvm.common.objects import Message


class CommentaryPair(BaseModel, arbitrary_types_allowed=True):
    a: Message
    b: Message
    commentary: str

    @field_validator('a', 'b', mode='before')
    @classmethod
    def convert_to_message(cls, value):
        if isinstance(value, dict):
            return Message.from_json(value)
        elif isinstance(value, str):
            return Message.from_json(json.loads(value))
        raise ValueError(f"Invalid type for 'a' or 'b': {type(value)}")

    class Config:
        json_encoders = {
            Message: lambda u: u.to_json()
        }


class Comparison(BaseModel, arbitrary_types_allowed=True):
    date_time: str = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    test_name: str
    commentaries: list[CommentaryPair]
    model_a: str
    model_b: str
    model_a_command: Optional[str]
    model_b_command: Optional[str]


class ModelOutput(BaseModel):
    """Represents a single output from a model."""
    content: Union[str, List[str]] = Field(..., description="The output text or list of text segments")
    model_id: str = Field(..., description="Identifier for the model (e.g., 'A', 'B')")
    role: str = Field(..., description="The role of the output (e.g., 'assistant', 'user')")
    output_id: Optional[str] = Field(None, description="Optional identifier for this specific output")

class ComparisonPair(BaseModel):
    """Represents a pair of outputs to be compared."""
    output_a: ModelOutput = Field(..., description="Output from model A")
    output_b: ModelOutput = Field(..., description="Output from model B")
    commentary: Optional[str] = Field(None, description="Optional commentary about the pair")

    def to_dict(self):
        return {
            "output_a": self.output_a.model_dump(),
            "output_b": self.output_b.model_dump()
        }

class ComparisonSet(BaseModel):
    """Represents a complete set of comparison pairs."""
    pairs: List[ComparisonPair] = Field(default_factory=list, description="List of comparison pairs")
    model_a: str
    model_b: str

    def add_pair(self, output_a: ModelOutput, output_b: ModelOutput, commentary: str = ''):
        """Add a new comparison pair."""
        self.pairs.append(ComparisonPair(output_a=output_a, output_b=output_b, commentary=commentary))

    def to_json(self):
        """Convert to JSON string."""
        return json.dumps({
            "model_a": self.model_a,
            "model_b": self.model_b,
            "pairs": [pair.to_dict() for pair in self.pairs]
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str):
        """Create ComparisonSet from JSON string."""
        data = json.loads(json_str)
        pairs = []
        for pair_data in data['pairs']:
            output_a = ModelOutput(**pair_data["output_a"])
            output_b = ModelOutput(**pair_data["output_b"])
            commentary = pair_data.get("commentary", "")
            pairs.append(ComparisonPair(output_a=output_a, output_b=output_b, commentary=commentary))
        return cls(pairs=pairs, model_a=data['model_a'], model_b=data['model_b'])

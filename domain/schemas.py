from enum import Enum
from pydantic import BaseModel
from typing import List


class BoundingBox(BaseModel):
    object_name: str
    y1: float
    x1: float
    y2: float
    x2: float


class BoundingBoxes(BaseModel):
    boxes: List[BoundingBox]


class ModelProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
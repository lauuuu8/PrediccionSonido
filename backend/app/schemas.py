from typing import List
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    filename: str
    class_id: int
    class_name: str
    probs: List[float]

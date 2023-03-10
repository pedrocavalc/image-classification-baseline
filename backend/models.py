from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class InferenceInput(BaseModel):
    """
    Input values for model inference
    """
    image: str = Field(..., title='image in b64')


class InferenceResponse(BaseModel):
    """
    Input values for model inference
    """
    classe: str = Field(..., title='class')

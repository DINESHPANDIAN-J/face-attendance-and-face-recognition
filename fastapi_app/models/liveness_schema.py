from pydantic import BaseModel

class LivenessResult(BaseModel):
    name: str
    user_id: str
    liveness_score: float
    status: str

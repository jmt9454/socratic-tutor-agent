from typing import Literal
from pydantic import BaseModel, Field

class Evaluation(BaseModel):
    """A response from the Evaluator Node"""
    decision: Literal["ADVANCE", "REMEDIATE"] = Field(
        description="ADVANCE if the student has grasped the gist of the FIRST remaining learning outcome and it should be marked complete. REMEDIATE if the student is fundamentally lost and the outcome should be retaught."
    )
    justification: str = Field(
        description="A string representing the internal monologue justification of the Evaluator's decision to be passed to the Inquisitor. This message should be concise and refer to the user in the third person."
    )
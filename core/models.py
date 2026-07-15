from typing import Literal
from pydantic import BaseModel, Field

class Evaluation(BaseModel):
    """A response from the Evaluator Node"""
    decision: Literal["ADVANCE", "REMEDIATE", "NO_ATTEMPT"] = Field(
        description="ADVANCE if the student has grasped the gist of the FIRST remaining learning outcome and it should be marked complete. REMEDIATE if the student attempted to engage but is fundamentally lost, so the outcome should be retaught. NO_ATTEMPT if the message is a greeting, social nicety, or meta-comment that doesn't engage with the concept at all."
    )
    justification: str = Field(
        description="A string representing the internal monologue justification of the Evaluator's decision to be passed to the Inquisitor. This message should be concise and refer to the user in the third person."
    )
    student_question: str = Field(
        default="",
        description="If the student's latest message contains an explicit question or request "
                    "for clarification, restate it here concisely. Otherwise, empty string."
)
from typing import Literal
from pydantic import BaseModel, Field

class Evaluation(BaseModel):
    """A response from the Evaluator Node"""
    decision: Literal["ADVANCE", "REMEDIATE", "NO_ATTEMPT", "MOVE_ON"] = Field(
        description="ADVANCE if the student has grasped the gist of the current teaching target and it should be marked complete. REMEDIATE if the student attempted to engage but is fundamentally lost, so it should be retaught. NO_ATTEMPT if the message is a greeting, social nicety, or meta-comment that doesn't engage with the concept at all. MOVE_ON if the student EXPLICITLY asks to skip this, move on, or just be told the answer."
    )
    justification: str = Field(
        description="A string representing the internal monologue justification of the Evaluator's decision to be passed to the Inquisitor. This message should be concise and refer to the user in the third person."
    )
    student_question: str = Field(
        default="",
        description="If the student's latest message contains an explicit question or request "
                    "for clarification, restate it here concisely. Otherwise, empty string."
)
    satisfied_upcoming: list[int] = Field(
        default_factory=list,
        description="Only meaningful when decision is ADVANCE and UPCOMING BEATS are listed "
                    "in the prompt. The numbers (+1, +2, ...) of the upcoming beats whose "
                    "questions the student's OWN words have ALREADY fully answered — in this "
                    "answer or earlier in the conversation. Include a beat number ONLY if the "
                    "student themselves explicitly provided what that beat is designed to "
                    "elicit; tutor statements and partial coverage never count. Often empty."
    )

class ArcPlan(BaseModel):
    """A discovery arc from the Arc Planner Node"""
    beats: list[str] = Field(
        description="2-4 sequential teaching beats for the target outcome. Each beat is a "
                    "concrete, single-turn instruction to the tutor: what to present/show and "
                    "what single question to ask. Beats progress from observation toward the "
                    "student articulating the core idea themselves."
    )
    formal_term: str = Field(
        default="",
        description="The formal name of the concept, ONLY if this outcome teaches exactly one "
                    "genuine term of art (e.g., 'homograph attack', 'typosquatting', 'open "
                    "redirect'). Empty string if the outcome compares multiple named concepts, "
                    "teaches a habit, or has no established name — NEVER invent a label."
    )
from pydantic import BaseModel, Field

class Evaluation(BaseModel):
    """A response from the Evaluator Node"""
    remaining_learning_outcomes: list[str] = Field(
        description="A list of strings. Each string represents learning outcomes in which the user has not demonstrated proficiency"
    )
    justification: str = Field(
        description="A string representing the internal monologue justification of the Evaluator's decision to be passed to the Inquisitor. This message should be concise and refer to the user in the third person."
    )
# agent/nodes.py
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from .state import AgentState

# --- Model Initialization ---
# It's common to initialize your model here so all nodes can share it.
# Or, you could pass it into your node functions if you need more flexibility.
model = ChatOpenAI(model="gpt-4o-mini")

# --- Data Models ---

class Evaluation(BaseModel):
    """A response from the Evaluator Node"""
    remaining_learning_outcomes: list[str] = Field(
        description="A list of strings. Each string represents learning outcomes in which the user has not demonstrated proficiency"
    )
    justification: str = Field(
        description="A string representing the internal monologue justification of the Evaluator's decision to be passed to the Inquisitor."
    )

# --- Node Functions ---

def router_node(state: AgentState):
    """
    A simple router node that decides the next node based on unmet topics.
    """
    print("Router node invoked.")
    return {}

def planner_node(state: AgentState):
    """
    A planner node which manages AgentState
    """
    print("Planner node invoked.")
    remaining_topics = state.get("remaining_topics", [])
    completed_topics = state.get("completed_topics", [])
    if not remaining_topics and not completed_topics:
        for topic, outcomes in reversed(state.get("learning_outcomes").items()):
            remaining_topics.append(topic)
    print(completed_topics)
    for topic in completed_topics:
        remaining_topics.remove(topic)

    print(remaining_topics)
    return {"remaining_topics": remaining_topics}


def inquisitor_node(state: AgentState):
    """
    An inquisitor node that asks the user a question about the current topic.
    """
    print("Inquisitor node invoked.")
    inquisitor_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    messages = state.get("messages")
    internal_monologue = state.get("internal_monologue")
    overall_goal = state.get("overall_goal")
    current_topic = state.get("remaining_topics")[-1]
    learning_outcomes = state.get("learning_outcomes")[current_topic]

    prompt = f"""
    You are "The Inquisitor," a master Socratic tutor. Your role is to guide a student to discover knowledge for themselves.

    **Your Core Directives:**
    1.  **NEVER give the direct answer.** Always respond with a question.
    2.  **ASK ONE CONCISE QUESTION AT A TIME.** Your goal is a dialogue, not an interrogation.
    3.  **USE YOUR CONTEXT.** Read the 'Internal Monologue' to understand the student's status and the evaluator's recent thoughts.
    4.  **STAY FOCUSED.** Your *only* goal is to guide the student to the *first* unmet outcome on their list.

    ---
    **Overall Mission:** Help the student learn: {overall_goal}

    **Internal Monologue (Your Private Thoughts):**
    {internal_monologue if internal_monologue else "No thoughts yet."}

    **Current Topic:** {current_topic}

    **Learning Plan (Your Goal is the FIRST item on this list):**
    {learning_outcomes}

    **Your Task:**
    Based on the student's history and your 'Internal Monologue', formulate **one single, concise, Socratic question** to guide them toward the *first* item on the Learning Plan.
    """
    model_messages = [SystemMessage(content=prompt)]
    model_messages.extend(messages)

    response = inquisitor_model.invoke(model_messages)
    return {"messages": [response]}

def evaluator_node(state: AgentState):
    """
    An evaluator node that judges user understanding of the current topic.
    """
    print("Evaluator node invoked.")
    evaluator_model = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Evaluation)
    messages = state.get("messages")
    overall_goal = state.get("overall_goal")
    current_topic = state.get("remaining_topics")[-1]
    learning_outcomes = state.get("learning_outcomes")[current_topic]
    remaining_learning_outcomes = state.get("remaining_learning_outcomes",learning_outcomes)
    
    prompt = f"""
    You are an expert Computer Science educator. Your goal is to evaluate if a student's answer conceptually fulfills the learning objectives. The student is in an introductory course and should be graded as such.

    **Your Process:**
    1.  Read the **Remaining Rubric** and the **Entire Conversation History**.
    2.  For each item in the rubric, check the history.

    **You must follow these 3 cases for each item:**

    **Case 1: The answer is correct (either academic or a good analogy).**
    * *Example:* "It's a pointer to memory" OR "It's like a labeled box."
    * **Action:** Mark this item as MET.
    * **Justification:** Briefly praise the student's correct answer.

    **Case 2: The answer is *way off*, *very confused*, or *conceptually wrong*.**
    * *Example:* "Is it like a function?" or "It's the letter 'x'."
    * **Action:** Do **NOT** mark this as MET.
    * **Justification:** This is critical. Your justification **MUST** start with a "HINT:" tag. This is a secret note for the Socratic Inquisitor to give the student an example.
    * *Example Justification:* "HINT: The student is very confused about the definition. They need a concrete example to get started."

    **Case 3: The answer is missing, or the student said "I don't know."**
    * **Action:** Do **NOT** mark this as MET.
    * **Justification:** Simply state that the item is still unmet.
    * *Example Justification:* "The student has not yet provided a definition for a variable."

    **Your Task:**
    Return a list of all `met_items` that are now satisfied (only Case 1).
    Provide a `justification` that follows these rules (especially Case 2).

    **Overall Goal:** {overall_goal}
    **Current Topic:** {current_topic}
    **Remaining Rubric Items (Learning Outcomes):**
    {remaining_learning_outcomes}

    Evaluate the student's conceptual understanding based on the full history and these 3 cases.
    """

    model_messages = [SystemMessage(content=prompt)] + messages
    response = evaluator_model.invoke(model_messages)
    remaining_learning_outcomes = response.remaining_learning_outcomes
    justification = response.justification
    print(f"remaining: {remaining_learning_outcomes}")

    return_payload = {
        "internal_monologue": [justification],
        "remaining_learning_outcomes": remaining_learning_outcomes
    }
    if not remaining_learning_outcomes:
        print(f"Topic Completed: {current_topic}")
        return_payload["completed_topics"] = [current_topic]

    return return_payload

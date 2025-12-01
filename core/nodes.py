# agent/nodes.py
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from state import AgentState

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
        description="A string representing the internal monologue justification of the Evaluator's decision to be passed to the Inquisitor. This message should be concise and refer to the user in the third person."
    )

# --- Node Functions ---

def router_node(state: AgentState):
    """
    A simple router node that decides the next node based on unmet topics.
    """
    #print("Router node invoked.")
    return {}

def planner_node(state: AgentState):
    """
    A planner node which manages AgentState
    """
    #print("Planner node invoked.")
    remaining_topics = state.get("remaining_topics", [])
    completed_topics = state.get("completed_topics", [])
    if not remaining_topics and not completed_topics:
        for topic, outcomes in reversed(state.get("learning_outcomes").items()):
            remaining_topics.append(topic)
    #print(completed_topics)
    for topic in completed_topics:
        remaining_topics.remove(topic)

    current_topic = remaining_topics[-1]
    remaining_learning_outcomes = state.get("learning_outcomes")[current_topic]
    #print(f"Planning for topic: {current_topic}")
    #print(f"Remaining topics: {remaining_topics}")
    #print(f"Remaining learning outcomes: {remaining_learning_outcomes}")
    return {"remaining_topics": remaining_topics, "remaining_learning_outcomes": remaining_learning_outcomes}

def topic_summarizer_node(state: AgentState):
    pass

def inquisitor_node(state: AgentState):
    """
    An inquisitor node that asks the user a question about the current topic.
    """
    #print("Inquisitor node invoked.")
    inquisitor_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    messages = state.get("messages")
    internal_monologue = state.get("internal_monologue")
    overall_goal = state.get("overall_goal")
    current_topic = state.get("remaining_topics")[-1]
    learning_outcomes = state.get("learning_outcomes")
    remaining_outcomes = state.get("remaining_learning_outcomes")

    print(current_topic)
    print(remaining_outcomes)

    prompt = f"""
    You are "The Inquisitor," a master Socratic tutor. Your role is to guide a student to discover knowledge for themselves.

    Your Core Directives:

    Always end your response with a question.

    ASK ONE CONCISE QUESTION AT A TIME. Your goal is a dialogue, not an interrogation.

    USE YOUR CONTEXT. Read the 'Internal Monologue' to understand the student's status and the evaluator's recent thoughts.

    STAY FOCUSED. Your only goal is to guide the student to the first unmet outcome on their list.

    PROVIDE A ENCOURAGING STATEMENT. Precede your question with one short, supportive, educational sentence that validates the student's last answer or reframes the topic before asking the next question.

    USE CONTEXTUAL EXAMPLES AND CASES. If the 'Internal Monologue' contains a "HINT:" or the conversation shows a gap, also precede the question with some example or case for the student to consider before receiving the next question. Incorporate examples or analogies to help the student grasp the concept.

    Overall Mission: Help the student learn: {overall_goal}

    Internal Monologue (Your Private Thoughts): {internal_monologue if internal_monologue else "No thoughts yet."}

    Current Topic: {current_topic}

    All learning outcomes for this topic: {learning_outcomes[current_topic]}

    Remaining Learning Plan (Your Goal is the FIRST item on this list): {remaining_outcomes}

    Your Task: Based on the student's history and your 'Internal Monologue', formulate one single, concise, Socratic question and precede it with one short, guiding sentence to move them toward the first item on the Learning Plan.
    """
    model_messages = [SystemMessage(content=prompt)]
    model_messages.extend(messages)

    response = inquisitor_model.invoke(model_messages)
    return {"messages": [response]}

def evaluator_node(state: AgentState):
    """
    An evaluator node that judges user understanding of the current topic.
    """
    #print("Evaluator node invoked.")
    evaluator_model = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Evaluation)
    messages = state.get("messages")
    overall_goal = state.get("overall_goal")
    current_topic = state.get("remaining_topics")[-1]
    learning_outcomes = state.get("learning_outcomes")[current_topic]
    remaining_learning_outcomes = state.get("remaining_learning_outcomes",learning_outcomes)
    
    prompt = f"""
    You are an empathetic Computer Science educator. Your goal is to evaluate if a student's answer demonstrates a **working intuition** of the learning objectives. The student is in an introductory course; **rigor and technical precision are NOT required at this stage.**

    Your responses are sent to the tutoring agent's Inquisitor node to help guide further questioning.

    **Your Process:**
    1.  Read the **Remaining Rubric** and the **Entire Conversation History**.
    2.  For each item in the rubric, check the history.

    **Guiding Principle for "Sufficiency":**
    If the student understands the general idea, the "gist," or can describe *what* the concept does (even without using the correct terminology), mark it as MET. We want to maintain flow and confidence. **Do not nitpick.**

    **You must follow these 3 cases for each item:**

    **Case 1: The answer is sufficient, functional, or directionally correct.**
    * **Criteria:** The student shows they "get it," even if the explanation is messy, informal, or uses analogies. Partial understanding is acceptable if it allows them to move forward.
    * *Example:* "It holds the number" (Acceptable for a variable).
    * *Example:* "It makes the code happen again" (Acceptable for a loop).
    * **Action:** Mark this item as MET.
    * **Justification:** Briefly validate the student's intuition.

    **Case 2: The answer reveals a fundamental misconception that will block future learning.**
    * **Criteria:** The answer is not just wrong, but actively harmful to their understanding or completely unrelated.
    * *Example:* "A variable is a function."
    * *Example:* "The loop runs backwards only."
    * **Action:** Do **NOT** mark this as MET.
    * **Justification:** Explain the critical gap. Your justification **MUST** include a "HINT:" tag.
    * *Example Justification:* "HINT: The student is confusing data storage with action. Use the 'box' analogy."

    **Case 3: The answer is entirely missing, or the student said "I don't know."**
    * **Action:** Do **NOT** mark this as MET.
    * **Justification:** Explain the gap. Your justification **MUST** include a "HINT:" tag.
    * *Example Justification:* "The student hasn't defined X yet. HINT: Provide a real-world scenario where X is needed."

    **Your Task:**
    Return a list of all `met_items` that are now satisfied (Case 1).
    Provide a `justification` that follows these rules.

    **Overall Goal:** {overall_goal}
    **Current Topic:** {current_topic}
    **Remaining Rubric Items (Learning Outcomes):**
    {remaining_learning_outcomes}

    Evaluate the student's conceptual understanding based on the full history and these 3 cases.
    """

    model_messages = [SystemMessage(content=prompt)] + messages
    response = evaluator_model.invoke(model_messages)
    new_remaining_learning_outcomes = response.remaining_learning_outcomes
    justification = response.justification
    print(f"remaining: {justification}")
    #print(f"remaining: {new_remaining_learning_outcomes}")

    return_payload = {
        "internal_monologue": [justification],
        "remaining_learning_outcomes": new_remaining_learning_outcomes
    }

    if not new_remaining_learning_outcomes:
        #print(f"Topic Completed: {current_topic}")
        return_payload["completed_topics"] = [current_topic]

    return return_payload

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from models import Evaluation
from state import AgentState

model = ChatOpenAI(model="gpt-4o-mini")

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
    inquisitor_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    messages = state.get("messages", [])
    internal_monologue = state.get("internal_monologue", "No specific thoughts right now.")
    overall_goal = state.get("overall_goal", "the current subject")
    learning_outcomes = state.get("learning_outcomes", {})
    remaining_outcomes = state.get("remaining_learning_outcomes", [])
    remaining_topics = state.get("remaining_topics", [])

    if not remaining_topics:
        current_topic = "Review"
        current_outcomes = "Review all previous material."
    else:
        current_topic = remaining_topics[-1]
        current_outcomes = learning_outcomes.get(current_topic, "No specific outcomes provided.")

    base_prompt = f"""
    You are "The Inquisitor," a master Socratic tutor. Your role is to guide a student to discover knowledge for themselves.
    
    Your Core Directives:
    - Always end your response with a single, concise question.
    - PROVIDE AN ENCOURAGING STATEMENT. Precede your question with one short, supportive sentence or analogy.
    - MANAGE TANGENTS. If the student gets sidetracked, briefly acknowledge what they said, then pivot directly back to your current goal. Do not follow them down rabbit holes.
    
    Overall Mission: Help the student learn: {overall_goal}
    """
    
    anchor_prompt = f"""
    CRITICAL INSTRUCTIONS FOR THIS TURN:
    Current Topic: {current_topic}
    All learning outcomes for this topic: {current_outcomes}
    Read your 'Internal Monologue' below. You MUST base your next question entirely on the HINT or direction provided here. Do not get sidetracked by the student's last message if it strays from this goal.
    
    Your Internal Monologue: {internal_monologue}
    
    Remaining Learning Plan (Your Goal is the FIRST item): {remaining_outcomes if remaining_outcomes else "Assess overall understanding."}
    
    Task: Formulate your next sentence and question to execute the exact strategy described in your internal monologue.
    """
    
    model_messages = [SystemMessage(content=base_prompt)]
    model_messages.extend(messages)  # The conversation history
    model_messages.append(SystemMessage(content=anchor_prompt)) # The anchor forces the LLM to prioritize the monologue

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
    * **Action:** Mark this item as MET.
    * **Justification:** Briefly validate the student's intuition.

    **Case 2: The answer reveals a fundamental misconception that will block future learning.**
    * **Action:** Do **NOT** mark this as MET.
    * **Justification:** Explain the critical gap. Your justification **MUST** include a "HINT:" tag.

    **Case 3: The answer is entirely missing, or the student said "I don't know."**
    * **Action:** Do **NOT** mark this as MET.
    * **Justification:** Explain the gap. Your justification **MUST** include a "HINT:" tag.

    **Your Task:**
    Analyze the student's current understanding. 
    1. Formulate your `justification` detailing what they got right, or providing a 'HINT:' for what they are missing.
    2. Output a strictly updated list of the `remaining_learning_outcomes`. Remove any items that are MET. Keep the exact text for items that are NOT MET.

    **Overall Goal:** {overall_goal}
    **Current Topic:** {current_topic}
    **Remaining Rubric Items (Learning Outcomes):**
    {remaining_learning_outcomes}
    """

    model_messages = [SystemMessage(content=prompt)] + messages
    response = evaluator_model.invoke(model_messages)
    new_remaining_learning_outcomes = response.remaining_learning_outcomes
    justification = response.justification
    print(f"Evaluator Node: Justification - {justification}")
    print(f"Evaluator Node: Remaining Learning Outcomes - {new_remaining_learning_outcomes}")

    return_payload = {
        "internal_monologue": [justification],
        "remaining_learning_outcomes": new_remaining_learning_outcomes
    }

    if not new_remaining_learning_outcomes:
        print(f"Topic Completed: {current_topic}")
        return_payload["completed_topics"] = [current_topic]

    return return_payload

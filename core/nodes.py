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
    You are "The Socratic Guide," a master tutor. Your role is to actively teach complex concepts by guiding the student to connect the dots and discover the mechanics themselves.

    Overall Curriculum Goal: {overall_goal}

    Your Core Directives:
    1. VALIDATE & CORRECT: Always begin by addressing the student's last input. If they are correct, provide a brief encouraging statement. If they are incorrect or partially right, gently correct the false premise before introducing new information.
    2. TEACH IN MICRO-STEPS: Do not just quiz the student. First, introduce a small, digestible piece of the current objective. Use brief, vivid analogies or concrete scenarios to anchor abstract concepts.
    3. THE SOCRATIC HOOK: Always end your response with exactly ONE concise question. This question must force the student to apply the micro-step you just taught, deduce the next logical piece of the puzzle, or explain the "why."
    4. MANAGE TANGENTS: If the student strays, briefly acknowledge their point, but seamlessly pivot the conversation directly back to the active learning goal. Do not follow them down rabbit holes.
    5. NEVER LECTURE: Never give away the full answer or explain the entire concept at once. If the student struggles, break the concept down into an even smaller piece and ask a simpler guiding question.
    """

    anchor_prompt = f"""
    CRITICAL INSTRUCTIONS FOR THIS TURN:
    Current Topic: {current_topic}
    All Learning Outcomes for this topic: 
    {current_outcomes}

    Your Internal Monologue (Strategic Direction):
    {internal_monologue if internal_monologue else "The conversation is just now starting. Greet the student and introduce the first micro-step of the topic."}

    Remaining Learning Plan (Your immediate target is the FIRST item): 
    {remaining_outcomes if remaining_outcomes else "Assess overall understanding and summarize."}

    Task: Formulate your response to execute the exact strategy described in your Internal Monologue.
    - Ground your explanation in the immediate target from the Remaining Learning Plan.
    - Deliver the necessary micro-step of information.
    - End with a single Socratic question that drives the student toward satisfying this current outcome.
    """
    
    model_messages = [SystemMessage(content=base_prompt)]
    model_messages.extend(messages)  # The conversation history
    model_messages.append(SystemMessage(content=anchor_prompt)) # The anchor forces the LLM to prioritize the monologue

    response = inquisitor_model.invoke(model_messages)
    return {"messages": [response]}

def evaluator_node(state):
    """
    An evaluator node that judges conversational understanding and dictates the tutor's next move.
    """
    print("Evaluator node invoked.")
    evaluator_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).with_structured_output(Evaluation)
    
    messages = state.get("messages", [])
    if not isinstance(messages, list):
        messages = [messages]

    overall_goal = state.get("overall_goal", "General concepts")
    
    remaining_topics = state.get("remaining_topics", [])
    current_topic = remaining_topics[-1] if remaining_topics else "Review"
    
    learning_outcomes = state.get("learning_outcomes", {}).get(current_topic, [])
    remaining_learning_outcomes = state.get("remaining_learning_outcomes", learning_outcomes)
    
    if not isinstance(remaining_learning_outcomes, list):
        remaining_learning_outcomes = [remaining_learning_outcomes]
    
    prompt = f"""
    You are the 'Internal Strategist' for a Socratic tutor. Your goal is to assess conversational progress, NOT to grade a test. We want to maintain a flowing, encouraging dialogue. 

    **Your Core Philosophy:** 1. **Close enough is good enough!** If the student explains the concept in their own words, uses a valid analogy, or gets the "gist" right, they have succeeded. 
    2. **Reward Intuition over Confidence:** If the student provides a directionally correct answer but expresses doubt, answers briefly (e.g., "just vowels?"), or seems unsure, TREAT THIS AS A FULL SUCCESS. Do not penalize them for a lack of confidence.
    3. **The Anti-Trapping Rule:** Do not trap the student in an endless loop on a single concept. If they grasp the core mechanism of the concept, move them forward immediately.
    4. **Strict Scope Isolation:** Look ONLY at the FIRST outcome. If the outcome is just about understanding *what* something is, and they demonstrate that, pass them. Do not hold them back because they didn't explain *how to defend against it* (defense is likely a later outcome).

    **Your Task:**
    Analyze the student's latest response. Focus ONLY on the FIRST item in the Remaining Learning Outcomes list. Decide how the Socratic Guide should respond based on two scenarios:

    **Scenario A: The Gist is Grasped (Move Forward) - DEFAULT TO THIS IF IN DOUBT**
    - Trigger: The student's answer shows they understand the core directional idea of that FIRST outcome, even if brief or unsure.
    - Decision: Output `ADVANCE`.
    - Monologue: Briefly note what they got right. Instruct the Socratic Guide to validate them, celebrate the win, and smoothly introduce the NEXT concept on the list.

    **Scenario B: Fundamental Misunderstanding (Pivot & Re-engage)**
    - Trigger: The student is completely lost, confidently incorrect about the core mechanism, or entirely dodged the concept.
    - Decision: Output `REMEDIATE`.
    - Monologue: Identify the specific point of friction. Instruct the Socratic Guide to validate the effort, and suggest a specific, simpler analogy or narrower question.

    **Overall Goal:** {overall_goal}
    **Current Topic:** {current_topic}
    
    **Remaining Learning Outcomes (Your immediate target is the FIRST one):**
    {remaining_learning_outcomes}
    
    Output your `decision` (ADVANCE or REMEDIATE) and your strategic advice in the `justification` (which becomes the internal monologue).
    """

    model_messages = [SystemMessage(content=prompt)]
    model_messages.extend(messages)

    response = evaluator_model.invoke(model_messages)

    # The model only decides; Python owns list membership.
    if response.decision == "ADVANCE" and remaining_learning_outcomes:
        new_remaining_learning_outcomes = remaining_learning_outcomes[1:]
    else:
        new_remaining_learning_outcomes = list(remaining_learning_outcomes)

    justification = response.justification

    print(f"Evaluator Node: Decision - {response.decision}")
    print(f"Evaluator Node: Justification - {justification}")
    print(f"Evaluator Node: Remaining Learning Outcomes - {new_remaining_learning_outcomes}")

    return_payload = {
        "internal_monologue": [justification],
        "remaining_learning_outcomes": new_remaining_learning_outcomes
    }

    # 6. Re-integrated your original logic for completing a topic!
    if not new_remaining_learning_outcomes:
        print(f"Topic Completed: {current_topic}")
        return_payload["completed_topics"] = [current_topic]

    return return_payload
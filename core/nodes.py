import re

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from models import Evaluation, ArcPlan
from state import AgentState

model = ChatOpenAI(model="gpt-4.1-mini")

_QUOTE = "['\"‘’“”]?"

def _mask_term(text: str, term: str, replacement: str) -> str:
    """Mask a formal term in text, tolerating quotes/hyphens around and between its
    words (so 'homograph attack' also matches \"a 'homograph' attack\")."""
    if not term or not text:
        return text
    words = [re.escape(w) for w in term.split()]
    sep = _QUOTE + r"[\s\-]*" + _QUOTE
    pattern = _QUOTE + sep.join(words) + _QUOTE
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)

def router_node(state: AgentState):
    """
    A simple router node that decides the next node based on unmet topics.
    """
    #print("Router node invoked.")
    return {}

def planner_node(state: AgentState):
    """
    A planner node which manages AgentState.
    Rebuilds remaining_topics from scratch each call (curriculum minus completed)
    rather than mutating state in place. Curriculum order is preserved: topics are
    stored reversed so the current topic is always remaining_topics[-1].
    """
    #print("Planner node invoked.")
    learning_outcomes = state.get("learning_outcomes", {})
    completed_topics = state.get("completed_topics") or []

    remaining_topics = [
        topic for topic in reversed(list(learning_outcomes.keys()))
        if topic not in completed_topics
    ]

    if not remaining_topics:
        # Curriculum finished; after_planner_node routes to END.
        return {"remaining_topics": [], "remaining_learning_outcomes": []}

    current_topic = remaining_topics[-1]
    remaining_learning_outcomes = list(learning_outcomes[current_topic])
    #print(f"Planning for topic: {current_topic}")
    #print(f"Remaining topics: {remaining_topics}")
    #print(f"Remaining learning outcomes: {remaining_learning_outcomes}")
    return {"remaining_topics": remaining_topics, "remaining_learning_outcomes": remaining_learning_outcomes}

def topic_summarizer_node(state: AgentState):
    pass

def arc_planner_node(state: AgentState):
    """
    Generates a short 'discovery arc' (2-4 teaching beats) for the current target
    outcome. Runs before each new outcome is taught; passes through untouched when
    a valid arc already exists. Context-aware: reads the conversation so far and
    weaves in the student's own examples and insights.
    """
    remaining_outcomes = state.get("remaining_learning_outcomes") or []
    target_outcome = remaining_outcomes[0] if remaining_outcomes else None

    if target_outcome is None:
        # Review mode / nothing to plan for.
        return {"current_arc": [], "arc_outcome": None, "arc_term": ""}

    # Arc already generated for this outcome (possibly mid-arc, possibly an empty
    # fallback after a failed generation) -> no-op, no LLM call.
    if state.get("arc_outcome") == target_outcome:
        return {}

    print(f"Arc Planner: generating arc for outcome - {target_outcome}")

    messages = state.get("messages", [])
    overall_goal = state.get("overall_goal", "the current subject")
    remaining_topics = state.get("remaining_topics") or []
    current_topic = remaining_topics[-1] if remaining_topics else "Review"
    later_outcomes = remaining_outcomes[1:]

    prompt = f"""
    You are the 'Lesson Choreographer' for a Socratic tutor. Design a short DISCOVERY ARC
    for the target outcome below: 2-4 sequential "beats". Each beat is one conversational
    turn — a concrete instruction to the tutor stating what to present and what ONE question
    to ask. The student should DISCOVER the idea, not be told it.

    Arc design principles:
    1. OBSERVE BEFORE NAMING: Beat 1 presents a concrete artifact or scenario (e.g., two
       nearly identical URLs, a suspicious email line) and asks for an observation or
       prediction — BEFORE any terminology or definition is given.
    2. NARROW THE EVIDENCE: Middle beats isolate or contrast the key detail (e.g., "place
       the two differing characters side by side and ask again") so the mechanism becomes
       visible to the student.
    3. STUDENT EXPLAINS, TUTOR NAMES LATER: Your final beat asks the student to explain the
       idea in their own words — NEVER to produce, guess, or recall the formal term, which
       they have not been taught and cannot know. Never write beats like "what would you
       call this trick?". Do NOT put the term reveal inside any beat — it is handled in a
       separate follow-up turn automatically. Simply report the concept's formal name in
       the `formal_term` field (empty if the outcome has no formal term).
    4. BUILD ON THE CONVERSATION: Review the chat history below. If the student previously
       offered examples, analogies, or insights (e.g., a domain they suggested, a scenario
       they described), reuse them in your beats to tie concepts together. Connect this
       outcome to ideas the student has already mastered when a genuine link exists.
       NEVER include a beat whose question the student has already answered earlier in the
       conversation — in particular, do NOT ask "why would attackers do this?" again if they
       already explained attacker motives for a sibling concept. For a concept similar to one
       the student already mastered, replace the motive beat with a CONTRAST beat: ask how
       this concept DIFFERS from the one they already know.
    5. STAY IN SCOPE: Do NOT cover material from the later outcomes listed below — they
       will get their own arcs. Where useful, a beat may break down terminology
       (e.g., homo- 'same' + -graph 'writing').
    6. Each beat must be executable in a single text message and must contain EXACTLY ONE
       question — never two questions chained together (no "...? And why...?"). Any example
       domains must be plausible and pedagogically accurate.
    7. URL SAFETY: Write every example URL or domain as plain text in backticks (e.g.,
       `paypa1.com`) — NEVER as a markdown link, and never with an https:// href. These
       beats will be shown to students; lookalike domains must never be clickable.

    Overall Goal: {overall_goal}
    Current Topic: {current_topic}

    TARGET OUTCOME (design the arc for this):
    {target_outcome}

    LATER OUTCOMES (do NOT teach these yet):
    {later_outcomes if later_outcomes else "(none)"}

    The conversation so far follows. Output 2-4 beats.
    """

    arc_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(ArcPlan)
    model_messages = [SystemMessage(content=prompt)]
    model_messages.extend(messages)

    try:
        plan = arc_model.invoke(model_messages)
        beats = [b.strip() for b in (plan.beats or []) if b and b.strip()][:4]
        formal_term = (plan.formal_term or "").strip()
    except Exception as e:
        print(f"Arc Planner: generation failed ({e}); falling back to outcome-level teaching.")
        beats = []
        formal_term = ""

    # Deterministic finale: the term reveal gets its own turn so it can never be
    # skipped when the evaluator advances past the student's explanation.
    if beats and formal_term:
        beats.append(
            f"Congratulate the student: what they just described is exactly what security "
            f"experts call a '{formal_term}' — reveal that term now as the payoff for their "
            f"explanation. Then ask ONE summary question: now that they know the name, ask "
            f"them to tell you what a '{formal_term}' is in their own words."
        )

    for i, beat in enumerate(beats, 1):
        print(f"Arc Planner: Beat {i} - {beat}")

    return {"current_arc": beats, "arc_outcome": target_outcome, "arc_term": formal_term}

def inquisitor_node(state: AgentState):
    inquisitor_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
    messages = state.get("messages", [])
    # internal_monologue accumulates every evaluator justification; only the
    # LATEST entry is the current strategy — older ones are stale directives.
    monologue_history = state.get("internal_monologue") or []
    internal_monologue = monologue_history[-1] if monologue_history else None
    overall_goal = state.get("overall_goal", "the current subject")
    learning_outcomes = state.get("learning_outcomes", {})
    remaining_outcomes = state.get("remaining_learning_outcomes", [])
    target_outcome = remaining_outcomes[0] if remaining_outcomes else None
    current_arc = state.get("current_arc") or []
    current_beat = current_arc[0] if current_arc else None
    upcoming_beats = current_arc[1:]
    if upcoming_beats:
        upcoming_block = "\n".join(f"  +{i+1}. {b}" for i, b in enumerate(upcoming_beats))
    else:
        upcoming_block = "(none — this is the final beat of the arc)" if current_beat else "(no arc)"
    remaining_topics = state.get("remaining_topics", [])

    if not remaining_topics:
        current_topic = "Review"
        current_outcomes = "Review all previous material."
    else:
        current_topic = remaining_topics[-1]
        current_outcomes = learning_outcomes.get(current_topic, "No specific outcomes provided.")

    # Term masking: while the reveal beat is still 2+ turns away, scrub the formal
    # term from every outcome text shown to the tutor — the last leak channel.
    arc_term = (state.get("arc_term") or "").strip()
    display_target = target_outcome
    display_outcomes = current_outcomes
    if arc_term and len(current_arc) >= 2:
        display_target = _mask_term(display_target or "", arc_term, "[term withheld until the reveal beat]")
        if isinstance(display_outcomes, list):
            display_outcomes = [_mask_term(o, arc_term, "[term withheld]") for o in display_outcomes]
        else:
            display_outcomes = _mask_term(str(display_outcomes), arc_term, "[term withheld]")

    base_prompt = f"""
    You are "The Socratic Guide," a master tutor. Your role is to actively teach complex concepts by guiding the student to connect the dots and discover the mechanics themselves.

    Overall Curriculum Goal: {overall_goal}

    Your Core Directives:
    1. VALIDATE & CORRECT: Always begin by addressing the student's last input. If they are correct, provide a brief encouraging statement. If they are incorrect or partially right, gently correct the false premise before introducing new information.
    2. TEACH IN MICRO-STEPS: Do not just quiz the student. First, introduce a small, digestible piece of the current objective. Use brief, vivid analogies or concrete scenarios to anchor abstract concepts.
    3. THE SOCRATIC HOOK: Always end your response with exactly ONE concise question. This question must force the student to apply what you just taught, deduce the next logical piece of the puzzle, or explain the "why."
    4. MANAGE TANGENTS: If the student strays, briefly acknowledge their point, but seamlessly pivot the conversation directly back to the active learning goal. Do not follow them down rabbit holes.
    5. NEVER LECTURE: Never give away the full answer or explain the entire concept at once. If the student struggles, break the concept down into an even smaller piece and ask a simpler guiding question.
    6. SPEAK NATURALLY: These directives are stage directions, not vocabulary. Never narrate, reference, or label them in your replies — no meta-commentary like "since this is our first turn there's nothing to validate," and no internal jargon like "micro-step" or "target outcome" as headings or phrases. Write only what a warm, natural human tutor would actually say aloud.
    7. URL SAFETY: Write every example URL or domain name in backticks as inline code (e.g., `paypa1.com`), NEVER as a markdown link. Example domains must never render as clickable — some lookalike domains are registered by real attackers. If your instructions or notes contain a URL in markdown link format, strip the link and render only the domain as plain inline code.
    """

    anchor_prompt = f"""
    CRITICAL INSTRUCTIONS FOR THIS TURN:
    Current Topic: {current_topic}
    All Learning Outcomes for this topic:
    {display_outcomes}

    Your Internal Monologue (Strategic Direction):
    {internal_monologue if internal_monologue else "This is the very first message of the lesson — the student has not said anything yet. Skip any validation or correction entirely; do not mention the absence of prior messages. Simply greet the student warmly and begin teaching the first small piece of the topic."}

    Remaining Learning Plan:
    {remaining_outcomes if remaining_outcomes else "Assess overall understanding and summarize."}

    YOUR TARGET OUTCOME (for your orientation ONLY — never state its content or terminology before the arc's final beat calls for it):
    {display_target if display_target else "No specific outcome remains; assess overall understanding and summarize."}

    YOUR CURRENT TEACHING BEAT (this turn's concrete move — execute THIS and only this):
    {current_beat if current_beat else "No scripted beat this turn — teach directly toward the Target Outcome."}

    UPCOMING BEATS (your full lesson plan — EMBARGOED. Each gets its own future turn, so nothing is lost by waiting. The student must not hear their content, answers, or terminology yet. Before explaining or naming anything, check this list: if it appears below, it is scheduled — leave it for its turn):
    {upcoming_block}

    Task: Formulate your response to execute the exact strategy described in your Internal Monologue.
    - If your Internal Monologue instructs you to answer a student question, answer it fully and plainly FIRST, before anything else. Never skip or defer their question.
    - Execute the Current Teaching Beat: present what it says to present, and ask the question it says to ask, woven naturally into your reply. Do not skip ahead to future beats or reveal what the beat is building toward.
    - If the Current Teaching Beat asks only for an observation, comparison, or prediction: present the artifact and ask — deliver NO explanation, and do NOT reveal or hint at the difference, trick, or answer the beat is designed to elicit. Giving away what the student is supposed to discover is a failure.
    - Never ask the student to produce, guess, or recall a formal term they have not been taught (e.g., "what would you call this?"). Ask only for an explanation in their own words; once they explain it, YOU supply the formal term as the payoff.
    - If the student has ALREADY answered the Current Teaching Beat's question earlier in the conversation, do NOT re-ask it verbatim. Acknowledge their earlier point ("you actually touched on this when you said...") and ask a follow-up that goes one level deeper instead.
    - Ground your explanation in the Target Outcome above.
    - Deliver the necessary micro-step of information.
    - If the Target Outcome enumerates multiple items (types, channels, tactics), introduce ONLY ONE item this turn and question on it. Let subsequent turns cover the rest as the conversation progresses. Never enumerate the full list in a single message.
    - End with a single Socratic question that DIRECTLY probes the Target Outcome. The student's answer will be used as evidence that this exact outcome is met, so do not ask about adjacent material (e.g., defense or prevention tips) unless the Target Outcome itself is about that.
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
    evaluator_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1).with_structured_output(Evaluation)
    
    messages = state.get("messages", [])
    if not isinstance(messages, list):
        messages = [messages]

    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human is None:
        # Nothing from the student to evaluate (e.g., fresh or resumed thread); leave state untouched.
        print("Evaluator Node: No student response found; skipping evaluation.")
        return {}

    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    tutor_question = last_ai.content if last_ai else "(no prior tutor message)"

    overall_goal = state.get("overall_goal", "General concepts")
    
    remaining_topics = state.get("remaining_topics", [])
    current_topic = remaining_topics[-1] if remaining_topics else "Review"
    
    learning_outcomes = state.get("learning_outcomes", {}).get(current_topic, [])
    remaining_learning_outcomes = state.get("remaining_learning_outcomes", learning_outcomes)
    
    if not isinstance(remaining_learning_outcomes, list):
        remaining_learning_outcomes = [remaining_learning_outcomes]

    current_arc = state.get("current_arc") or []
    current_beat = current_arc[0] if current_arc else None
    upcoming_beats_eval = current_arc[1:]
    if upcoming_beats_eval:
        upcoming_eval_block = "\n".join(f"  +{i+1}. {b}" for i, b in enumerate(upcoming_beats_eval))
    else:
        upcoming_eval_block = "(none)"

    prompt = f"""
    You are the 'Internal Strategist' for a Socratic tutor. Your goal is to assess conversational progress, NOT to grade a test. We want to maintain a flowing, encouraging dialogue. 

    **Your Core Philosophy:** 1. **Close enough is good enough!** If the student explains the concept in their own words, uses a valid analogy, or gets the "gist" right, they have succeeded. 
    2. **Reward Intuition over Confidence:** If the student provides a directionally correct answer but expresses doubt, answers briefly (e.g., "just vowels?"), or seems unsure, TREAT THIS AS A FULL SUCCESS. Do not penalize them for a lack of confidence.
    3. **The Anti-Trapping Rule:** Do not trap the student in an endless loop on a single concept. If they grasp the core mechanism of the concept, move them forward immediately.
    4. **Strict Scope Isolation:** Look ONLY at the FIRST outcome. If the outcome is just about understanding *what* something is, and they demonstrate that, pass them. Do not hold them back because they didn't explain *how to defend against it* (defense is likely a later outcome). For enumerated outcomes (e.g., "email vs. SMS vs. voice"), the student must show they can DISTINGUISH the items — grasping only one item is progress, not completion.
    5. **The Echo Guard:** Brief answers are fine, but the answer must CONTAIN something — at least one concrete element of the outcome's core mechanism, in the student's own words. An answer that merely restates the tutor's phrasing, repeats the question's premise, or affirms without content (e.g., "because I can see it", "because they don't know") is NOT evidence. REMEDIATE with a narrower question that asks for the missing piece.

    **Your Task:**
    Evaluate ONLY the student's latest answer, quoted below. The conversation history is provided for context only. IMPORTANT: assistant messages in the history are the TUTOR speaking, NOT the student — never credit the student with anything the tutor said. Focus ONLY on the FIRST item in the Remaining Learning Outcomes list. Decide how the Socratic Guide should respond based on two scenarios:

    **Scenario A: The Gist is Grasped (Move Forward) - DEFAULT TO THIS IF IN DOUBT**
    - Trigger: The student's answer shows they understand the core directional idea of that FIRST outcome, even if brief or unsure.
    - Decision: Output `ADVANCE`.
    - Monologue: Briefly note what they got right. Instruct the Socratic Guide to validate them, celebrate the win, and smoothly introduce the NEXT concept on the list.

    **Scenario B: Fundamental Misunderstanding (Pivot & Re-engage)**
    - Trigger: The student is completely lost, confidently incorrect about the core mechanism, entirely dodged the concept, OR asked how/why the current concept works. A question about the mechanism is direct evidence they do NOT yet grasp it — prefer REMEDIATE and make answering their question the core of your monologue.
    - Decision: Output `REMEDIATE`.
    - Monologue: Identify the specific point of friction. Instruct the Socratic Guide to validate the effort, and suggest a specific, simpler analogy or narrower question.

    **Scenario C: No Attempt (Social / Meta / Blank)**
    - Trigger: The message is a greeting, social nicety, meta-comment ("ok", "sure", "let's go"), or otherwise not an attempt to engage with the concept at all. This is NOT a misunderstanding — do not treat it as one.
    - Decision: Output `NO_ATTEMPT`.
    - Monologue: Instruct the Guide to respond warmly and briefly, then re-ask its pending question — do NOT re-teach the concept from scratch.

    **Scenario D: Student Explicitly Asks to Move On**
    - Trigger: The student EXPLICITLY asks to skip this, move on, or just be told the answer ("can we skip this", "let's move on", "I give up, just tell me"). General boredom or complaints without an explicit request are NOT this — that is NO_ATTEMPT.
    - Decision: Output `MOVE_ON`.
    - Monologue: Note that the student wants to move forward; the system will handle the concession.

    **Question Capture (applies to ALL scenarios):**
    If the student's answer contains ANY explicit question or request for clarification, you MUST restate it concisely in `student_question` — even when you ADVANCE. A student's question must never be silently dropped.

    **Term Secrecy (applies to ALL scenarios):**
    If the concept being taught has a formal name (e.g., 'homograph attack', 'typosquatting'), do NOT use that name anywhere in your `justification` — the tutor reveals it only at its scheduled moment, and your justification is fed directly to the tutor. Refer to it as "this concept" or "this tactic" instead.

    **Overall Goal:** {overall_goal}
    **Current Topic:** {current_topic}
    
    **Remaining Learning Outcomes (Your immediate target is the FIRST one):**
    {remaining_learning_outcomes}

    **THE CURRENT TEACHING BEAT (if present, this is your judging target — NOT the whole outcome):**
    {f'"{current_beat}" — ADVANCE means the student cleared THIS beat alone (e.g., made the requested observation correctly). Do not require mastery of the full outcome; later beats handle the rest. Your philosophy rules (gist, Echo Guard, question capture) apply at the beat level.' if current_beat else "(none — judge the student's answer against the FIRST outcome directly)"}

    **Final-Beat Strictness (overrides the leniency rules above):** If the Current Teaching Beat asks the student to define or explain the concept in their own words (including using its just-revealed name), ADVANCE requires the answer to actually CONTAIN that definition — the mechanism, stated by the student. A defensive takeaway ("so I should check links carefully?"), a bare question, or simple agreement is NOT a definition — REMEDIATE and re-ask for the explanation.

    **UPCOMING BEATS in this arc (for `beats_cleared` assessment only):**
    {upcoming_eval_block}

    **Multi-Beat Clearing (`beats_cleared`, only when you ADVANCE):** Check the upcoming beats above. If the student's OWN words — in this answer or earlier in the conversation — have ALREADY fully provided what an upcoming beat is designed to elicit, count it: beats_cleared = 1 means just the current beat, 2 means current plus the next, and so on, counting consecutively from the current beat. STRICT: something the TUTOR said never counts as student evidence; a partial or implied answer never counts.
    POSITIVE EXAMPLE — do NOT hoard beats: if the current beat asks for an observation and the student's answer makes the observation AND explains the mechanism AND gives the attacker's motive that the next two beats were going to elicit, output 3, not 1. Rich, detailed answers SHOULD clear multiple beats; withholding credit forces a knowledgeable student to repeat themselves, which is a failure of your role. Reserve beats_cleared = 1 for answers that genuinely address only the current beat.

    **THE TUTOR'S MOST RECENT MESSAGE (what the student was responding to):**
    "{tutor_question}"

    **THE STUDENT'S LATEST ANSWER (the only text you are evaluating):**
    "{last_human.content}"

    NOTE: If the tutor's question drifted away from the FIRST outcome (e.g., asked about prevention when the outcome is a definition), judge the student's answer against the FIRST outcome itself, not against the drifted question. A reasonable answer to the question actually asked is NOT automatic evidence the outcome is met.

    Output your `decision` (ADVANCE, REMEDIATE, NO_ATTEMPT, or MOVE_ON), your strategic advice in the `justification` (which becomes the internal monologue), and any `student_question` (empty string if none).
    """

    model_messages = [SystemMessage(content=prompt)]
    model_messages.extend(messages)

    response = evaluator_model.invoke(model_messages)

    # The model only decides; Python owns list membership.
    # With an arc: ADVANCE pops beats_cleared beats (capped so the reveal beat can
    # never be skipped from an earlier beat); the outcome pops when the arc empties.
    # Without an arc (legacy/fallback): ADVANCE pops the outcome directly.
    new_arc = list(current_arc)
    new_arc_outcome = state.get("arc_outcome")
    new_remaining_learning_outcomes = list(remaining_learning_outcomes)
    cleared = 1

    # Frustration cap: 3 consecutive failed remediations on the same beat, or an
    # explicit MOVE_ON from the student, triggers a concession — the tutor teaches
    # the answer plainly and the lesson advances one step.
    old_count = state.get("remediation_count") or 0
    if response.decision == "REMEDIATE":
        new_count = old_count + 1
    elif response.decision == "NO_ATTEMPT":
        new_count = old_count  # not a failed attempt; doesn't build frustration
    else:
        new_count = 0
    concede = response.decision == "MOVE_ON" or (response.decision == "REMEDIATE" and new_count >= 3)
    if concede:
        new_count = 0

    if response.decision == "ADVANCE" or concede:
        if current_arc:
            if response.decision == "ADVANCE":
                try:
                    cleared = max(1, int(getattr(response, "beats_cleared", 1) or 1))
                except (TypeError, ValueError):
                    cleared = 1
            else:
                cleared = 1  # concessions advance exactly one step
            has_reveal = bool((state.get("arc_term") or "").strip())
            if has_reveal and len(current_arc) > 1:
                # Reveal beat may only be popped when it is itself the current beat.
                cleared = min(cleared, len(current_arc) - 1)
            else:
                cleared = min(cleared, len(current_arc))
            new_arc = current_arc[cleared:]
            if not new_arc and remaining_learning_outcomes:
                new_remaining_learning_outcomes = remaining_learning_outcomes[1:]
                new_arc_outcome = None
        elif remaining_learning_outcomes:
            new_remaining_learning_outcomes = remaining_learning_outcomes[1:]
            new_arc_outcome = None

    justification = response.justification

    # Concession: the tutor gives the answer away — no more questioning on this step.
    if concede:
        justification = (
            "The student is stuck or has asked to move on. Do NOT ask them to try this again. "
            "Warmly and briefly TEACH the answer to what was just being asked — give it away "
            "completely, no quiz. Then continue forward with the next step as usual. "
            f"Context: {justification}"
        )

    # Term secrecy backstop: deterministically scrub the formal term from the
    # monologue until the reveal beat is the tutor's very next move
    # (len(new_arc) == 1 means the reveal beat is now current; term is allowed).
    arc_term = (state.get("arc_term") or "").strip()
    if arc_term and len(new_arc) >= 2:
        justification = _mask_term(justification, arc_term, "this concept")

    # Guarantee a captured student question is answered before anything else,
    # regardless of ADVANCE/REMEDIATE. Python owns this, not the model's prose.
    student_question = (response.student_question or "").strip()
    if student_question:
        justification = (
            f'FIRST: directly and plainly answer the student\'s question: '
            f'"{student_question}". THEN: {justification}'
        )

    print(f"Evaluator Node: Decision - {response.decision}")
    if concede:
        reason = "student requested" if response.decision == "MOVE_ON" else "frustration cap (3 failed remediations)"
        print(f"Evaluator Node: CONCEDING - {reason}")
    if student_question:
        print(f"Evaluator Node: Student Question - {student_question}")
    print(f"Evaluator Node: Justification - {justification}")
    if current_beat:
        if cleared > 1:
            print(f"Evaluator Node: Multi-beat clear - {cleared} beats")
        print(f"Evaluator Node: Beats remaining in arc - {len(new_arc)}")
    print(f"Evaluator Node: Remaining Learning Outcomes - {new_remaining_learning_outcomes}")

    return_payload = {
        "internal_monologue": [justification],
        "remaining_learning_outcomes": new_remaining_learning_outcomes,
        "current_arc": new_arc,
        "arc_outcome": new_arc_outcome,
        "remediation_count": new_count
    }

    # 6. Re-integrated your original logic for completing a topic!
    if not new_remaining_learning_outcomes:
        print(f"Topic Completed: {current_topic}")
        return_payload["completed_topics"] = [current_topic]

    return return_payload
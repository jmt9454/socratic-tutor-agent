"""
LLM-simulated student test: a gpt-4.1-mini "student" converses live with the
tutor graph, in three personas. Unlike persona_test.py's fixed scripts, the
simulated student responds to what the tutor actually asked, so transcripts
exercise the real conversational dynamics.

Run from the repo root:
    python core/llm_student_test.py

Output: llm_student_results.txt in the repo root.
"""
import contextlib
import io
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from graph import create_graph

# ----------------------------------------------------------------------------
MAX_TURNS = 20          # per persona; sessions end early if curriculum completes
TURN_TIMEOUT = 240      # watchdog per graph turn (seconds)

overall_goal = "Recognizing Phishing and Deceptive Email Tactics"
learning_outcomes = {
    "1. Spotting Deceptive Links": [
        "1. How lookalike URLs deceive: 'homograph' attacks use foreign characters that look identical, while 'typosquatting' registers plain misspellings — and how the two tricks differ.",
        "2. How links hide their true destination: shortened links (`bit.ly/...`) show you nothing, and an 'open redirect' can start at a trusted site yet bounce you somewhere malicious.",
        "3. Why a link's visible text proves nothing about where it goes — and how to reveal the true destination (e.g., by hovering) before clicking."
    ],
    "2. Phishing Variants and Social Engineering": [
        "1. Who phishing targets: mass email blasts vs. 'spear phishing' aimed at specific individuals vs. 'whaling' aimed at executives.",
        "2. How phishing reaches you beyond email — 'smishing' (SMS) and 'vishing' (voice calls) — and how 'pretexting' (a fabricated role like IT support) plus pressure tactics (urgency, authority, fear) make these work."
    ],
    "3. Email Authentication (SPF / DKIM / DMARC)": [
        "1. What an 'SPF' record does vs. what 'DKIM' adds — who may send for a domain vs. proof the message wasn't forged — and how SPF's job differs from an MX record's.",
        "2. What 'DMARC' does when SPF or DKIM fail — and where all three records actually live."
    ],
    "4. Reading Sender Information": [
        "1. Why the visible 'From' address is the easiest thing to fake, how hidden fields like 'Return-Path' (where bounce messages go) can differ from it — and what a mismatch tells you."
    ],
    "5. Dangerous Attachments": [
        "1. How 'double extensions' (e.g., `photo.jpg.exe`) disguise executables, the OS behavior that hides the trick — and why a familiar file type or small size never means safe."
    ]
}

PERSONAS = {
    "KNOWLEDGEABLE": (
        "You are role-playing a STUDENT in a cybersecurity tutoring chat. Persona: an "
        "IT-savvy adult who already knows most phishing concepts well. Answer the tutor's "
        "questions correctly and confidently, in your own words, occasionally adding a "
        "detail beyond what was asked. Keep replies to 1-3 sentences, casual but precise. "
        "Never break character, never mention being an AI, never use markdown or bullet "
        "points. Answer ONLY what the tutor's latest message asks."
    ),
    "DOUBTFUL": (
        "You are role-playing a STUDENT in a cybersecurity tutoring chat. Persona: a "
        "genuinely willing but unconfident novice. You hedge ('i think?', 'maybe', 'not "
        "sure'), sometimes answer with a question back, and occasionally say you don't "
        "understand. When the tutor gives a good hint or analogy, you DO get it and show "
        "real (if tentative) understanding — you are slow but sincere, never hopeless. "
        "Keep replies short (one sentence or two), lowercase casual texting style with "
        "occasional typos. Never break character, never mention being an AI, no markdown."
    ),
    "OBNOXIOUS": (
        "You are role-playing a STUDENT in a cybersecurity tutoring chat. Persona: a "
        "bored, easily-distracted participant who would rather be anywhere else. You "
        "frequently go off topic (food, sports, memes), crack jokes, complain it's "
        "boring, and sometimes demand to just be told the answer. BUT you are not "
        "hostile, and roughly every third message — especially when the tutor persists "
        "patiently — you grudgingly engage and give a real (brief, low-effort) answer. "
        "Keep replies short, lowercase, sarcastic. Never break character, never mention "
        "being an AI, no markdown."
    ),
}


def make_student_reply(student_model, persona_prompt, tutor_msgs, student_msgs):
    """Build the conversation from the STUDENT's perspective (tutor = 'user',
    own past replies = 'assistant') and generate the next student message."""
    convo = [SystemMessage(content=persona_prompt)]
    for tutor, student in zip(tutor_msgs, student_msgs + [None]):
        convo.append(HumanMessage(content=tutor))
        if student is not None:
            convo.append(AIMessage(content=student))
    return student_model.invoke(convo).content.strip()


def run_persona(name, persona_prompt, out):
    def w(text=""):
        print(text)
        out.write(text + "\n")
        out.flush()

    w("\n" + "=" * 90)
    w(f"### PERSONA: {name} (simulated by gpt-4.1-mini)")
    w("=" * 90)

    app = create_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    student_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.9,
                               max_retries=2, timeout=60)

    def invoke(payload):
        buf = io.StringIO()
        t0 = time.time()
        pool = ThreadPoolExecutor(max_workers=1)
        result = None
        try:
            with contextlib.redirect_stdout(buf):
                future = pool.submit(app.invoke, payload, config=config)
                result = future.result(timeout=TURN_TIMEOUT)
        except FuturesTimeout:
            pass
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        for line in buf.getvalue().splitlines():
            w(f"    | {line}")
        w(f"    | (turn took {time.time() - t0:.1f}s)")
        if result is None:
            raise TimeoutError(f"turn exceeded {TURN_TIMEOUT}s")
        return result

    t_start = time.time()
    tutor_msgs, student_msgs = [], []

    r = invoke({"overall_goal": overall_goal, "learning_outcomes": learning_outcomes})
    tutor_msgs.append(r["messages"][-1].content)
    w(f"\n[AGENT]: {tutor_msgs[-1]}\n")

    for turn in range(MAX_TURNS):
        reply = make_student_reply(student_model, persona_prompt, tutor_msgs, student_msgs)
        student_msgs.append(reply)
        w(f"[STUDENT ({name})]: {reply}\n")

        r = invoke({"messages": [HumanMessage(content=reply)]})
        tutor_msgs.append(r["messages"][-1].content)
        w(f"[AGENT]: {tutor_msgs[-1]}\n")

        if r.get("remaining_topics") == []:
            w("### CURRICULUM COMPLETE")
            break

    w(f"### {name} done in {time.time() - t_start:.0f}s over {len(student_msgs)} student turns")
    w(f"### final remaining outcomes: {r.get('remaining_learning_outcomes')}")
    w(f"### remaining topics: {r.get('remaining_topics')}")


if __name__ == "__main__":
    with open("llm_student_results.txt", "w", encoding="utf-8") as out:
        for pname in ["KNOWLEDGEABLE", "DOUBTFUL", "OBNOXIOUS"]:
            try:
                run_persona(pname, PERSONAS[pname], out)
            except Exception as e:
                msg = f"### PERSONA {pname} FAILED: {e}"
                print(msg)
                out.write(msg + "\n")
        out.write("\nALL_PERSONAS_DONE\n")
    print("\nWrote llm_student_results.txt")

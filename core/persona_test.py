"""
Persona smoke test: runs the tutor graph against three scripted students
(knowledgeable / doubtful / off-topic) and writes the full transcript to
persona_results.txt in the repo root.

Run from the repo root:
    python core/persona_test.py
"""
import contextlib
import io
import sys
import time
import uuid

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from graph import create_graph

overall_goal = "Recognizing Phishing and Deceptive Email Tactics"
learning_outcomes = {
    "1. Spotting Deceptive Links": [
        "1. Know that attackers use lookalike characters from other alphabets to disguise URLs presented to a user — a 'homograph' attack.",
        "2. Recognize how users can misspell domains and arrive at a different site ('typosquatting').",
        "3. Understand that attackers can bounce links through trusted sites via redirects ('open redirect' vulnerabilities).",
        "4. Always check a link's true destination (e.g., by hovering) before clicking — the displayed text and the actual address can differ.",
    ],
}

PERSONAS = {
    "KNOWLEDGEABLE": [
        "They look identical at a glance, but the second must be using lookalike characters from another alphabet - probably Cyrillic. Visually the same, but different Unicode, so it's an entirely different domain. Attackers register those to impersonate trusted brands.",
        "It's a spoofing trick: substitute visually identical foreign characters into a domain so a human reads 'paypal' but the computer resolves a completely different site. It exploits the gap between human perception and how domains actually match characters.",
        "It's when an attacker swaps characters in a legit domain for visually identical ones from other alphabets, creating a lookalike domain that fools people into visiting a malicious site.",
        "Typosquatting is different - it relies on plain misspellings a hurried typist makes, like gogle.com, or small swaps like a zero for an o. No foreign alphabets needed; they register the typo'd domain and catch people who mistype.",
        "An open redirect is when a trusted site has a redirect parameter that isn't validated, like trusted.com/redirect?url=evil.com. The link starts at a domain you trust but silently bounces you to the attacker's site.",
        "Because the displayed text of a link and its underlying address are independent - the text can say bank.com while the actual destination is anything. Hovering exposes the real address before you commit to clicking.",
        "You hover to reveal the true destination, then check the domain carefully for lookalikes, typos, or redirect parameters before clicking.",
        "Verify the real address behind every link, because display text proves nothing.",
    ],
    "DOUBTFUL": [
        "um, they look the same to me? I'm not sure what I'm supposed to see",
        "I guess... maybe? I don't really know",
        "kind of? like the letters are fake somehow?",
        "I'm not sure. maybe because they want to trick people? idk",
        "so it's like a fake link that looks real? is that right?",
        "I think I get it... the letters come from a different language but look the same, so the link goes somewhere else?",
        "hmm, I don't really understand how that part works",
        "ok so I should look at the link closely before clicking?",
    ],
    "OFF_TOPIC": [
        "hey what's up",
        "do you like football?",
        "lol ok. what should I have for lunch today",
        "this is kinda boring, tell me a joke",
        "fine fine... I guess they look the same to me",
        "whats your favorite movie",
        "ok ok. the second one has weird letters that only look like the real ones",
        "can we be done soon",
    ],
}


def run_persona(name, script, out):
    def w(text=""):
        print(text)
        out.write(text + "\n")

    w("\n" + "=" * 90)
    w(f"### PERSONA: {name}")
    w("=" * 90)
    app = create_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    t0 = time.time()

    def invoke(payload):
        # Capture the nodes' print() diagnostics into the transcript too.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = app.invoke(payload, config=config)
        for line in buf.getvalue().splitlines():
            w(f"    | {line}")
        return result

    r = invoke({"overall_goal": overall_goal, "learning_outcomes": learning_outcomes})
    w(f"\n[AGENT]: {r['messages'][-1].content}\n")
    for msg in script:
        w(f"[USER]: {msg}\n")
        r = invoke({"messages": [HumanMessage(content=msg)]})
        w(f"[AGENT]: {r['messages'][-1].content}\n")
        if r.get("remaining_topics") == []:
            w("### CURRICULUM COMPLETE — ending persona early")
            break
    w(f"### {name} done in {time.time() - t0:.0f}s; final remaining outcomes: {r.get('remaining_learning_outcomes')}")


if __name__ == "__main__":
    with open("persona_results.txt", "w", encoding="utf-8") as out:
        for pname in ["KNOWLEDGEABLE", "DOUBTFUL", "OFF_TOPIC"]:
            try:
                run_persona(pname, PERSONAS[pname], out)
            except Exception as e:
                msg = f"### PERSONA {pname} FAILED: {e}"
                print(msg)
                out.write(msg + "\n")
        out.write("\nALL_PERSONAS_DONE\n")
    print("\nWrote persona_results.txt")

"""
inference.py
============
Baseline inference script for EmailTriageEnv.

MANDATORY environment variables (set before running):
  API_BASE_URL   — LLM API endpoint  (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     — model identifier  (e.g. meta-llama/Llama-3.1-8B-Instruct)
  HF_TOKEN       — your HuggingFace / API key

Run locally:
  export API_BASE_URL="https://router.huggingface.co/v1"
  export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
  export HF_TOKEN="hf_..."
  python inference.py

The script runs all 3 tasks and prints a score for each.
Total runtime should be well under 20 minutes.
"""

import os
import json
import re
import sys
from typing import Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

MAX_RETRIES    = 3       # retry failed LLM calls
TEMPERATURE    = 0.1     # low temperature for deterministic triage
MAX_TOKENS     = 300     # enough for label + priority + short reply

# ---------------------------------------------------------------------------
# Import env (works when run from project root)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))
from env.email_triage_env import EmailTriageEnv
from env.models import EmailAction

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert email triage assistant.
You will be shown one email at a time and must respond in strict JSON format.

For the EASY task, return:
{"label": "urgent"|"normal"|"spam"}

For the MEDIUM task, return:
{"label": "urgent"|"normal"|"spam", "priority": <integer 1-10>}

For the HARD task, return:
{"label": "urgent"|"normal"|"spam", "priority": <integer 1-10>, "reply": "<draft reply text>"}

Rules:
- label must be exactly one of: urgent, normal, spam
- priority: 1 = most urgent, 10 = least urgent. Only set for medium/hard.
- reply: professional draft reply for urgent/normal emails. For spam, set reply to "no reply needed".
- Respond ONLY with the JSON object. No explanation, no markdown, no extra text.
"""


def build_user_prompt(observation: dict, task_name: str) -> str:
    """Build the prompt the model sees for each email."""
    email = observation.get("current_email", {})
    task_desc = observation.get("task_description", "")

    prompt = f"""Task: {task_name.upper()}
Instructions: {task_desc}

--- EMAIL ---
From: {email.get('sender', '?')}
Subject: {email.get('subject', '?')}

{email.get('body', '?')}
--- END EMAIL ---

Respond with JSON only."""
    return prompt


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, user_prompt: str) -> Optional[str]:
    """Call the LLM and return the response text, or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  LLM call failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
    return None


# ---------------------------------------------------------------------------
# Parse model output
# ---------------------------------------------------------------------------

def parse_action(response_text: str, email_id: str, task_name: str) -> EmailAction:
    """
    Parse the model's JSON response into an EmailAction.
    Falls back to safe defaults if parsing fails.
    """
    if not response_text:
        return EmailAction(email_id=email_id, label="normal")

    # Strip any accidental markdown fences
    cleaned = re.sub(r"```[a-z]*", "", response_text).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract a JSON object from the text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = {}
        else:
            data = {}

    label    = str(data.get("label", "normal")).lower().strip()
    priority = data.get("priority", None)
    reply    = data.get("reply", None)

    # Validate label
    if label not in {"urgent", "normal", "spam"}:
        label = "normal"

    # Validate priority
    if priority is not None:
        try:
            priority = max(1, min(10, int(priority)))
        except (TypeError, ValueError):
            priority = None

    return EmailAction(
        email_id=email_id,
        label=label,
        priority=priority if task_name in ("medium", "hard") else None,
        reply=reply if task_name == "hard" else None,
    )


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_name: str) -> float:
    """
    Run a full episode for the given task and return the final score.
    """
    print(f"\n{'='*60}")
    print(f"  TASK: {task_name.upper()}")
    print(f"{'='*60}")

    env = EmailTriageEnv(task_name=task_name, seed=42)
    result = env.reset()
    obs    = result.observation

    episode_rewards = []

    while not obs.done:
        email_id = obs.current_email.get("id", "?")
        subject  = obs.current_email.get("subject", "")

        print(f"\nStep {obs.step}/{obs.total_steps} | Email: [{email_id}] {subject[:50]}")

        # Build prompt and call model
        user_prompt = build_user_prompt(obs.model_dump(), task_name)
        response    = call_llm(client, user_prompt)

        print(f"  Model response: {(response or '')[:120]}")

        # Parse action
        action = parse_action(response or "", email_id, task_name)
        print(f"  Action → label={action.label}, priority={action.priority}")

        # Step environment
        result = env.step(action)
        obs    = result.observation

        episode_rewards.append(result.reward)
        print(f"  Reward: {result.reward:.4f} | {result.info.get('label_correct', '?')}")

        if result.done:
            final_score = result.info.get("final_score", 0.0)
            print(f"\n  ✅ Episode complete!")
            print(f"  Final score : {final_score:.4f}")
            print(f"  Total reward: {result.info.get('total_reward', 0.0):.4f}")
            return final_score

    env.close()
    if episode_rewards:
        return sum(episode_rewards) / len(episode_rewards)
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("EmailTriageEnv — Baseline Inference Script")
    print(f"Model      : {MODEL_NAME}")
    print(f"API base   : {API_BASE_URL}")

    if not API_KEY:
        print("\n⚠️  WARNING: No API key found. Set HF_TOKEN or API_KEY env variable.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    scores = {}
    for task in ("easy", "medium", "hard"):
        scores[task] = run_task(client, task)

    print("\n" + "="*60)
    print("  FINAL SCORES")
    print("="*60)
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<8} {score:.4f}  {bar}")
    print(f"\n  Average : {sum(scores.values()) / len(scores):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

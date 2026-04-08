"""
tasks.py
--------
Defines the three tasks for EmailTriageEnv.

Task 1 — EASY   : Label emails as urgent / normal / spam
Task 2 — MEDIUM : Label + assign a priority rank (1–10)
Task 3 — HARD   : Label + priority + write a draft reply

Each task has:
  - name            : str
  - description     : str shown to the agent
  - grade(action, email) -> (reward: float, info: dict)
"""

from typing import Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_LABELS = {"urgent", "normal", "spam"}


def _label_score(predicted: str, ground_truth: str) -> float:
    """Returns 1.0 for correct label, 0.0 for wrong label."""
    return 1.0 if predicted.lower().strip() == ground_truth else 0.0


def _priority_score(predicted: int, ground_truth: int) -> float:
    """
    Partial credit for priority.
    Exact match → 1.0
    Off by 1   → 0.7
    Off by 2   → 0.4
    Off by 3+  → 0.0
    """
    diff = abs(predicted - ground_truth)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.7
    elif diff == 2:
        return 0.4
    else:
        return 0.0


def _reply_score(reply: str, keywords: list) -> float:
    """
    Scores a draft reply based on:
      - Minimum length (at least 10 words)           → 0.2 base
      - Keyword coverage (up to 5 relevant words)    → 0.6 max
      - Polite greeting/sign-off present             → 0.2 bonus
    Returns a float in [0.0, 1.0].
    """
    if not reply or not reply.strip():
        return 0.0

    score = 0.0
    reply_lower = reply.lower()

    # Base score for minimum length
    word_count = len(reply.split())
    if word_count >= 10:
        score += 0.2

    # Keyword coverage
    if keywords:
        matched = sum(1 for kw in keywords if kw.lower() in reply_lower)
        keyword_ratio = matched / len(keywords)
        score += keyword_ratio * 0.6
    else:
        # Spam email — agent should NOT reply; penalise if they do
        score = 0.0
        return score

    # Politeness check
    polite_words = ["hi", "hello", "dear", "thanks", "thank you",
                    "regards", "best", "sincerely", "cheers", "apolog"]
    if any(pw in reply_lower for pw in polite_words):
        score += 0.2

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Task 1: EASY
# ---------------------------------------------------------------------------

class EasyTask:
    name = "easy"
    description = (
        "Label each email as exactly one of: 'urgent', 'normal', or 'spam'. "
        "Set the label field in your action. "
        "You do not need to set priority or reply for this task."
    )

    @staticmethod
    def grade(action, email: dict) -> Tuple[float, Dict[str, Any]]:
        """
        Grade an action against the ground-truth email.
        Returns (reward, info_dict).
        Reward is purely based on correct label.
        """
        predicted_label = action.label.lower().strip()

        if predicted_label not in VALID_LABELS:
            return 0.0, {
                "error": f"Invalid label '{action.label}'. Must be urgent/normal/spam.",
                "label_correct": False,
            }

        label_reward = _label_score(predicted_label, email["label"])

        info = {
            "label_predicted": predicted_label,
            "label_ground_truth": email["label"],
            "label_correct": label_reward == 1.0,
            "reward_breakdown": {"label": label_reward},
        }
        return round(label_reward, 4), info


# ---------------------------------------------------------------------------
# Task 2: MEDIUM
# ---------------------------------------------------------------------------

class MediumTask:
    name = "medium"
    description = (
        "Label each email as 'urgent', 'normal', or 'spam', AND assign a priority rank. "
        "Priority is an integer from 1 (most urgent) to 10 (least urgent). "
        "Set both the label and priority fields in your action."
    )

    @staticmethod
    def grade(action, email: dict) -> Tuple[float, Dict[str, Any]]:
        """
        Reward = 0.6 * label_score + 0.4 * priority_score
        """
        predicted_label = action.label.lower().strip()

        if predicted_label not in VALID_LABELS:
            return 0.0, {"error": f"Invalid label '{action.label}'."}

        if action.priority is None:
            # Missing priority — only get partial credit for label
            label_reward = _label_score(predicted_label, email["label"])
            return round(label_reward * 0.6, 4), {
                "error": "Priority not provided. Only label scored.",
                "label_correct": label_reward == 1.0,
                "reward_breakdown": {"label": label_reward * 0.6, "priority": 0.0},
            }

        label_reward = _label_score(predicted_label, email["label"])
        priority_reward = _priority_score(action.priority, email["priority"])

        total = 0.6 * label_reward + 0.4 * priority_reward

        info = {
            "label_predicted": predicted_label,
            "label_ground_truth": email["label"],
            "label_correct": label_reward == 1.0,
            "priority_predicted": action.priority,
            "priority_ground_truth": email["priority"],
            "reward_breakdown": {
                "label": round(0.6 * label_reward, 4),
                "priority": round(0.4 * priority_reward, 4),
            },
        }
        return round(total, 4), info


# ---------------------------------------------------------------------------
# Task 3: HARD
# ---------------------------------------------------------------------------

class HardTask:
    name = "hard"
    description = (
        "Label each email as 'urgent', 'normal', or 'spam'. "
        "Assign a priority rank (1 = most urgent, 10 = least urgent). "
        "Write a professional draft reply in the reply field. "
        "For spam emails, leave reply empty or write 'no reply needed'. "
        "Your reply will be scored on relevance, keyword coverage, and politeness."
    )

    @staticmethod
    def grade(action, email: dict) -> Tuple[float, Dict[str, Any]]:
        """
        Reward = 0.4 * label_score + 0.3 * priority_score + 0.3 * reply_score
        """
        predicted_label = action.label.lower().strip()

        if predicted_label not in VALID_LABELS:
            return 0.0, {"error": f"Invalid label '{action.label}'."}

        label_reward = _label_score(predicted_label, email["label"])

        priority_reward = 0.0
        if action.priority is not None:
            priority_reward = _priority_score(action.priority, email["priority"])

        # For spam, the correct reply is no reply
        if email["label"] == "spam":
            reply_text = action.reply or ""
            no_reply = (
                reply_text.strip() == ""
                or "no reply" in reply_text.lower()
                or "spam" in reply_text.lower()
            )
            reply_reward = 1.0 if no_reply else 0.0
        else:
            reply_reward = _reply_score(action.reply or "", email["reply_keywords"])

        total = 0.4 * label_reward + 0.3 * priority_reward + 0.3 * reply_reward

        info = {
            "label_predicted": predicted_label,
            "label_ground_truth": email["label"],
            "label_correct": label_reward == 1.0,
            "priority_predicted": action.priority,
            "priority_ground_truth": email["priority"],
            "reply_score": round(reply_reward, 4),
            "reward_breakdown": {
                "label": round(0.4 * label_reward, 4),
                "priority": round(0.3 * priority_reward, 4),
                "reply": round(0.3 * reply_reward, 4),
            },
        }
        return round(total, 4), info


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}

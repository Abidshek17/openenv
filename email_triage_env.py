"""
email_triage_env.py
-------------------
The main EmailTriageEnv class.

Implements the full OpenEnv interface:
  - reset()        → returns initial StepResult
  - step(action)   → returns StepResult
  - state()        → returns current state dict
  - close()        → cleanup (no-op here)

Usage
-----
    from env.email_triage_env import EmailTriageEnv
    from env.models import EmailAction

    env = EmailTriageEnv(task_name="easy")
    result = env.reset()

    action = EmailAction(
        email_id=result.observation.current_email["id"],
        label="urgent",
    )
    result = env.step(action)
"""

import random
from copy import deepcopy
from typing import Optional

from env.models import EmailAction, EmailObservation, StepResult
from env.tasks import TASKS
from data.emails import EMAILS


class EmailTriageEnv:
    """
    Email Triage and Prioritization environment.

    The agent is presented with one email per step and must classify,
    prioritize, and (for the hard task) draft a reply to each one.

    Parameters
    ----------
    task_name : str
        One of "easy", "medium", "hard". Default is "easy".
    seed : int, optional
        Random seed for reproducible episode ordering.
    """

    METADATA = {
        "name": "EmailTriageEnv",
        "version": "1.0.0",
        "description": (
            "An AI agent works through a realistic office inbox, "
            "classifying each email as urgent/normal/spam, assigning priority, "
            "and drafting replies. Three difficulty levels."
        ),
        "tasks": list(TASKS.keys()),
        "action_space": "EmailAction (label, priority, reply)",
        "observation_space": "EmailObservation",
        "reward_range": [0.0, 1.0],
    }

    def __init__(self, task_name: str = "easy", seed: Optional[int] = 42):
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from: {list(TASKS.keys())}"
            )

        self.task_name = task_name
        self.task = TASKS[task_name]
        self.seed = seed

        # Episode state (set during reset)
        self._email_queue = []
        self._current_index = 0
        self._cumulative_reward = 0.0
        self._step_count = 0
        self._done = False
        self._last_action_result = ""

    # ------------------------------------------------------------------
    # reset()  — start a new episode
    # ------------------------------------------------------------------

    def reset(self) -> StepResult:
        """
        Start a new episode.

        Shuffles the email inbox and returns the first observation.
        """
        rng = random.Random(self.seed)
        self._email_queue = deepcopy(EMAILS)
        rng.shuffle(self._email_queue)

        self._current_index = 0
        self._cumulative_reward = 0.0
        self._step_count = 0
        self._done = False
        self._last_action_result = ""

        observation = self._build_observation()
        return StepResult(
            observation=observation,
            reward=0.0,
            done=False,
            info={"message": "Episode started. Good luck!"},
        )

    # ------------------------------------------------------------------
    # step(action)  — advance one step
    # ------------------------------------------------------------------

    def step(self, action: EmailAction) -> StepResult:
        """
        Process one action from the agent.

        Parameters
        ----------
        action : EmailAction
            The agent's labelling/prioritization/reply for the current email.

        Returns
        -------
        StepResult with observation, reward, done, info.
        """
        if self._done:
            # Episode already ended — return terminal observation
            obs = self._build_observation()
            obs.done = True
            return StepResult(
                observation=obs,
                reward=0.0,
                done=True,
                info={"warning": "Episode already done. Call reset() to start a new one."},
            )

        current_email = self._email_queue[self._current_index]

        # Validate email_id matches
        if action.email_id != current_email["id"]:
            self._last_action_result = (
                f"Wrong email_id '{action.email_id}'. "
                f"Expected '{current_email['id']}'. Giving 0 reward for this step."
            )
            reward = 0.0
            info = {"error": self._last_action_result}
        else:
            # Grade the action
            reward, info = self.task.grade(action, current_email)
            self._last_action_result = self._format_result(reward, info)

        self._cumulative_reward += reward
        self._step_count += 1
        self._current_index += 1

        # Check if episode is done
        done = self._current_index >= len(self._email_queue)
        self._done = done

        if done:
            # Build final observation
            final_score = self._cumulative_reward / len(self._email_queue)
            obs = EmailObservation(
                step=self._step_count,
                total_steps=len(self._email_queue),
                task_name=self.task_name,
                task_description=self.task.description,
                current_email={},
                emails_remaining=0,
                cumulative_reward=round(self._cumulative_reward, 4),
                last_action_result=self._last_action_result,
                done=True,
            )
            info["episode_complete"] = True
            info["final_score"] = round(final_score, 4)
            info["total_reward"] = round(self._cumulative_reward, 4)
            info["emails_processed"] = self._step_count
        else:
            obs = self._build_observation()

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=done,
            info=info,
        )

    # ------------------------------------------------------------------
    # state()  — return current internal state (for debugging/logging)
    # ------------------------------------------------------------------

    def state(self) -> dict:
        """
        Returns the current state of the environment as a plain dict.
        Useful for logging, checkpointing, and debugging.
        """
        return {
            "task_name": self.task_name,
            "step": self._step_count,
            "total_emails": len(self._email_queue),
            "current_index": self._current_index,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "done": self._done,
            "seed": self.seed,
        }

    # ------------------------------------------------------------------
    # close()  — cleanup
    # ------------------------------------------------------------------

    def close(self):
        """No resources to release, but required by OpenEnv spec."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> EmailObservation:
        """Build the observation from current state."""
        if self._current_index >= len(self._email_queue):
            # Terminal state
            return EmailObservation(
                step=self._step_count,
                total_steps=len(self._email_queue),
                task_name=self.task_name,
                task_description=self.task.description,
                current_email={},
                emails_remaining=0,
                cumulative_reward=round(self._cumulative_reward, 4),
                last_action_result=self._last_action_result,
                done=True,
            )

        email = self._email_queue[self._current_index]
        # Only expose fields the agent should see (no ground-truth labels)
        visible_email = {
            "id": email["id"],
            "subject": email["subject"],
            "sender": email["sender"],
            "body": email["body"],
        }

        return EmailObservation(
            step=self._step_count + 1,
            total_steps=len(self._email_queue),
            task_name=self.task_name,
            task_description=self.task.description,
            current_email=visible_email,
            emails_remaining=len(self._email_queue) - self._current_index - 1,
            cumulative_reward=round(self._cumulative_reward, 4),
            last_action_result=self._last_action_result,
            done=False,
        )

    def _format_result(self, reward: float, info: dict) -> str:
        """Format a human-readable result string for the agent."""
        breakdown = info.get("reward_breakdown", {})
        parts = [f"Step reward: {reward:.2f}"]
        if "label_correct" in info:
            parts.append(
                f"Label: {'✓' if info['label_correct'] else '✗'} "
                f"(predicted={info.get('label_predicted','?')}, "
                f"truth={info.get('label_ground_truth','?')})"
            )
        if breakdown:
            detail = ", ".join(f"{k}={v:.2f}" for k, v in breakdown.items())
            parts.append(f"Breakdown: [{detail}]")
        if "error" in info:
            parts.append(f"Error: {info['error']}")
        return " | ".join(parts)

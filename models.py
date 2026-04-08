"""
models.py
---------
Typed Pydantic models for the EmailTriageEnv.

OpenEnv requires:
  - Observation  : what the agent sees each step
  - Action       : what the agent can do
  - Reward       : numeric signal returned after each step
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# OBSERVATION  — what the agent sees
# ---------------------------------------------------------------------------

class EmailObservation(BaseModel):
    """
    Observation returned to the agent at every step.

    Fields
    ------
    step : int
        Current step number (starts at 1).
    total_steps : int
        Maximum steps allowed in this episode.
    task_name : str
        Which task is active: "easy", "medium", or "hard".
    task_description : str
        Plain-English description of what the agent must do.
    current_email : dict
        The email the agent must act on right now.
        Keys: id, subject, sender, body.
    emails_remaining : int
        How many emails are left in the inbox after this one.
    cumulative_reward : float
        Total reward accumulated so far this episode.
    last_action_result : str
        Feedback on the previous action (empty on first step).
    done : bool
        True when the episode has ended.
    """

    step: int = Field(..., description="Current step number")
    total_steps: int = Field(..., description="Max steps in this episode")
    task_name: str = Field(..., description="Task difficulty: easy | medium | hard")
    task_description: str = Field(..., description="What the agent must accomplish")
    current_email: dict = Field(..., description="The email to act on (id, subject, sender, body)")
    emails_remaining: int = Field(..., description="Emails left after this one")
    cumulative_reward: float = Field(0.0, description="Reward accumulated so far")
    last_action_result: str = Field("", description="Feedback on the last action")
    done: bool = Field(False, description="True when episode is over")


# ---------------------------------------------------------------------------
# ACTION  — what the agent can do
# ---------------------------------------------------------------------------

class EmailAction(BaseModel):
    """
    Action submitted by the agent each step.

    Fields
    ------
    email_id : str
        The ID of the email being acted on (must match current_email.id).
    label : str
        Classification: "urgent" | "normal" | "spam"
    priority : Optional[int]
        Required for medium/hard tasks.
        Integer 1–10, where 1 = most urgent.
    reply : Optional[str]
        Required for hard task only.
        Draft reply text (at least 10 words expected).
    """

    email_id: str = Field(..., description="ID of the email being acted on")
    label: str = Field(
        ...,
        description="Email classification: 'urgent' | 'normal' | 'spam'",
    )
    priority: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Priority rank 1 (most urgent) to 10 (least urgent). Required for medium/hard.",
    )
    reply: Optional[str] = Field(
        None,
        description="Draft reply text. Required for hard task. Ignored for easy/medium.",
    )


# ---------------------------------------------------------------------------
# STEP RESULT  — what env.step() returns
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """
    Returned by env.step(action).

    Fields
    ------
    observation : EmailObservation
    reward      : float   — reward for this single step (0.0 – 1.0)
    done        : bool    — True if the episode is finished
    info        : dict    — extra diagnostics (grader breakdown, etc.)
    """

    observation: EmailObservation
    reward: float = Field(..., description="Per-step reward (0.0 – 1.0)")
    done: bool = Field(..., description="True when episode is complete")
    info: dict = Field(default_factory=dict, description="Grader breakdown and diagnostics")

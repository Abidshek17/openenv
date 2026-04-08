"""
server.py
---------
FastAPI HTTP server that exposes EmailTriageEnv via REST endpoints.

Endpoints
---------
GET  /              — health check
POST /reset         — start a new episode
POST /step          — submit an action
GET  /state         — get current state
GET  /tasks         — list available tasks

Run with:
    uvicorn server:app --host 0.0.0.0 --port 7860
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import logging

from env.email_triage_env import EmailTriageEnv
from env.models import EmailAction

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EmailTriageEnv",
    description=(
        "An OpenEnv environment for email triage and prioritization. "
        "An AI agent classifies emails, assigns priority, and drafts replies."
    ),
    version="1.0.0",
)

# One environment instance per server (stateful, single-user for HF Spaces)
_envs: dict[str, EmailTriageEnv] = {}


def get_or_create_env(task_name: str = "easy") -> EmailTriageEnv:
    if task_name not in _envs:
        _envs[task_name] = EmailTriageEnv(task_name=task_name)
    return _envs[task_name]


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "easy"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    task_name: str = "easy"
    email_id: str
    label: str
    priority: Optional[int] = None
    reply: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    """Health check — returns 200 if server is running."""
    return {"status": "ok", "env": "EmailTriageEnv", "version": "1.0.0"}


@app.post("/reset")
def reset(request: ResetRequest):
    """
    Start a new episode.

    Body: { "task_name": "easy"|"medium"|"hard", "seed": 42 }
    Returns: initial observation and step result.
    """
    try:
        env = EmailTriageEnv(task_name=request.task_name, seed=request.seed)
        _envs[request.task_name] = env
        result = env.reset()
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    """
    Submit one action and advance the environment.

    Body: { "task_name": "easy", "email_id": "e001", "label": "urgent", ... }
    Returns: observation, reward, done, info.
    """
    env = _envs.get(request.task_name)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active episode for task '{request.task_name}'. Call /reset first.",
        )

    action = EmailAction(
        email_id=request.email_id,
        label=request.label,
        priority=request.priority,
        reply=request.reply,
    )

    try:
        result = env.step(action)
        return result.model_dump()
    except Exception as e:
        logger.error(f"step() error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state(task_name: str = "easy"):
    """Get the current internal state of the environment."""
    env = _envs.get(task_name)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active episode for task '{task_name}'. Call /reset first.",
        )
    return env.state()


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    from env.tasks import TASKS
    return {
        name: {"name": name, "description": cls.description}
        for name, cls in TASKS.items()
    }

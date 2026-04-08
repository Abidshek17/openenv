# 📧 EmailTriageEnv

**A real-world OpenEnv environment for email triage and prioritization.**

An AI agent works through a realistic office inbox and must classify each email as `urgent`, `normal`, or `spam`, assign a priority rank, and draft professional replies — just like a real executive assistant.

---

## 🌍 Why This Environment?

Email triage is one of the most universal knowledge-worker tasks. It requires:
- **NLU** — understanding tone, urgency, and context
- **Judgment** — distinguishing what's genuinely critical vs noise
- **Communication** — writing appropriate, professional replies

This makes it an excellent benchmark for evaluating real-world AI agent capabilities.

---

## 🎯 Tasks

| Task | Difficulty | What the agent must do | Score weights |
|------|-----------|------------------------|---------------|
| `easy` | ⭐ Easy | Label each email: `urgent` / `normal` / `spam` | 100% label |
| `medium` | ⭐⭐ Medium | Label + assign priority rank (1–10) | 60% label + 40% priority |
| `hard` | ⭐⭐⭐ Hard | Label + priority + draft a reply | 40% label + 30% priority + 30% reply |

All scores are in the range **[0.0, 1.0]**.

### Grader details

**Label scoring:** Exact match — `1.0` for correct, `0.0` for wrong.

**Priority scoring (partial credit):**
- Exact match → `1.0`
- Off by 1 → `0.7`
- Off by 2 → `0.4`
- Off by 3+ → `0.0`

**Reply scoring:**
- Base score for ≥10 word reply → `0.2`
- Keyword coverage (domain-relevant terms) → up to `0.6`
- Politeness/greeting present → `0.2`
- Spam emails: correct answer is **no reply** → `1.0` if left empty

---

## 🔧 Action Space

```python
class EmailAction(BaseModel):
    email_id: str          # ID of the email being acted on (required)
    label: str             # "urgent" | "normal" | "spam" (required)
    priority: int | None   # 1 (most urgent) to 10 (least urgent) — medium/hard
    reply: str | None      # Draft reply text — hard task only
```

## 👁️ Observation Space

```python
class EmailObservation(BaseModel):
    step: int
    total_steps: int
    task_name: str
    task_description: str
    current_email: dict         # {id, subject, sender, body}
    emails_remaining: int
    cumulative_reward: float
    last_action_result: str     # feedback on previous action
    done: bool
```

---

## 🚀 Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the server

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Use the API

```bash
# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy", "seed": 42}'

# Submit an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy", "email_id": "e001", "label": "urgent"}'

# Check current state
curl http://localhost:7860/state?task_name=easy
```

### Run the baseline inference script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"

python inference.py
```

---

## 🐳 Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  email-triage-env
```

---

## 📁 Project Structure

```
email_triage_env/
├── server.py              ← FastAPI HTTP server (OpenEnv endpoints)
├── inference.py           ← Baseline agent script (uses OpenAI client)
├── openenv.yaml           ← OpenEnv metadata spec
├── requirements.txt
├── Dockerfile
├── env/
│   ├── __init__.py
│   ├── email_triage_env.py ← Main environment class
│   ├── models.py           ← Pydantic models (Observation, Action, StepResult)
│   └── tasks.py            ← Task definitions + graders (Easy/Medium/Hard)
└── data/
    ├── __init__.py
    └── emails.py           ← 12 realistic emails with ground-truth labels
```

---

## 📊 Expected Baseline Scores

Scores from running `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Router:

| Task | Expected Score |
|------|---------------|
| easy | ~0.75 – 0.85 |
| medium | ~0.55 – 0.70 |
| hard | ~0.40 – 0.60 |

Frontier models (GPT-4, Claude 3.5 Sonnet) are expected to score:

| Task | Expected Score |
|------|---------------|
| easy | ~0.90 – 1.00 |
| medium | ~0.75 – 0.90 |
| hard | ~0.65 – 0.80 |

---

## 📋 Pre-Submission Checklist

- [x] HF Space deploys and responds to `/reset`
- [x] `openenv.yaml` present with typed models
- [x] `step()` / `reset()` / `state()` endpoints implemented
- [x] 3 tasks with difficulty range (easy → medium → hard)
- [x] Graders deterministic and reproducible
- [x] Scores in [0.0, 1.0]
- [x] `inference.py` uses OpenAI client with env variables
- [x] Dockerfile builds and runs
- [x] README with full documentation

---

## License

MIT

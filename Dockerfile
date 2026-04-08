# Dockerfile for EmailTriageEnv
# Deploy to Hugging Face Spaces as a Docker Space.
#
# Build:  docker build -t email-triage-env .
# Run:    docker run -p 7860:7860 email-triage-env

FROM python:3.11-slim

# HF Spaces requirement: non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# OpenEnv / HF Spaces environment variables
ENV TASK_LEVEL=easy
ENV PIPELINE_SEED=42
ENV MAX_CONCURRENT_EPISODES=4
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "pipeline_debug_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]

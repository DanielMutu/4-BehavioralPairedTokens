# Behavioral Paired Tokens — CPU/debug image.
# For GPU runs on the GTX 970 (sm_52) install torch cu121 wheels on the host
# or swap the base image for pytorch/pytorch:*-cuda12.1-cudnn8-runtime.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TZ=Europe/Rome

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# debug-mode sanity run by default (100 examples, 2 epochs, CPU)
CMD ["python", "-m", "src.train", "--debug"]

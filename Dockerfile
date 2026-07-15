# Behavioral Paired Tokens — reproducible CPU image (debug/CI).
# The environment is defined by uv.lock; do not install from loose ranges.
# For GPU runs, build a variant image on a CUDA base and re-lock explicitly.
FROM python:3.11-slim@sha256:baf89808ec37adeaab83cec287adb4a2afa4a11c1d51e961c7ec737877e61af6

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TZ=Europe/Rome \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv

# Lockfile-first layer: dependency install is cached independently of code.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --dev

COPY . .
RUN uv sync --frozen --dev

ENV PATH="/opt/venv/bin:${PATH}"

# Default: validation, not training. Training is an explicit command:
#   docker run --rm <image> python -m src.train --debug
CMD ["python", "-m", "pytest", "-q"]

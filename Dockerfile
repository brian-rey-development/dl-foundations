FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy FOUNDATIONS_MODEL_PATH=/app/outputs/model.npz

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "foundations.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

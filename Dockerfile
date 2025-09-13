FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV WARP_LOG_LEVEL=info
ENV WARP_ACCESS_LOG=true
ENV OPENAI_LOG_LEVEL=info
ENV OPENAI_ACCESS_LOG=true
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .
CMD ["uv", "run", "./start.py"]
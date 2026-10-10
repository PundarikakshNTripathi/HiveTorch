FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Dependencies first (cached layer). The CPU torch index is pinned in pyproject.toml.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen --no-install-project

COPY . .

# The code imports `src.fedavg_core...`, so the project root must be on the path.
ENV PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 50051 8000
ENTRYPOINT ["python", "main.py"]
CMD ["simulate"]

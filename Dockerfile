FROM node:24-alpine AS dashboard
WORKDIR /ui
COPY dashboard/package*.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY configs ./configs
COPY src ./src
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev
COPY --from=dashboard /ui/dist ./dashboard/dist
RUN useradd --create-home app && mkdir -p /app/data /app/runtime && chown -R app:app /app
USER app
EXPOSE 8000
ENTRYPOINT ["/app/.venv/bin/cardiotwin"]
CMD ["serve", "--host", "0.0.0.0", "--data-dir", "/app/data/demo", "--db", "/app/runtime/state.sqlite"]

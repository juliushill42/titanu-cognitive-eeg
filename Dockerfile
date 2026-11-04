FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 symbiosis

FROM base AS runtime
COPY --chown=symbiosis:symbiosis src/ /app/src/
COPY --chown=symbiosis:symbiosis templates/ /app/templates/
RUN mkdir -p /app/data && chown -R symbiosis:symbiosis /app
USER symbiosis
EXPOSE 8100
HEALTHCHECK --interval=15s --timeout=5s --start-period=12s --retries=3 \
  CMD curl -sf http://127.0.0.1:8100/health || exit 1
ENV COG_HOST=0.0.0.0 \
    COG_PORT=8100 \
    COG_REQUIRE_AUTH=true \
    COG_DATA_DIR=/app/data \
    COG_TICK_INTERVAL_SEC=1.2 \
    COG_CHANNELS=1024 \
    COG_NEURAL_BACKEND=simulator \
    COG_MODEL_BACKEND=simulator
CMD ["python", "-m", "src.cognitive.main"]

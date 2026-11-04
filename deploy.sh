#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "════════════════════════════════════════════════════════════"
echo "  cognitive.sh v2 — Cognitive Bandwidth Multiplier"
echo "  Hardened Brain-AI Symbiosis Layer"
echo "════════════════════════════════════════════════════════════"
echo

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COG_PORT="${COG_PORT:-8100}"
export COG_HOST="${COG_HOST:-127.0.0.1}"
export COG_REQUIRE_AUTH="${COG_REQUIRE_AUTH:-true}"
export COG_DATA_DIR="${COG_DATA_DIR:-$ROOT/data}"

mkdir -p "$COG_DATA_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "→ Docker mode"
  docker compose down --remove-orphans 2>/dev/null || true
  docker compose build --quiet
  docker compose up -d
  echo "Waiting for health..."
  for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${COG_PORT}/health" >/dev/null 2>&1; then
      echo
      echo "✓ Online  http://127.0.0.1:${COG_PORT}/dashboard"
      echo "  Logs: docker compose logs -f"
      echo "  API key is printed in container logs on first start if generated."
      exit 0
    fi
    sleep 1
    printf "."
  done
  docker compose logs --tail=40
  exit 1
fi

echo "→ Local Python mode on ${COG_HOST}:${COG_PORT}"
exec python3 -m src.cognitive.main

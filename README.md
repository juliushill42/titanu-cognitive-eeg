# cognitive.sh v2 — Cognitive Bandwidth Multiplier

**Hardened Brain-AI Symbiosis Layer**

Addresses the full audit surface: authentication, validation, thread safety, persistent memory, XSS-safe UI, pluggable neural/model adapters, localhost-default bind, stronger CI.

## Quick Start

```bash
cp .env.example .env   # set COG_API_KEY
chmod +x deploy.sh && ./deploy.sh
```

Dashboard: http://127.0.0.1:8100/dashboard  
Paste your API key in the Auth box (or set `Authorization: Bearer …`).

## Audit fixes (v1 → v2)

| Finding | Resolution |
|---------|------------|
| Synthetic BCI only | `NeuralInputAdapter` + `simulator` / `file` backends |
| Simulated model only | `ModelClient` + `simulator` / `http` (OpenAI-compatible) |
| Unauthenticated 0.0.0.0 API | Default `COG_HOST=127.0.0.1`, `COG_REQUIRE_AUTH=true`, Bearer / X-API-Key |
| Dashboard XSS risk | All dynamic text HTML-escaped client-side |
| Ephemeral memory | SQLite `PersistentMemory` under `COG_DATA_DIR` |
| Unsynchronized shared state | `threading.RLock` on engine |
| No body validation | Max size + JSON type checks |
| Hardcoded Compose port | `${COG_PORT:-8100}` |
| Smoke-only CI | Unit tests + live 401/200 auth checks |

## Configuration

See `.env.example`. Important:

- `COG_API_KEY` — required when auth enabled
- `COG_NEURAL_BACKEND=simulator|file`
- `COG_MODEL_BACKEND=simulator|http` + `COG_MODEL_URL` for real models

## API

All `/api/*` routes require auth when `COG_REQUIRE_AUTH=true`.

- `GET /health` — public
- `GET /api/v1/state|attention|memory|events`
- `POST /api/v1/intent` `{"intent":"focus_narrow","call_ai":true}`
- `POST /api/v1/recall` `{"query":"project goals"}`

## Tests

```bash
PYTHONPATH=. python tests/test_smoke.py
```

## License

MIT


## Milestone 3 — EEG → state → adaptive model → measured gain

Pipeline:

1. **EEG window** (`EEGSample`) — real samples or `synthesize_eeg(label)` for CI
2. **Band powers** — delta/theta/alpha/beta/gamma via Goertzel (stdlib)
3. **CognitiveState** — load, focus, engagement, fatigue, mode, confidence
4. **ModelPolicy** — temperature, max_tokens, system prefix, retrieval depth, style
5. **A/B eval** — adaptive vs fixed baseline on labeled tasks

```bash
PYTHONPATH=. python -m src.cognitive.eval_harness
```

`COG_NEURAL_BACKEND=eeg` (default) uses the EEG path.  
Hardware bridge: push `EEGSample` into `EEGNeuralAdapter.push_sample()` from an LSL/BrainFlow process.

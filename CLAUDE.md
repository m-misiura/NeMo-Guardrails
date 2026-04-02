# CLAUDE.md - NVIDIA NeMo Guardrails (TrustyAI fork)

## Overview

NVIDIA NeMo Guardrails — toolkit for adding programmable guardrails to LLM apps.
This is the **trustyai-fork** (`trustyai-explainability/NeMo-Guardrails`) with Red Hat / TrustyAI customisations.

## Repository layout

```
nemoguardrails/              # main package
  rails/llm/llmrails.py      # LLMRails — core engine
  rails/llm/config.py        # RailsConfig
  server/api.py              # FastAPI server (+ fork-specific /v1/guardrail/checks endpoint)
  actions/llm/utils.py       # llm_call(), header forwarding
  context.py                 # context vars (request headers, runtime auth registry)
  logging/callbacks.py       # LangChain callback handler (invocation param logging)
  library/                   # built-in rails (30+ guardrail types)
  colang/v1_0/               # Colang v1
  colang/v2_x/               # Colang v2
tests/                       # pytest tests
scripts/                     # entrypoint, filter_guardrails, model discovery
Dockerfile.server            # UBI9 multi-stage build (fork-specific)
```

## Git remotes

- `upstream` — NVIDIA/NeMo-Guardrails (original)
- `trustyai-fork` — trustyai-explainability/NeMo-Guardrails
- `midstream` — opendatahub-io/NeMo-Guardrails
- `downstream` — red-hat-data-services/NeMo-Guardrails

Default branch with the latest changes: `develop`

## Build and install python package

```bash
# Install with Poetry (Python 3.10-3.13)
poetry install --all-extras
```

## Running the server locally

```bash
# Start with a config directory (each subdirectory is a config_id)
nemoguardrails server --config /path/to/configs --port 8000

# With a default config and verbose logging
nemoguardrails server --config /path/to/configs --default-config-id my-config --verbose
```

## Running tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_header_forwarding.py -v

# Fork-specific tests
pytest tests/test_guardrail_checks_api.py -v
```

## CI

PR workflows (`.github/workflows/pr-tests.yml` → `_test.yml`):

1. `poetry check --lock` — lock file must be in sync with `pyproject.toml`
2. `pytest` across Python 3.10, 3.11, 3.12, 3.13 (coverage on 3.11)

Fork-specific (`security.yml`): Trivy + Bandit scans on PRs and pushes to `develop`.

## Key fork changes

- `/v1/guardrail/checks` endpoint — check messages against rails without LLM generation
- Header forwarding — X-* headers forwarded to LLM providers, auth token redaction in logs
- `Dockerfile.server` — UBI9 multi-stage build with baked-in models
- `scripts/filter_guardrails.py` — open/closed source guardrail filtering via profiles
- `requirements.txt` — pinned deps via `uv pip compile`

#!/bin/bash

# Allow runtime overrides via env vars or args
CONFIG_ID="${CONFIG_ID:-${1:-nemo}}"
PORT="${PORT:-${2:-8000}}"

CONFIG_DIR="/app/config/${CONFIG_ID}"

echo "🚀 Starting NeMo Guardrails with config from: $CONFIG_DIR (port: $PORT)"

# Check if models are available (either baked in or from PVC)
CACHE_DIR="${HF_HOME:-/app/.cache/huggingface}"
if [[ ! -d "$CACHE_DIR" ]] || [[ -z "$(ls -A $CACHE_DIR 2>/dev/null)" ]]; then
  echo "⚠️  WARNING: Model cache appears empty at $CACHE_DIR"
  echo "   Models should be either:"
  echo "   1. Baked into the image (build with --build-arg DOWNLOAD_MODELS=true), or"
  echo "   2. Provided via PVC mounted at /app/.cache (using init container)"
  echo ""
  echo "   The application may fail if required models are not available."
  echo ""
fi

# Validate config exists
if [[ ! -f "$CONFIG_DIR/config.yaml" ]]; then
  echo "❌ ERROR: config.yaml not found in $CONFIG_DIR"
  exit 1
fi

if [[ ! -f "$CONFIG_DIR/rails.co" ]]; then
  echo "❌ ERROR: rails.co not found in $CONFIG_DIR (ConfigMap is read-only, please provide it)"
  exit 1
fi

echo "✅ Configuration validated. Starting server..."
exec /app/.venv/bin/nemoguardrails server \
  --config "/app/config" \
  --port "$PORT" \
  --default-config-id "$CONFIG_ID" \
  --disable-chat-ui
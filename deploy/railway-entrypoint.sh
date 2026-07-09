#!/usr/bin/env bash
# Entrypoint for the Railway/Docker deployment of Wizard: inject the Lichess
# token into lichess-bot's config, then run it. The token comes from the
# LICHESS_BOT_TOKEN environment variable (set in Railway -> Variables) so it is
# never baked into the image. Engine paths are already the container layout
# (/app/engine/...) in deploy/config.yml.
set -euo pipefail

CFG=/app/lichess-bot/config.yml

if [ -z "${LICHESS_BOT_TOKEN:-}" ]; then
  echo "ERROR: LICHESS_BOT_TOKEN environment variable is not set." >&2
  echo "Set it in Railway -> your service -> Variables." >&2
  exit 1
fi

# Inject the token (Railway masks secrets in logs).
sed -i "s|PASTE_YOUR_BOT_TOKEN_HERE|${LICHESS_BOT_TOKEN}|" "$CFG"

# Optional overrides via env (handy for tuning without editing the repo):
#   THREADS, HASH, MOVE_OVERHEAD
if [ -n "${THREADS:-}" ];       then sed -i "s|Threads: [0-9]*|Threads: ${THREADS}|" "$CFG"; fi
if [ -n "${HASH:-}" ];          then sed -i "s|Hash: [0-9]*|Hash: ${HASH}|" "$CFG"; fi
if [ -n "${MOVE_OVERHEAD:-}" ]; then sed -i "s|Move Overhead: [0-9]*|Move Overhead: ${MOVE_OVERHEAD}|" "$CFG"; fi

cd /app/lichess-bot
echo "Starting lichess-bot (engine: /app/engine/wizard/wizard)..."
exec python3 lichess-bot.py

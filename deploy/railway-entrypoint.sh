#!/usr/bin/env bash
# Entrypoint for the Railway/Docker deployment: inject the token + container paths
# into lichess-bot's config, then run it. Token comes from the LICHESS_BOT_TOKEN
# environment variable (set in Railway → Variables) so it's never baked into the image.
set -euo pipefail

CFG=/app/lichess-bot/config.yml

if [ -z "${LICHESS_BOT_TOKEN:-}" ]; then
  echo "ERROR: LICHESS_BOT_TOKEN environment variable is not set." >&2
  echo "Set it in Railway -> your service -> Variables." >&2
  exit 1
fi

# Inject token (GitHub/Railway mask secrets in logs) and rewrite the engine paths
# from the config's placeholders to this container's layout.
sed -i "s|PASTE_YOUR_BOT_TOKEN_HERE|${LICHESS_BOT_TOKEN}|" "$CFG"
sed -i "s|dir: \"/home/user/engine/bin\"|dir: \"/app/engine/bin\"|" "$CFG"
sed -i "s|working_dir: \"/home/user/engine\"|working_dir: \"/app/engine\"|" "$CFG"

# Optional overrides via env (handy for testing without editing the repo):
#   THREADS, HASH, MOVE_OVERHEAD
if [ -n "${THREADS:-}" ];       then sed -i "s|Threads: [0-9]*|Threads: ${THREADS}|" "$CFG"; fi
if [ -n "${HASH:-}" ];          then sed -i "s|Hash: [0-9]*|Hash: ${HASH}|" "$CFG"; fi
if [ -n "${MOVE_OVERHEAD:-}" ]; then sed -i "s|Move Overhead: [0-9]*|Move Overhead: ${MOVE_OVERHEAD}|" "$CFG"; fi

cd /app/lichess-bot
echo "Starting lichess-bot (engine: /app/engine/bin/engine)..."
exec python3 lichess-bot.py

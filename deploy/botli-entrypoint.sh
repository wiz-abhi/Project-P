#!/usr/bin/env bash
# Entrypoint for the BotLi + Wizard 3.0 (Stockfish 18) deployment on Railway.
# Injects the Lichess token from the LICHESS_BOT_TOKEN env var, then runs BotLi in
# continuous matchmaking mode (also accepts incoming challenges).
set -euo pipefail

CFG=/app/botli/config.yml

if [ -z "${LICHESS_BOT_TOKEN:-}" ]; then
  echo "ERROR: LICHESS_BOT_TOKEN is not set. Set it in Railway -> Variables." >&2
  exit 1
fi

sed -i "s|PASTE_YOUR_BOT_TOKEN_HERE|${LICHESS_BOT_TOKEN}|" "$CFG"

# Optional env overrides (tune without editing the repo).
if [ -n "${THREADS:-}" ];       then sed -i "s|Threads: [0-9]*|Threads: ${THREADS}|" "$CFG"; fi
if [ -n "${HASH:-}" ];          then sed -i "s|Hash: [0-9]*|Hash: ${HASH}|" "$CFG"; fi
if [ -n "${MOVE_OVERHEAD:-}" ]; then sed -i "s|Move Overhead: [0-9]*|Move Overhead: ${MOVE_OVERHEAD}|" "$CFG"; fi

cd /app/botli

# Engine self-test: prove the binary actually runs on THIS replica before BotLi
# starts (a dead/incompatible binary otherwise hangs BotLi at 'Testing engine ...').
echo "Engine self-test: UCI handshake ..."
if ! printf 'uci\nisready\nquit\n' | timeout 30 ./engines/wizard | grep -m1 uciok; then
  echo "ERROR: engine failed the UCI handshake on this machine (crash/illegal instruction?)." >&2
  echo "CPU flags on this replica:" >&2
  grep -m1 -o 'avx2\|bmi2\|popcnt\|ssse3' /proc/cpuinfo | sort -u >&2 || true
  exit 1
fi
echo "Engine OK."

echo "Starting BotLi (Wizard 3.0 / Stockfish 18) in matchmaking mode..."
# Prefer the uv-managed venv (BotLi's documented run path); fall back to plain python.
if command -v uv >/dev/null 2>&1 && [ -d .venv ]; then
  exec uv run user_interface.py matchmaking
else
  exec python3 user_interface.py matchmaking
fi

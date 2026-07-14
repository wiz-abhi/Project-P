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

# --- Run BotLi, with an optional rating cap --------------------------------
# If STOP_AT_RATING is set (e.g. 3100), the bot stops playing once it reaches
# that rating in STOP_PERF (default: bullet), locking the rating in. Games in
# progress finish gracefully — we send SIGTERM, and BotLi drains its running
# game tasks before exiting (game_manager.run: `for task in self.tasks: await task`),
# so nothing is abandoned/forfeited. Leave STOP_AT_RATING unset to play forever.
STOP_PERF="${STOP_PERF:-bullet}"
POLL_SECONDS="${POLL_SECONDS:-120}"

run_botli() {
  echo "Starting BotLi (Wizard 3.0 / Stockfish 18) in matchmaking mode..."
  if command -v uv >/dev/null 2>&1 && [ -d .venv ]; then
    uv run user_interface.py matchmaking &
  else
    python3 user_interface.py matchmaking &
  fi
  BOTLI_PID=$!
}

# Forward container SIGTERM/redeploy to BotLi so it shuts down gracefully too.
trap 'echo "Signal received — stopping BotLi gracefully."; kill -TERM "${BOTLI_PID:-0}" 2>/dev/null || true; wait "${BOTLI_PID:-0}" 2>/dev/null || true; exit 0' TERM INT

current_rating() {
  curl -s --max-time 10 "https://lichess.org/api/account" \
    -H "Authorization: Bearer ${LICHESS_BOT_TOKEN}" \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['perfs']['${STOP_PERF}']['rating'])" 2>/dev/null || echo 0
}

# No cap configured -> original behaviour: run forever in the foreground.
if [ -z "${STOP_AT_RATING:-}" ]; then
  run_botli
  wait "$BOTLI_PID"
  exit $?
fi

echo "Rating cap ENABLED: bot will stop once ${STOP_PERF} rating >= ${STOP_AT_RATING}."
while true; do
  R="$(current_rating)"
  # Fail-safe: a non-numeric/0 reading (API hiccup) is treated as "below target"
  # so a transient error never stops the bot by mistake.
  if [ "${R:-0}" -ge "$STOP_AT_RATING" ] 2>/dev/null; then
    echo "[cap] ${STOP_PERF} rating ${R} >= ${STOP_AT_RATING}: target held. Bot idle (not playing)."
    echo "[cap] To resume: lower/unset STOP_AT_RATING in Railway Variables and redeploy."
    sleep 3600 & wait $!    # stay up (healthy container) without playing; re-check hourly
    continue
  fi

  echo "[cap] ${STOP_PERF} rating ${R} < ${STOP_AT_RATING}: playing."
  run_botli
  while kill -0 "$BOTLI_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS" & wait $!
    R="$(current_rating)"
    echo "[cap] ${STOP_PERF} rating: ${R} / target ${STOP_AT_RATING}"
    if [ "${R:-0}" -ge "$STOP_AT_RATING" ] 2>/dev/null; then
      echo "[cap] Target reached (${R}). SIGTERM -> BotLi finishes in-progress games, then exits."
      kill -TERM "$BOTLI_PID" 2>/dev/null || true
      wait "$BOTLI_PID" 2>/dev/null || true
      break
    fi
  done
  # Loop back: re-check rating. If the games that just finished dropped us back
  # under target, we restart and keep going; if we're still >= target, we idle.
done

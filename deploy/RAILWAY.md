# Deploying the Lichess bot on Railway (EU) — recommended host

Railway runs the bot as an always-on container in an **EU region** (near Lichess),
which removes both the latency and the CPU-throttling/6h-cap limitations of the
GitHub Actions deployment. This is the better host for chasing a higher rating.

> Uses the repo `Dockerfile` (builds the engine, bakes in the NNUE net + book,
> installs `lichess-bot`) and `deploy/config.yml` (bullet 1+1, book on, pondering).

## Steps

### 1. Lichess BOT token
Same as before — a BOT account with a `bot:play`-scoped token (see `deploy/DEPLOY.md` §1).
`checker-op` is already a BOT, so reuse its token (regenerate it if it's been exposed).

### 2. Create the Railway project from the repo
1. <https://railway.app> → **New Project → Deploy from GitHub repo** → pick `wiz-abhi/Project-P`.
2. Railway auto-detects the **`Dockerfile`** and builds it (engine `make` + perft self-test + net/book download + lichess-bot install). First build ~2–4 min.

### 3. Set the EU region  ← the whole point
Service → **Settings → Region → `Europe West (Amsterdam)`** (or the nearest EU region).
Redeploy after changing region if needed.

### 4. Set the token (and optional tuning) as variables
Service → **Variables**:
- `LICHESS_BOT_TOKEN` = your BOT token   **(required)**
- optional: `THREADS` (e.g. `4`), `HASH` (MB), `MOVE_OVERHEAD` (ms) — override the
  config without editing the repo. On Railway with lower latency you can try
  `MOVE_OVERHEAD=200` for more think-time (watch for flags first).

### 5. Deploy
It builds and starts automatically. This is a **worker** (no HTTP port needed) — you
do NOT need to add a public domain. Watch **Deployments → Logs**: you want to see
`ALL POSITIONS PASS`, `NNUE loaded ... 21022697 / 21022697`, then
`Welcome checker-op!` / `awaiting challenges`, and games starting.

## Cost (be aware)
Railway isn't free long-term: trial credit first, then **~$5/mo + usage**. A 24/7
CPU-bound bot at 3–4 threads uses real CPU, so the meter runs — fine for a **test**,
budget it if you keep it running. To pause, stop/sleep the service in Railway.

## What to compare (the experiment)
Let it play **~20–30 bullet games**, then compare vs the GitHub-Actions numbers:
- **Flag rate** should stay ~0 (EU latency is lower than the US runner).
- **Depth per move** should be higher (better CPU) → stronger play.
- If the **rating climbs above the ~2547 it settled at on Actions**, that confirms the
  US-runner hardware/latency was the ceiling — and justifies keeping an EU host.

## Note
Because the engine now builds as `bin/engine` (no `.exe`) on Linux with `-pthread`
and a portable allocator (the cross-platform work already done), the Docker build
"just works". If `make` ever fails on a Railway base-image g++ that's too old,
switch the Dockerfile base to a newer image (e.g. `debian:trixie` + `g++`), but
`python:3.11-slim` (g++ 12) supports the C++20 we use.

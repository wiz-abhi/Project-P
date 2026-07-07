# Running the Lichess bot on GitHub Actions (cron)

This runs the engine as a Lichess **blitz (3+2)** bot using GitHub Actions on a
cron schedule — no server to manage. It works, but read the **caveats** first;
this is a compromise, not the ideal host.

> Time control is **blitz 3+2**, not bullet, on purpose: GitHub runners are far
> from Lichess (EU) and bullet punishes network latency hard. Blitz still lets an
> engine of this strength climb toward 3100 while tolerating latency.

## Caveats (know these going in)
- **Not truly 24/7.** Jobs are killed at 6h; a cron relaunches. Cron on GitHub is
  often delayed or skipped, so expect **downtime gaps**. Gaps don't lose rating (an
  idle Lichess bot isn't penalized) — they just slow the climb.
- **Public repo required.** Free unlimited Actions minutes need a **public** repo.
  Private repos only get ~2000 min/month (far too little).
- **ToS gray area.** GitHub Actions is for CI/CD, not persistent bots. This can be
  flagged as abuse. Proceed with that understanding; a small EU VPS
  (`deploy/DEPLOY.md`) is the ToS-clean, lower-latency alternative if this gets
  shut down.
- **Auto-disable after 60 days** of no repo commits (scheduled workflows only).
  We'll reach 3100 well before that; a trivial commit re-arms it if needed.

## One-time setup

### 1. Lichess BOT account + token
Same as `deploy/DEPLOY.md` §1: register a **fresh** account, create an API token
with the **`bot:play`** scope, then upgrade:
```bash
curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer <TOKEN>"
```

### 2. Push this project to a PUBLIC GitHub repo
```bash
cd C:/Users/abhis/Desktop/OSS/Client
git init && git add -A && git commit -m "Chess engine + Lichess Actions bot"
# create a PUBLIC repo on github.com, then:
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
The NNUE net is git-ignored (21 MB); the workflow **downloads it at run time**, so
you don't commit it.

### 3. Add the token as a repository secret
GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**:
- Name: `LICHESS_BOT_TOKEN`
- Value: your Lichess BOT token

Never commit the token — it only lives as a secret. The workflow injects it and
GitHub masks it in logs.

### 4. Start it
GitHub repo → **Actions** tab → enable workflows if prompted → select
**"lichess-bot"** → **Run workflow** (this is the `workflow_dispatch` manual
start). After that, the cron (`every 5h`) keeps relaunching it; each run plays for
up to ~5h50m and the next queued run takes over.

## How the continuous-run trick works
- `schedule: cron "0 */5 * * *"` fires a run every 5 hours.
- `concurrency: {group: lichess-bot, cancel-in-progress: false}` means a new run
  **queues** (doesn't kill) while one is active; as the running job hits its
  `timeout-minutes: 350` (~5h50m), the queued run starts → near-seamless handoff.
- If a cron is dropped, the next one self-heals within a few hours.

## Monitoring
- **Actions tab**: watch the running job's logs (build → `NNUE loaded` → games).
  The build step runs `perft` — if it prints `ALL POSITIONS PASS`, the Linux build
  is confirmed good.
- **Lichess**: the bot's profile shows the blitz rating climbing. Expect the rating
  deviation to shrink over the first ~30–50 games and settle.

## If the climb stalls below 3100
Levers, in order of easiness: raise `Threads`/`Hash` in `deploy/config.yml`; add the
opening book; add the incremental-NNUE speedup; or move to an EU VPS
(`deploy/DEPLOY.md`) to kill latency and run truly 24/7.

# Deploying the engine to Lichess (bullet rating climb)

This guide takes you from a finished engine to a **BOT account climbing toward 3100
on Lichess bullet**. It covers the Lichess account/token, the server, building the
engine on that server, and running `lichess-bot` with the tuned config in this
folder (`deploy/config.yml`).

> Goal recap: touch **3100** in **bullet** (engines dominate fast time controls),
> with matchmaking that seeks strong opponents (~2800–3500) so the rating climbs
> quickly.

---

## 1. Create a Lichess BOT account

A BOT account must be **brand new** — Lichess will not convert an account that has
already played games.

1. **Register a fresh account** at <https://lichess.org/signup> (pick the bot's
   playing name; it will show a "BOT" tag).
2. Log in as that account, go to **Preferences → API Access tokens →
   <https://lichess.org/account/oauth/token>**.
3. Click **New token**, give it a description, and enable the scope
   **"Play games with the Board and Bot API" (`bot:play`)**. Create it and
   **copy the token** (shown once).
4. **Upgrade the account to a BOT** (irreversible; the account must have zero
   played games):
   ```bash
   curl -d '' https://lichess.org/api/bot/account/upgrade \
        -H "Authorization: Bearer <YOUR_TOKEN>"
   ```
   A `{"ok":true}` response means it's now a BOT account.

Keep the token secret — anyone with it controls the bot. Put it in an environment
variable or the `config.yml` `token:` field on the server only.

---

## 2. Pick a server

Bullet is latency-sensitive: a slow connection to Lichess (EU-hosted) costs you
games on time. Recommended:

| Spec | Recommendation |
|---|---|
| **Region** | **Europe** (Lichess servers are in EU) — e.g. Hetzner (DE/FI), or DigitalOcean/Vultr/Linode Frankfurt/Amsterdam. Low ping matters. |
| **CPU** | 4 physical cores (the engine uses Lazy SMP; set `Threads` = physical cores). 2 cores works, 4 is better. |
| **RAM** | 4 GB (plenty; engine hash is only 128–256 MB for bullet). |
| **OS** | Ubuntu 22.04/24.04 LTS (Linux; the engine builds natively there). |
| **Cost** | ~€4–8/month (e.g. Hetzner CX22). |

A cheap EU VPS with 4 cores comfortably runs one bullet bot.

---

## 3. Build the engine on the server

The engine is **built from source on the target machine** (so `-march=native`
matches that CPU). Do NOT copy the Windows `.exe`.

```bash
# On the Ubuntu server:
sudo apt update
sudo apt install -y g++ make python3 python3-pip python3-venv git   # g++ must be >= 13 for C++20

# Copy the project to the server (from your machine):
#   scp -r C:/Users/abhis/Desktop/OSS/Client user@server:/home/user/engine
# (or git clone if you push it to a repo)

cd ~/engine
make                       # produces bin/engine  (Linux binary)
./bin/engine perft         # sanity: must print "ALL POSITIONS PASS"
```

**Important — the NNUE net file:** `nets/nn-halfkp.nnue` (21 MB) is required at
run time and is git-ignored, so if you cloned via git you must copy it separately:
```bash
scp C:/Users/abhis/Desktop/OSS/Client/nets/nn-halfkp.nnue user@server:~/engine/nets/
```
Confirm the engine finds it:
```bash
printf 'uci\nquit\n' | ./bin/engine | grep -i "NNUE loaded"
# -> NNUE loaded: nets/nn-halfkp.nnue, bytes consumed = 21022697 / 21022697
```

---

## 4. Install lichess-bot

`lichess-bot` is the official bridge that runs a UCI engine on a Lichess BOT
account.

```bash
cd ~
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Use our tuned config:
cp ~/engine/deploy/config.yml ./config.yml
```

Then edit `config.yml`:
- Set `token:` to your BOT token (or leave it and export `LICHESS_BOT_TOKEN`).
- Set `engine: dir:` to the absolute path of `~/engine/bin` and `name: engine`.
- Confirm `engine: working_dir:` points to `~/engine` so the relative
  `nets/nn-halfkp.nnue` and book paths resolve.

> lichess-bot's config schema evolves. If your cloned version's
> `config.yml.default` has keys ours doesn't (or vice versa), start from THAT
> default and copy over the `engine:`, `uci_options:`, `matchmaking:`, and
> `challenge:` blocks from `deploy/config.yml`.

---

## 5. Run it (24/7)

```bash
# quick test (foreground):
python3 lichess-bot.py -v

# persistent (survives disconnect) — use tmux or systemd:
tmux new -s bot
python3 lichess-bot.py
#   detach: Ctrl-b then d ;  reattach: tmux attach -t bot
```

For production, a **systemd service** is cleaner (auto-restart on crash/reboot) —
see `deploy/lichess-bot.service` for a template.

---

## 6. How the rating climb works

With `deploy/config.yml`:
- The bot **seeks rated bullet games** against opponents rated **2800–3500**
  (`matchmaking`), so wins move the rating up fast (Glicko-2 rewards beating
  strong opponents and settles quickly).
- It also **accepts** incoming bullet challenges in that band.
- `Move Overhead` is set high (250 ms) so network latency never flags the bot.

Watch the climb on the bot's Lichess profile. Expect the rating deviation (RD) to
shrink over the first ~30–50 games and settle. If it stalls below 3100, the levers
are: raise `Threads`/`Hash`, add the opening book, or the incremental-NNUE speedup.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Bot doesn't start games | Ensure `allow_matchmaking: true` and the token has `bot:play` scope. |
| Loses on time (flagging) | Increase `Move Overhead` to 300–400; check server→Lichess ping. |
| "NNUE loaded ... 0 / ..." or missing | The net file isn't at `nets/nn-halfkp.nnue` relative to `working_dir`. |
| Engine not found | `engine: dir:`/`name:` wrong; use absolute dir and `name: engine`. |
| Only unrated/casual games | Set `challenge_mode: rated` and `matchmaking` rated. |

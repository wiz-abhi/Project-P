#!/usr/bin/env bash
# One-shot setup for an Ubuntu EU VPS: build the engine + install lichess-bot.
# Run AFTER copying the project to the server. See deploy/DEPLOY.md for the full
# walkthrough (account creation, token, systemd).
#
# Usage:
#   ENGINE_DIR=~/engine ./deploy/setup.sh
set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-$HOME/engine}"
BOT_DIR="${BOT_DIR:-$HOME/lichess-bot}"

echo "==> Installing build + python dependencies"
sudo apt update
sudo apt install -y g++ make python3 python3-pip python3-venv git

echo "==> Checking g++ supports C++20 (need >= 13)"
g++ --version | head -1

echo "==> Building the engine in $ENGINE_DIR"
cd "$ENGINE_DIR"
make
echo "==> Perft self-test (must say ALL POSITIONS PASS)"
./bin/engine perft | tail -1

echo "==> Verifying NNUE net is present"
if printf 'uci\nquit\n' | ./bin/engine | grep -q "NNUE loaded"; then
  printf 'uci\nquit\n' | ./bin/engine | grep -i "NNUE loaded"
else
  echo "!! NNUE net NOT loaded — copy nets/nn-halfkp.nnue to $ENGINE_DIR/nets/"
fi

echo "==> Installing lichess-bot in $BOT_DIR"
if [ ! -d "$BOT_DIR" ]; then
  git clone https://github.com/lichess-bot-devs/lichess-bot.git "$BOT_DIR"
fi
cd "$BOT_DIR"
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --quiet -r requirements.txt

if [ ! -f "$BOT_DIR/config.yml" ]; then
  cp "$ENGINE_DIR/deploy/config.yml" "$BOT_DIR/config.yml"
  echo "==> Copied config.yml — NOW EDIT IT: set token, and engine dir/working_dir paths."
fi

cat <<EOF

==> Done.
Next:
  1. Edit $BOT_DIR/config.yml : set 'token', and engine 'dir'/'working_dir' to $ENGINE_DIR.
  2. Test:   cd $BOT_DIR && source venv/bin/activate && python3 lichess-bot.py -v
  3. 24/7:   sudo cp $ENGINE_DIR/deploy/lichess-bot.service /etc/systemd/system/ && \\
             sudo systemctl daemon-reload && sudo systemctl enable --now lichess-bot
EOF

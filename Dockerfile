# Dockerfile — run the Wizard chess engine as a Lichess bot on Railway (or any
# container host). Builds Wizard on Linux (embedding the NNUE net at build time),
# bakes in the opening book, installs lichess-bot, and launches it. The Lichess
# token comes from the LICHESS_BOT_TOKEN environment variable at runtime (set it
# in Railway -> Variables). See deploy/RAILWAY.md.
#
# Wizard is a derivative of Stockfish 15.1 (GPL-3); see wizard/NOTICE.
FROM python:3.11-slim

# Build toolchain (g++ >= 12 supports the C++17 Wizard needs) + fetch tools.
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ make git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- Build the Wizard engine ---
# `make build` runs `make net`, which downloads the SF15.1 default network
# (nn-ad9b42354671.nnue) and embeds it into the binary, so the engine is
# self-contained (no runtime EvalFile needed). ARCH=x86-64-avx2 is safe on all
# modern cloud CPUs (Intel Haswell+ / AMD Zen+) and gives fast NNUE inference.
WORKDIR /app/engine
COPY . /app/engine
RUN cd wizard \
 && make clean \
 && make -j build ARCH=x86-64-avx2 COMP=gcc \
 && ./wizard bench 2>&1 | tail -4        # self-test: prints "Nodes searched"

# --- Bake in the opening book (git-ignored, so fetched here) ---
RUN mkdir -p book \
 && curl -L --fail --retry 3 -o book/komodo.bin \
      "https://raw.githubusercontent.com/michaeldv/donna_opening_books/master/komodo.bin"

# --- Install lichess-bot ---
RUN git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git /app/lichess-bot \
 && pip install --no-cache-dir -r /app/lichess-bot/requirements.txt
COPY deploy/config.yml /app/lichess-bot/config.yml

# --- Launch ---
COPY deploy/railway-entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]

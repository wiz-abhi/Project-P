# Dockerfile — run the chess engine as a Lichess bot on Railway (or any container
# host). Builds the engine on Linux, bakes in the NNUE net + opening book, installs
# lichess-bot, and launches it. The Lichess token comes from the LICHESS_BOT_TOKEN
# environment variable at runtime (set it in Railway → Variables). See deploy/RAILWAY.md.
FROM python:3.11-slim

# Build toolchain (g++ >= 12 on bookworm supports the C++20 we use) + fetch tools.
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ make git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- Build the engine ---
WORKDIR /app/engine
COPY . /app/engine
RUN make && ./bin/engine perft | tail -1        # self-test: must print ALL POSITIONS PASS

# --- Bake in the NNUE net + opening book (git-ignored, so fetched here) ---
RUN mkdir -p nets book \
 && curl -L --fail --retry 3 -o nets/nn-halfkp.nnue \
      "https://tests.stockfishchess.org/api/nn/nn-62ef826d1a6d.nnue" \
 && curl -L --fail --retry 3 -o book/komodo.bin \
      "https://raw.githubusercontent.com/michaeldv/donna_opening_books/master/komodo.bin" \
 && printf 'uci\nquit\n' | ./bin/engine | grep -i "NNUE loaded"

# --- Install lichess-bot ---
RUN git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git /app/lichess-bot \
 && pip install --no-cache-dir -r /app/lichess-bot/requirements.txt
COPY deploy/config.yml /app/lichess-bot/config.yml

# --- Launch ---
COPY deploy/railway-entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]

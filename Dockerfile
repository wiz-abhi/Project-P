# Dockerfile — Wizard 3.0 (Stockfish 18) running under the BotLi framework on Railway.
# BotLi (https://github.com/Torom/BotLi) is the framework the top rated bullet bots use:
# aggressive matchmaking + concurrency + strong opening books + anti-draw play. The Lichess
# token comes from the LICHESS_BOT_TOKEN env var at runtime (set it in Railway -> Variables).
#
# Wizard 3.0 is a derivative of Stockfish 18 (GPL-3); see wizard/NOTICE.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ make git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- Build Wizard 3.0 (Stockfish 18); NNUE nets are embedded, so the binary is self-contained ---
WORKDIR /app/engine
COPY . /app/engine
RUN cd wizard \
 && make clean \
 && make -j profile-build ARCH=x86-64-avx2 COMP=gcc \
 && ./wizard bench 2>&1 | tail -3

# --- BotLi framework (uv-managed venv; fall back to pip if uv sync fails) ---
RUN git clone --depth 1 https://github.com/Torom/BotLi.git /app/botli \
 && pip install --no-cache-dir uv
WORKDIR /app/botli
RUN uv sync || pip install --no-cache-dir -r requirements.txt || true

# --- Place the engine binary + opening book into BotLi's ./engines dir ---
RUN mkdir -p /app/botli/engines \
 && cp /app/engine/wizard/wizard /app/botli/engines/wizard \
 && chmod +x /app/botli/engines/wizard \
 && curl -L --fail --retry 3 -o /app/botli/engines/book.bin \
      "https://raw.githubusercontent.com/michaeldv/donna_opening_books/master/komodo.bin"

COPY deploy/botli-config.yml /app/botli/config.yml
COPY deploy/botli-entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]

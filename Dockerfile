# Dockerfile — run Wizard 2.0 as a Lichess bot on Railway (or any container host).
# Builds Wizard 2.0 on Linux (embedding the NNUE nets at build time), installs
# lichess-bot, and launches it. The Lichess token comes from the LICHESS_BOT_TOKEN
# environment variable at runtime (set it in Railway -> Variables).
#
# Wizard 2.0 is a derivative of Stockfish 17.1 (GPL-3); see wizard/NOTICE.
FROM python:3.11-slim

# Build toolchain + fetch tools (coreutils provides sha256sum for net verification).
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ make git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- Build Wizard 2.0 (Stockfish 17.1 derivative) ---
# `make build` runs ../scripts/net.sh, which downloads the two default NNUE nets
# and embeds them, so the binary is self-contained. The wizard/ Makefile expects
# a sibling scripts/ dir (both are in this repo). ARCH=x86-64-avx2 is safe on all
# modern cloud CPUs (Intel Haswell+ / AMD Zen+) and gives fast NNUE inference.
WORKDIR /app/engine
COPY . /app/engine
RUN cd wizard \
 && make clean \
 && make -j profile-build ARCH=x86-64-avx2 COMP=gcc \
 && ./wizard bench 2>&1 | tail -4        # PGO build (self-benches then recompiles) ~= +20-40 Elo
# NOTE: profile-build runs the compiled binary during the build to collect the PGO profile, so the
# build machine must support the target ARCH. avx2 is universally safe; bump to x86-64-bmi2 if the
# build host is a modern Intel/AMD (faster PEXT magics) — the GitHub Actions runner already uses bmi2.

# --- Install lichess-bot ---
RUN git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git /app/lichess-bot \
 && pip install --no-cache-dir -r /app/lichess-bot/requirements.txt
COPY deploy/config.yml /app/lichess-bot/config.yml

# --- Launch ---
COPY deploy/railway-entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]

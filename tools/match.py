#!/usr/bin/env python3
"""Play a UCI match between two engines and report W/D/L, score% and Elo diff.

Reusable across the project: Phase-3 self-play regression, the Phase-7 Stockfish
benchmark, and sanity-checking the Lichess build. Uses python-chess as referee so
game-termination (mate/stalemate/repetition/50-move/insufficient material) is
handled correctly and independently of the engines under test.

Examples
--------
  # 40 games, 300ms/move, new build vs the saved Phase-2 baseline
  python tools/match.py --engineA bin/engine.exe \
      --engineB baseline/engine_phase2.exe --games 40 --movetime 300

  # fixed depth instead of time
  python tools/match.py --engineA bin/engine.exe --engineB other.exe \
      --games 20 --depth 8
"""
import argparse
import math
import os
import sys

import chess
import chess.engine

# A spread of balanced opening positions (as move sequences from startpos) so the
# games are varied rather than 100 copies of the same line. Each is played twice,
# with the engines swapping colors, to cancel first-move/opening bias.
OPENINGS = [
    [],                                   # 1. e-nothing: engines choose from scratch
    ["e2e4", "e7e5"],                     # 2. Open game
    ["e2e4", "c7c5"],                     # 3. Sicilian
    ["d2d4", "d7d5"],                     # 4. Queen's pawn
    ["d2d4", "g8f6", "c2c4", "e7e6"],     # 5. Indian
    ["e2e4", "e7e6"],                     # 6. French
    ["c2c4", "e7e5"],                     # 7. English
    ["g1f3", "d7d5", "g2g3"],             # 8. Reti / KIA
    ["e2e4", "c7c6"],                     # 9. Caro-Kann
    ["d2d4", "f7f5"],                     # 10. Dutch
]

MAX_PLIES = 400  # adjudicate as draw if a game runs this long


def make_limit(args):
    if args.depth:
        return chess.engine.Limit(depth=args.depth)
    if args.nodes:
        return chess.engine.Limit(nodes=args.nodes)
    return chess.engine.Limit(time=args.movetime / 1000.0)


def play_game(engA, engB, opening, a_is_white, limit):
    """Return result from engine A's perspective: 1.0 win, 0.5 draw, 0.0 loss."""
    board = chess.Board()
    for uci in opening:
        board.push(chess.Move.from_uci(uci))

    white, black = (engA, engB) if a_is_white else (engB, engA)
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        engine = white if board.turn == chess.WHITE else black
        try:
            result = engine.play(board, limit)
        except chess.engine.EngineError as e:
            print(f"  engine error ({e}); adjudicating as loss for mover", file=sys.stderr)
            # Side to move forfeits.
            mover_is_a = (board.turn == chess.WHITE) == a_is_white
            return 0.0 if mover_is_a else 1.0
        if result.move is None:
            break
        board.push(result.move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    a_won = (outcome.winner == chess.WHITE) == a_is_white
    return 1.0 if a_won else 0.0


def elo_diff(score):
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engineA", required=True)
    ap.add_argument("--engineB", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--movetime", type=int, default=300, help="ms per move")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=0)
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--optA", default="", help="UCI options for A, e.g. 'UCI_Elo=2400,UCI_LimitStrength=true'")
    ap.add_argument("--optB", default="", help="UCI options for B (same format)")
    args = ap.parse_args()

    limit = make_limit(args)
    pathA = os.path.normpath(os.path.abspath(args.engineA))
    pathB = os.path.normpath(os.path.abspath(args.engineB))
    engA = chess.engine.SimpleEngine.popen_uci(pathA)
    engB = chess.engine.SimpleEngine.popen_uci(pathB)

    def parse_opts(s):
        opts = {}
        for kv in s.split(","):
            kv = kv.strip()
            if not kv:
                continue
            k, _, v = kv.partition("=")
            v = v.strip()
            if v.lower() in ("true", "false"):
                v = (v.lower() == "true")
            elif v.lstrip("-").isdigit():
                v = int(v)
            opts[k.strip()] = v
        return opts

    for e, extra in ((engA, args.optA), (engB, args.optB)):
        try:
            e.configure({"Hash": args.hash})
        except Exception:
            pass
        for k, v in parse_opts(extra).items():
            try:
                e.configure({k: v})
            except Exception as ex:
                print(f"  warn: could not set {k}={v}: {ex}", file=sys.stderr)

    wins = draws = losses = 0
    try:
        for g in range(args.games):
            opening = OPENINGS[(g // 2) % len(OPENINGS)]
            a_is_white = (g % 2 == 0)
            r = play_game(engA, engB, opening, a_is_white, limit)
            if r == 1.0:
                wins += 1
            elif r == 0.5:
                draws += 1
            else:
                losses += 1
            played = g + 1
            score = (wins + 0.5 * draws) / played
            print(f"game {played:3d}/{args.games}  "
                  f"A({'W' if a_is_white else 'B'}) r={r}  "
                  f"W-D-L {wins}-{draws}-{losses}  "
                  f"score {100*score:5.1f}%  elo {elo_diff(score):+.0f}",
                  flush=True)
    finally:
        engA.quit()
        engB.quit()

    n = wins + draws + losses
    score = (wins + 0.5 * draws) / n if n else 0.0
    print("=" * 56)
    print(f"A: {args.engineA}")
    print(f"B: {args.engineB}")
    print(f"games {n}  W-D-L {wins}-{draws}-{losses}  "
          f"score {100*score:.1f}%  elo {elo_diff(score):+.0f}")


if __name__ == "__main__":
    main()

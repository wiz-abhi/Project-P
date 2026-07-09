#!/usr/bin/env python3
"""Compare our engine's NNUE static eval against Stockfish 15.1 on the IDENTICAL
net, position by position. Ground-truth tool for fixing our SFNNv4 inference.

Both are asked for the static NNUE eval (white-relative, centipawns) via the `eval`
command. Prints per-position ours vs SF + error, and a summary (mean/max abs error).
A correct inference should match SF within a few cp on every position.

Usage:
  python tools/nnue_cmp.py --ours bin/engine.exe \
      --sf engines/Stockfish-sf_15.1/src/stockfish.exe \
      --net nets/nn-ad9b42354671.nnue
"""
import argparse, os, re, subprocess, sys

# White-to-move positions spanning piece counts (buckets). Low-material ones first
# — that's where our eval is currently most wrong.
FENS = [
    ("KPK-drawish",  "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"),
    ("KPKP",         "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("K3Peach",      "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"),
    ("queen-vs-bare","4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1"),
    ("R+P ending",   "8/5k2/8/8/8/3R4/5K2/8 w - - 0 1"),
    ("startpos",     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("open Sicilian","r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("Italian",      "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 5"),
    ("Kiwipete",     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("middlegame",   "2rq1rk1/pp1bppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/2KR1B1R w - - 0 11"),
]

def our_eval(exe, net, fen):
    cmd = f"setoption name EvalFile value {net}\nposition fen {fen}\neval\nquit\n"
    out = subprocess.run([os.path.abspath(exe)], input=cmd, capture_output=True,
                         text=True, timeout=30).stdout
    m = re.search(r"nnue\s+(-?\d+)", out)
    return int(m.group(1)) if m else None

def sf_eval(exe, net, fen):
    cmd = (f"uci\nsetoption name EvalFile value {os.path.abspath(net)}\n"
           f"position fen {fen}\neval\nquit\n")
    out = subprocess.run([os.path.abspath(exe)], input=cmd, capture_output=True,
                         text=True, timeout=30).stdout
    m = re.search(r"NNUE evaluation\s+([-+]?\d+\.\d+)", out)
    return int(round(float(m.group(1)) * 100)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--sf", required=True)
    ap.add_argument("--net", required=True)
    a = ap.parse_args()
    print(f"{'position':14} {'SF_cp':>7} {'ours_cp':>8} {'abs_err':>8}")
    errs = []
    for name, fen in FENS:
        sf = sf_eval(a.sf, a.net, fen)
        ov = our_eval(a.ours, a.net, fen)
        if sf is None or ov is None:
            print(f"{name:14} {str(sf):>7} {str(ov):>8}   (parse fail)")
            continue
        e = abs(ov - sf)
        errs.append(e)
        print(f"{name:14} {sf:>7} {ov:>8} {e:>8}")
    if errs:
        print(f"\nmean abs err = {sum(errs)/len(errs):.0f} cp   max = {max(errs)} cp"
              f"   (target: single digits everywhere)")

if __name__ == "__main__":
    main()

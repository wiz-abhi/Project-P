// eval.cpp — Static position evaluation.
//
// Base layer (Phase 2): a classic tapered, hand-crafted evaluation using the
// well-known "PeSTO" (Piece-Square Tables Only) evaluation by Ronald Friederich.
// The material values and piece-square tables are public-domain and, on their
// own, play at roughly 2400+ Elo strength.
//
// Extended layer (this file): the remaining hand-crafted terms, all tapered
// (mg/eg) and computed White−Black then flipped to the side to move:
//   - Piece mobility (N/B/R/Q)
//   - King safety (pawn shield + king-zone attackers)
//   - Pawn structure (isolated / doubled / backward / passed)
//   - Bishop pair
//   - Rook activity (open / semi-open file, 7th rank)
//   - Space (safe squares behind own pawns in the centre)
//   - Endgame scaling (opposite-coloured-bishop draw scaling)
//
// The evaluation keeps two running scores:
//   - mgScore : the midgame (opening) score
//   - egScore : the endgame score
// Both are accumulated as (white contributions) − (black contributions).
//
// The final score is a phase-weighted (tapered) blend of the two, returned
// relative to the side to move.
//
// Weights are literature-based (Chess Programming Wiki, Stockfish, PeSTO) and
// deliberately conservative on the risky terms (king safety, space) so the net
// effect is a strength gain rather than a regression.
//
// Sources:
//   PSQT/material : PeSTO (Ronald Friederich, rofChade), public domain.
//                   https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
//   Other terms   : https://www.chessprogramming.org/Evaluation and the
//                   corresponding Stockfish evaluation terms (scaled down).

#include "eval.hpp"

#include "attacks.hpp"
#include "bitboard.hpp"
#include "nnue.hpp"
#include "position.hpp"
#include "types.hpp"

namespace engine {

// Global toggle for NNUE evaluation (UCI "Use NNUE"). Defined here, declared
// extern in uci.cpp so the option handler can flip it. Defaults to true; the
// NNUE path is only taken when a net is actually loaded.
bool UseNNUE = true;

namespace Eval {

namespace {

// ---------------------------------------------------------------------------
// PeSTO material values (indexed by PieceType: PAWN..QUEEN). King = 0.
// ---------------------------------------------------------------------------
constexpr int mg_value[7] = { 0, 82, 337, 365, 477, 1025, 0 };  // -, P, N, B, R, Q, K
constexpr int eg_value[7] = { 0, 94, 281, 297, 512,  936, 0 };  // -, P, N, B, R, Q, K

// ---------------------------------------------------------------------------
// Piece-square tables, from White's point of view. Index 0 = a1 ... 63 = h8.
// (The PeSTO source lists them a8..h1; they are written here already flipped so
//  that index 0 corresponds to a1, matching this engine's Square enum.)
// For a Black piece on square s, mirror vertically with (s ^ 56).
// ---------------------------------------------------------------------------

constexpr int mg_pawn[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    -35,  -1, -20, -23, -15,  24,  38, -22,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -14,  13,   6,  21,  23,  12,  17, -23,
     -6,   7,  26,  31,  65,  56,  25, -20,
     98, 134,  61,  95,  68, 126,  34, -11,
      0,   0,   0,   0,   0,   0,   0,   0,
};

constexpr int eg_pawn[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     13,   8,   8,  10,  13,   0,   2,  -7,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
     32,  24,  13,   5,  -2,   4,  17,  17,
     94, 100,  85,  67,  56,  53,  82,  84,
    178, 173, 158, 134, 147, 132, 165, 187,
      0,   0,   0,   0,   0,   0,   0,   0,
};

constexpr int mg_knight[64] = {
   -105, -21, -58, -33, -17, -28, -19, -23,
    -29, -53, -12,  -3,  -1,  18, -14, -19,
    -23,  -9,  12,  10,  19,  17,  25, -16,
    -13,   4,  16,  13,  28,  19,  21,  -8,
     -9,  17,  19,  53,  37,  69,  18,  22,
    -47,  60,  37,  65,  84, 129,  73,  44,
    -73, -41,  72,  36,  23,  62,   7, -17,
   -167, -89, -34, -49,  61, -97, -15, -107,
};

constexpr int eg_knight[64] = {
    -29, -51, -23, -15, -22, -18, -50, -64,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -58, -38, -13, -28, -31, -27, -63, -99,
};

constexpr int mg_bishop[64] = {
    -33,  -3, -14, -21, -13, -12, -39, -21,
      4,  15,  16,   0,   7,  21,  33,   1,
      0,  15,  15,  15,  14,  27,  18,  10,
     -6,  13,  13,  26,  34,  12,  10,   4,
     -4,   5,  19,  50,  37,  37,   7,  -2,
    -16,  37,  43,  40,  35,  50,  37,  -2,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -29,   4, -82, -37, -25, -42,   7,  -8,
};

constexpr int eg_bishop[64] = {
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
     -3,   9,  12,   9,  14,  10,   3,   2,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
};

constexpr int mg_rook[64] = {
    -19, -13,   1,  17,  16,   7, -37, -26,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -24, -11,   7,  26,  24,  35,  -8, -20,
     -5,  19,  26,  36,  17,  45,  61,  16,
     27,  32,  58,  62,  80,  67,  26,  44,
     32,  42,  32,  51,  63,   9,  31,  43,
};

constexpr int eg_rook[64] = {
     -9,   2,   3,  -1,  -5, -13,   4, -20,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
      4,   3,  13,   1,   2,   1,  -1,   2,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
     11,  13,  13,  11,  -3,   3,   8,   3,
     13,  10,  18,  15,  12,  12,   8,   5,
};

constexpr int mg_queen[64] = {
     -1, -18,  -9,  10, -15, -25, -31, -50,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -28,   0,  29,  12,  59,  44,  43,  45,
};

constexpr int eg_queen[64] = {
    -33, -28, -22, -43,  -5, -32, -20, -41,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -18,  28,  19,  47,  31,  34,  39,  23,
      3,  22,  24,  45,  57,  40,  57,  36,
    -20,   6,   9,  49,  47,  35,  19,   9,
    -17,  20,  32,  41,  58,  25,  30,   0,
     -9,  22,  22,  27,  27,  19,  10,  20,
};

constexpr int mg_king[64] = {
    -15,  36,  12, -54,   8, -28,  24,  14,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -14, -14, -22, -46, -44, -30, -15, -27,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -17, -20, -12, -27, -30, -25, -14, -36,
     -9,  24,   2, -16, -20,   6,  22, -22,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
    -65,  23,  16, -15, -56, -34,   2,  13,
};

constexpr int eg_king[64] = {
    -53, -34, -21, -11, -28, -14, -24, -43,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -18,  -4,  21,  24,  27,  23,   9, -11,
     -8,  22,  24,  27,  26,  33,  26,   3,
     10,  17,  23,  15,  20,  45,  44,  13,
    -12,  17,  14,  17,  17,  38,  23,  11,
    -74, -35, -18, -18, -11,  15,   4, -17,
};

// Pointer tables indexed by PieceType (PAWN=1 .. KING=6). Slot 0 unused.
constexpr const int* mg_psqt[7] = {
    nullptr, mg_pawn, mg_knight, mg_bishop, mg_rook, mg_queen, mg_king
};
constexpr const int* eg_psqt[7] = {
    nullptr, eg_pawn, eg_knight, eg_bishop, eg_rook, eg_queen, eg_king
};

// Combined (material + PSQT) tables precomputed by init(), indexed
// [color][pieceType][square]. This lets evaluate() do a single table lookup per
// piece with the black-square mirror already baked in.
int mgTable[COLOR_NB][7][SQUARE_NB];
int egTable[COLOR_NB][7][SQUARE_NB];

// Small bonus (midgame units) for having the right to move.
constexpr int Tempo = 10;

// ---------------------------------------------------------------------------
// Extended-term weights (all in centipawns, {mg, eg}). White−Black convention.
// Values are literature-based (CPW / scaled Stockfish) and deliberately tame on
// king safety and space to avoid untuned blunders.
// ---------------------------------------------------------------------------

// A small (mg, eg) score pair. Kept trivial so the compiler inlines everything.
struct S { int mg; int eg; };
constexpr S operator+(S a, S b) { return { a.mg + b.mg, a.eg + b.eg }; }
constexpr S operator-(S a, S b) { return { a.mg - b.mg, a.eg - b.eg }; }
constexpr S operator*(int k, S a) { return { k * a.mg, k * a.eg }; }

// --- Mobility bonuses, indexed by number of safe reachable squares. ---
// Based on Stockfish mobility tables, scaled to this eval's centipawn range.
// Knight has up to 8 moves, bishop 13, rook 14, queen 27.
constexpr S MobilityKnight[9] = {
    {-30,-30}, {-20,-20}, {-6,-6}, {0,0}, {6,6}, {12,12}, {17,17}, {21,21}, {24,24}
};
constexpr S MobilityBishop[14] = {
    {-25,-30}, {-12,-15}, {2,-3}, {8,5}, {14,12}, {19,20}, {23,26}, {26,30},
    {28,33}, {30,35}, {31,37}, {32,38}, {33,39}, {34,40}
};
constexpr S MobilityRook[15] = {
    {-30,-40}, {-16,-18}, {-4,4}, {-2,14}, {0,24}, {2,32}, {5,40}, {8,48},
    {10,54}, {12,58}, {13,62}, {14,64}, {15,66}, {16,68}, {16,69}
};
constexpr S MobilityQueen[28] = {
    {-20,-25}, {-12,-15}, {-4,-8}, {-2,0}, {2,7}, {5,14}, {7,20}, {9,25},
    {11,29}, {13,33}, {14,36}, {15,39}, {16,41}, {17,43}, {18,45}, {18,47},
    {19,48}, {19,49}, {20,50}, {20,51}, {21,52}, {21,52}, {22,53}, {22,53},
    {23,54}, {23,54}, {23,55}, {24,55}
};

// --- Pawn structure (mg, eg), penalties are negative. ---
constexpr S IsolatedPawn = {-8, -12};   // no friendly pawn on adjacent files
constexpr S DoubledPawn  = {-8, -18};   // per extra pawn on a file
constexpr S BackwardPawn = {-6, -10};   // cannot be defended by a friendly pawn advance

// Passed pawn bonus scaled by the pawn's relative rank (index = relative rank
// 0..7; only ranks 1..6 are reachable for a passer). Larger in the endgame.
constexpr S PassedPawn[8] = {
    {0,0}, {2,8}, {6,12}, {12,22}, {24,45}, {45,80}, {75,130}, {0,0}
};

// --- Bishop pair. --- (CPW: ~0.5 pawn; keep moderate.)
constexpr S BishopPair = {25, 45};

// --- Rook activity. ---
constexpr S RookOpenFile     = {28, 12};  // no pawns of either colour on the file
constexpr S RookSemiOpenFile = {12, 8};   // no friendly pawns on the file
constexpr S RookOn7th        = {12, 24};  // rook on relative 7th rank

// --- King safety. ---
// Pawn shield: bonus per friendly pawn on the three files around the king in the
// two ranks directly in front of it. Midgame-weighted; negligible in endgame.
constexpr S PawnShield = {8, 0};
// King-zone attacker weighting. We sum per-attacker weights, then convert to a
// mg penalty via a small quadratic-ish table. Kept conservative.
constexpr int KingAttackWeight[7] = { 0, 0, 2, 2, 3, 5, 0 };  // -, P, N, B, R, Q, K
// Penalty (mg) as a function of accumulated attacker weight (clamped index).
constexpr int KingDanger[16] = {
    0, 0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104, 119
};

// --- Space (mg-only). ---
constexpr int SpaceBonus = 2;  // per safe square behind own pawns in centre files

// Central files (C,D,E,F) used by the space term.
constexpr Bitboard CenterFiles = FileCBB | FileDBB | FileEBB | FileFBB;

}  // namespace

// ---------------------------------------------------------------------------
// init() — precompute the combined material+PSQT tables for both colors.
// ---------------------------------------------------------------------------
void init() {
    for (int pt = PAWN; pt <= KING; ++pt) {
        for (int s = 0; s < SQUARE_NB; ++s) {
            // White: table indexed directly from a1.
            mgTable[WHITE][pt][s] = mg_value[pt] + mg_psqt[pt][s];
            egTable[WHITE][pt][s] = eg_value[pt] + eg_psqt[pt][s];
            // Black: mirror the square vertically (s ^ 56).
            mgTable[BLACK][pt][s] = mg_value[pt] + mg_psqt[pt][s ^ 56];
            egTable[BLACK][pt][s] = eg_value[pt] + eg_psqt[pt][s ^ 56];
        }
    }

    // Attempt to load an NNUE network: the modern HalfKAv2_hm/SFNNv4 net first,
    // then the classic HalfKP net. Try repo-relative paths first, then absolute
    // fallbacks. Failure is non-fatal: evaluate() falls back to the hand-crafted
    // evaluation above. (NNUE::load itself also falls back to the HalfKP net if
    // a requested file fails to parse, so the engine is never netless.)
    if (!NNUE::load("nets/nn-ad9b42354671.nnue"))
        if (!NNUE::load("C:/Users/abhis/Desktop/OSS/Client/nets/nn-ad9b42354671.nnue"))
            if (!NNUE::load("nets/nn-halfkp.nnue"))
                NNUE::load("C:/Users/abhis/Desktop/OSS/Client/nets/nn-halfkp.nnue");
}

namespace {

// Forward span in front of a pawn on square s for colour c (exclusive of s),
// restricted to the pawn's own file.
inline Bitboard forward_file_bb(Color c, Square s) {
    Bitboard file = file_bb(s);
    // Ranks strictly ahead of s from c's perspective.
    Bitboard ahead = (c == WHITE) ? ~Bitboard(0) << (8 * (rank_of(s) + 1))
                                  : (rank_of(s) == 0 ? 0 : (Bitboard(1) << (8 * rank_of(s))) - 1);
    return file & ahead;
}

// The three-file span (own file + adjacent) ahead of a pawn — its "passed pawn"
// blocking mask: enemy pawns anywhere here stop it from being passed.
inline Bitboard passed_span_bb(Color c, Square s) {
    Bitboard front = forward_file_bb(c, s);
    return front | shift<EAST>(front) | shift<WEST>(front);
}

// Adjacent-file mask for a square (files left and right of s), all ranks.
inline Bitboard adjacent_files_bb(Square s) {
    Bitboard f = file_bb(s);
    return shift<EAST>(f) | shift<WEST>(f);
}

// Accumulate all extended (non-PSQT) terms into mg/eg (White − Black).
// Written to be exactly color-symmetric.
void evaluate_extended(const Position& pos, int& mg, int& eg) {
    const Bitboard occ      = pos.pieces();
    const Bitboard wPawns   = pos.pieces(WHITE, PAWN);
    const Bitboard bPawns   = pos.pieces(BLACK, PAWN);
    const Bitboard allPawns = wPawns | bPawns;

    // Squares attacked by enemy pawns (used to restrict mobility & space).
    const Bitboard wPawnAtt = pawn_attacks_bb<WHITE>(wPawns);
    const Bitboard bPawnAtt = pawn_attacks_bb<BLACK>(bPawns);

    for (int c = WHITE; c <= BLACK; ++c) {
        const Color  us   = Color(c);
        const Color  them = ~us;
        const int    sign = (us == WHITE) ? 1 : -1;

        const Bitboard ourPieces   = pos.pieces(us);
        const Bitboard ourPawns    = (us == WHITE) ? wPawns : bPawns;
        const Bitboard theirPawns  = (us == WHITE) ? bPawns : wPawns;
        const Bitboard enemyPawnAtt = (us == WHITE) ? bPawnAtt : wPawnAtt;

        // Mobility area: any square not occupied by our own pieces and not
        // attacked by an enemy pawn.
        const Bitboard mobArea = ~ourPieces & ~enemyPawnAtt;

        S score{0, 0};

        // ---------------- Mobility (N/B/R/Q) ----------------
        {
            Bitboard b = pos.pieces(us, KNIGHT);
            while (b) {
                Square s = pop_lsb(b);
                int m = popcount(PseudoAttacks[KNIGHT][s] & mobArea);
                score = score + MobilityKnight[m];
            }
        }
        {
            Bitboard b = pos.pieces(us, BISHOP);
            while (b) {
                Square s = pop_lsb(b);
                int m = popcount(attacks_bb<BISHOP>(s, occ) & mobArea);
                score = score + MobilityBishop[m];
            }
        }
        {
            Bitboard b = pos.pieces(us, ROOK);
            while (b) {
                Square s = pop_lsb(b);
                int m = popcount(attacks_bb<ROOK>(s, occ) & mobArea);
                score = score + MobilityRook[m];
            }
        }
        {
            Bitboard b = pos.pieces(us, QUEEN);
            while (b) {
                Square s = pop_lsb(b);
                int m = popcount(attacks_bb<QUEEN>(s, occ) & mobArea);
                score = score + MobilityQueen[m];
            }
        }

        // ---------------- Rook activity ----------------
        {
            Bitboard b = pos.pieces(us, ROOK);
            while (b) {
                Square s = pop_lsb(b);
                Bitboard f = file_bb(s);
                if (!(allPawns & f))        score = score + RookOpenFile;
                else if (!(ourPawns & f))   score = score + RookSemiOpenFile;
                if (relative_rank(us, s) == RANK_7) score = score + RookOn7th;
            }
        }

        // ---------------- Bishop pair ----------------
        if (popcount(pos.pieces(us, BISHOP)) >= 2)
            score = score + BishopPair;

        // ---------------- Pawn structure ----------------
        {
            Bitboard b = ourPawns;
            while (b) {
                Square s = pop_lsb(b);
                Bitboard adj = adjacent_files_bb(s);

                // Isolated: no friendly pawn on either adjacent file.
                if (!(ourPawns & adj))
                    score = score + IsolatedPawn;

                // Passed: no enemy pawn on our file or adjacent files ahead of us.
                if (!(theirPawns & passed_span_bb(us, s)))
                    score = score + PassedPawn[relative_rank(us, s)];

                // Backward: no friendly pawn on adjacent files at or behind this
                // pawn's rank (so it can't be supported by a pawn advancing), and
                // the stop square in front is controlled by an enemy pawn.
                if (ourPawns & adj) {   // not isolated -> may be backward
                    Bitboard behindIncl = forward_file_bb(them, s) | rank_bb(s);
                    bool noSupport = !(ourPawns & adj & behindIncl);
                    Square stop = (us == WHITE) ? Square(s + NORTH) : Square(s + SOUTH);
                    bool stopAttacked = is_ok(stop) && (enemyPawnAtt & square_bb(stop));
                    if (noSupport && stopAttacked)
                        score = score + BackwardPawn;
                }
            }
            // Doubled: per extra pawn on each file.
            for (int f = FILE_A; f <= FILE_H; ++f) {
                int n = popcount(ourPawns & file_bb(File(f)));
                if (n > 1)
                    score = score + (n - 1) * DoubledPawn;
            }
        }

        // ---------------- King safety ----------------
        {
            Square ksq = pos.king_square(us);
            Bitboard kingRing = PseudoAttacks[KING][ksq] | square_bb(ksq);

            // Pawn shield: friendly pawns on the king's file + adjacent files,
            // in the two ranks directly in front of the king.
            Bitboard shieldFiles = file_bb(ksq) | adjacent_files_bb(ksq);
            Rank kr = rank_of(ksq);
            Bitboard shieldZone = 0;
            if (us == WHITE) {
                if (kr <= RANK_6) shieldZone |= rank_bb(Rank(kr + 1));
                if (kr <= RANK_5) shieldZone |= rank_bb(Rank(kr + 2));
            } else {
                if (kr >= RANK_2) shieldZone |= rank_bb(Rank(kr - 1));
                if (kr >= RANK_3) shieldZone |= rank_bb(Rank(kr - 2));
            }
            int shield = popcount(ourPawns & shieldFiles & shieldZone);
            score = score + shield * PawnShield;

            // King-zone attackers: sum attack weights of enemy pieces (N/B/R/Q)
            // that attack a square in the king ring.
            int attWeight = 0;
            Bitboard b;
            b = pos.pieces(them, KNIGHT);
            while (b) { Square s = pop_lsb(b);
                if (PseudoAttacks[KNIGHT][s] & kingRing) attWeight += KingAttackWeight[KNIGHT]; }
            b = pos.pieces(them, BISHOP);
            while (b) { Square s = pop_lsb(b);
                if (attacks_bb<BISHOP>(s, occ) & kingRing) attWeight += KingAttackWeight[BISHOP]; }
            b = pos.pieces(them, ROOK);
            while (b) { Square s = pop_lsb(b);
                if (attacks_bb<ROOK>(s, occ) & kingRing) attWeight += KingAttackWeight[ROOK]; }
            b = pos.pieces(them, QUEEN);
            while (b) { Square s = pop_lsb(b);
                if (attacks_bb<QUEEN>(s, occ) & kingRing) attWeight += KingAttackWeight[QUEEN]; }

            if (attWeight > 15) attWeight = 15;
            // Danger is a penalty against 'us' (mg only): subtract it.
            score.mg -= KingDanger[attWeight];
        }

        // ---------------- Space (mg-only) ----------------
        // Safe squares in the centre files behind our own pawns: not occupied,
        // not attacked by an enemy pawn. Counted only while there is enough
        // material for the term to matter (implicitly handled by mg taper).
        {
            Bitboard behind = ourPawns;
            if (us == WHITE) {
                behind |= behind >> 8; behind |= behind >> 16; behind |= behind >> 32;
            } else {
                behind |= behind << 8; behind |= behind << 16; behind |= behind << 32;
            }
            Bitboard safe = behind & CenterFiles & ~occ & ~enemyPawnAtt;
            score.mg += SpaceBonus * popcount(safe);
        }

        mg += sign * score.mg;
        eg += sign * score.eg;
    }
}

// Endgame scale factor (out of 64). Handles opposite-coloured bishops, which are
// strongly drawish. Returns 64 (no scaling) in the general case.
int endgame_scale(const Position& pos, int egScore) {
    const int wB = popcount(pos.pieces(WHITE, BISHOP));
    const int bB = popcount(pos.pieces(BLACK, BISHOP));

    // Opposite-coloured bishops: exactly one bishop each, on opposite colours.
    if (wB == 1 && bB == 1) {
        Square ws = pos.square(WHITE, BISHOP);
        Square bs = pos.square(BLACK, BISHOP);
        bool wDark = (DarkSquares & square_bb(ws)) != 0;
        bool bDark = (DarkSquares & square_bb(bs)) != 0;
        if (wDark != bDark) {
            // Only the bishops + pawns present (pure OCB ending) -> very drawish.
            Bitboard heavy = pos.pieces(KNIGHT) | pos.pieces(ROOK) | pos.pieces(QUEEN);
            if (!heavy)
                return 22;   // ~1/3 : pull the endgame score strongly toward a draw
        }
    }

    (void)egScore;
    return 64;
}

}  // namespace

// ---------------------------------------------------------------------------
// evaluate() — tapered material + PSQT + extended terms, side-to-move relative.
// ---------------------------------------------------------------------------
Value evaluate(const Position& pos) {
    // NNUE path: pure full-refresh evaluation (thread-safe). Take it whenever a
    // net is loaded and the "Use NNUE" option is enabled.
    if (UseNNUE && NNUE::loaded())
        return NNUE::evaluate(pos);

    int mgScore = 0;  // white − black, midgame
    int egScore = 0;  // white − black, endgame

    // Base material + PSQT: single pass over every piece via bitboards.
    for (int c = WHITE; c <= BLACK; ++c) {
        Color   col  = Color(c);
        int     sign = (col == WHITE) ? 1 : -1;
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard b = pos.pieces(col, PieceType(pt));
            while (b) {
                Square s = pop_lsb(b);
                mgScore += sign * mgTable[col][pt][s];
                egScore += sign * egTable[col][pt][s];
            }
        }
    }

    // Extended hand-crafted terms.
    evaluate_extended(pos, mgScore, egScore);

    // Endgame scaling (opposite-coloured bishops, etc.).
    int scale = endgame_scale(pos, egScore);
    if (scale != 64)
        egScore = egScore * scale / 64;

    // Tapered blend. phase: 24 = full material (opening) .. 0 = bare endgame.
    int phase = pos.game_phase();          // already clamped to [0, 24]
    int score = (mgScore * phase + egScore * (24 - phase)) / 24;

    // Convert to side-to-move relative, then add the tempo bonus for the mover.
    Value result = (pos.side_to_move() == WHITE) ? score : -score;
    result += Tempo;

    return result;
}

}  // namespace Eval
}  // namespace engine

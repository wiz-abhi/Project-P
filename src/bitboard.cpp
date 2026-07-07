// bitboard.cpp — Bitboard runtime tables and helpers.
//
// Implements everything declared in bitboard.hpp. Bitboards::init() uses
// sliding attacks and therefore MUST be called only after Attacks::init().

#include "bitboard.hpp"

#include <cstdlib>   // std::abs
#include <sstream>
#include <string>

#include "attacks.hpp"

namespace engine {

// ---------------------------------------------------------------------------
// Table definitions (declared extern in bitboard.hpp)
// ---------------------------------------------------------------------------
Bitboard SquareBB[SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
uint8_t  SquareDistance[SQUARE_NB][SQUARE_NB];

// ---------------------------------------------------------------------------
// Bitboards::init — must run AFTER Attacks::init() (uses slider attacks).
// ---------------------------------------------------------------------------
void Bitboards::init() {
    // Single-bit boards.
    for (Square s = SQ_A1; s <= SQ_H8; s += EAST)
        SquareBB[s] = 1ULL << s;

    // Chebyshev distance between every pair of squares.
    for (Square a = SQ_A1; a <= SQ_H8; a += EAST)
        for (Square b = SQ_A1; b <= SQ_H8; b += EAST) {
            int df = std::abs(file_of(a) - file_of(b));
            int dr = std::abs(rank_of(a) - rank_of(b));
            SquareDistance[a][b] = uint8_t(df > dr ? df : dr);
        }

    // Between / Line tables, computed from empty-board slider attacks.
    for (Square s1 = SQ_A1; s1 <= SQ_H8; s1 += EAST) {
        for (PieceType pt : {BISHOP, ROOK}) {
            for (Square s2 = SQ_A1; s2 <= SQ_H8; s2 += EAST) {
                // Aligned along this slider's lines?
                if (!(attacks_bb(pt, s1, 0) & s2))
                    continue;

                // Full infinite line through s1 and s2 (both endpoints included).
                LineBB[s1][s2] = (attacks_bb(pt, s1, 0) & attacks_bb(pt, s2, 0)) |
                                 square_bb(s1) | square_bb(s2);

                // Squares strictly between (exclusive of both endpoints).
                BetweenBB[s1][s2] = attacks_bb(pt, s1, square_bb(s2)) &
                                    attacks_bb(pt, s2, square_bb(s1));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// pretty — ASCII 8x8 diagram, rank 8 at top.
// ---------------------------------------------------------------------------
std::string Bitboards::pretty(Bitboard b) {
    std::ostringstream ss;

    ss << "+---+---+---+---+---+---+---+---+\n";
    for (int r = RANK_8; r >= RANK_1; --r) {
        for (int f = FILE_A; f <= FILE_H; ++f)
            ss << "| " << ((b & square_bb(make_square(File(f), Rank(r)))) ? "X " : "  ");
        ss << "| " << (1 + r) << "\n";
        ss << "+---+---+---+---+---+---+---+---+\n";
    }
    ss << "  a   b   c   d   e   f   g   h\n";

    return ss.str();
}

}  // namespace engine

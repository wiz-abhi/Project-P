// attacks.hpp — Precomputed piece attack lookups (magic bitboards for sliders).
//
// Interface contract. Implementation (magic initialization + tables) lives in
// attacks.cpp. Attacks::init() must be called once at startup before any query.
#pragma once

#include "bitboard.hpp"

namespace engine {

namespace Attacks {
void init();
}  // namespace Attacks

// Leaper attack tables (filled by Attacks::init()).
//   PawnAttacks[color][square], PseudoAttacks[pieceType][square]
extern Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
extern Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];

// Magic entry for a slider square: index into a shared attack pool.
struct Magic {
    Bitboard  mask;      // relevant occupancy mask (excludes board edges)
    Bitboard  magic;     // multiplier
    Bitboard* attacks;   // pointer into the shared attack table
    unsigned  shift;     // 64 - popcount(mask)

    // Map a board occupancy to the packed index for this square.
    unsigned index(Bitboard occ) const {
        return unsigned(((occ & mask) * magic) >> shift);
    }
};

extern Magic RookMagics[SQUARE_NB];
extern Magic BishopMagics[SQUARE_NB];

// ---------------------------------------------------------------------------
// Sliding attacks given a board occupancy. PieceType must be BISHOP or ROOK;
// queen = bishop | rook.
// ---------------------------------------------------------------------------
inline Bitboard bishop_attacks(Square s, Bitboard occ) {
    const Magic& m = BishopMagics[s];
    return m.attacks[m.index(occ)];
}

inline Bitboard rook_attacks(Square s, Bitboard occ) {
    const Magic& m = RookMagics[s];
    return m.attacks[m.index(occ)];
}

// Generic dispatch used by movegen / evaluation.
template <PieceType Pt>
inline Bitboard attacks_bb(Square s, Bitboard occ) {
    static_assert(Pt == BISHOP || Pt == ROOK || Pt == QUEEN, "unsupported slider");
    return Pt == BISHOP ? bishop_attacks(s, occ)
         : Pt == ROOK   ? rook_attacks(s, occ)
                        : bishop_attacks(s, occ) | rook_attacks(s, occ);
}

// Non-templated form (runtime piece type). Handles leapers and sliders.
inline Bitboard attacks_bb(PieceType pt, Square s, Bitboard occ) {
    switch (pt) {
        case BISHOP: return bishop_attacks(s, occ);
        case ROOK:   return rook_attacks(s, occ);
        case QUEEN:  return bishop_attacks(s, occ) | rook_attacks(s, occ);
        default:     return PseudoAttacks[pt][s];  // KNIGHT / KING
    }
}

inline Bitboard pawn_attacks(Color c, Square s) { return PawnAttacks[c][s]; }

}  // namespace engine

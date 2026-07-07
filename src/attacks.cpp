// attacks.cpp — Precomputed piece attack lookups (magic bitboards for sliders).
//
// Implements everything declared in attacks.hpp: the leaper/pseudo attack
// tables and the runtime magic-bitboard initialization for sliders.

#include "attacks.hpp"

#include <cstdlib>   // std::abs (used via bitboard.hpp distance helpers)

namespace engine {

// ---------------------------------------------------------------------------
// Table definitions (declared extern in attacks.hpp)
// ---------------------------------------------------------------------------
Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];

Magic RookMagics[SQUARE_NB];
Magic BishopMagics[SQUARE_NB];

// Shared attack pools (standard Stockfish fancy-magic sizes).
namespace {

Bitboard RookTable[0x19000];
Bitboard BishopTable[0x1480];

// Self-contained single-bit board. SquareBB[] is populated by Bitboards::init()
// which runs AFTER Attacks::init(), so square_bb() would return 0 here — we must
// compute the bit directly.
Bitboard sq_bb(Square s) { return Bitboard(1) << s; }

// Self-contained Chebyshev distance — SquareDistance[] is not yet populated
// during Attacks::init() (it is filled later by Bitboards::init()), so we must
// not rely on the distance() helper here.
int sq_distance(Square a, Square b) {
    int df = std::abs(file_of(a) - file_of(b));
    int dr = std::abs(rank_of(a) - rank_of(b));
    return df > dr ? df : dr;
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift64*) with a fixed seed for reproducible magics.
// ---------------------------------------------------------------------------
class PRNG {
    uint64_t s;

    uint64_t rand64() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }

   public:
    explicit PRNG(uint64_t seed) : s(seed) {}

    Bitboard rand() { return rand64(); }

    // Sparse random — good candidate magics have few set bits.
    Bitboard sparse_rand() { return rand64() & rand64() & rand64(); }
};

// ---------------------------------------------------------------------------
// Reference sliding attack: ray-walk from `sq` in each direction, stopping at
// board edges (detected via one-step distance) or blockers in `occupied`.
// ---------------------------------------------------------------------------
Bitboard sliding_attack(PieceType pt, Square sq, Bitboard occupied) {
    Bitboard attacks = 0;

    const Direction rookDirs[4]   = {NORTH, SOUTH, EAST, WEST};
    const Direction bishopDirs[4] = {NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST};
    const Direction* dirs = (pt == ROOK) ? rookDirs : bishopDirs;

    for (int i = 0; i < 4; ++i) {
        Direction d = dirs[i];
        Square s = sq;
        // Step while the next square is on-board (adjacent to current) and
        // the current square is not a blocker.
        while (is_ok(s + d) && sq_distance(s, s + d) == 1) {
            s += d;
            attacks |= sq_bb(s);
            if (occupied & sq_bb(s))
                break;
        }
    }
    return attacks;
}

// ---------------------------------------------------------------------------
// Fill a slider magic table for one piece type, finding magics that work with
// Magic::index() as defined in the header.
// ---------------------------------------------------------------------------
void init_magics(PieceType pt, Bitboard table[], Magic magics[]) {
    // Optimal PRNG seeds per rank (from Stockfish) give fast, deterministic
    // convergence for the magic search. The PRNG is reseeded for each square
    // based on that square's rank.
    const int seeds[RANK_NB] = {728, 10316, 55013, 32803, 12281, 15100, 16645, 255};

    Bitboard occupancy[4096];
    Bitboard reference[4096];
    int epoch[4096] = {};
    int cnt = 0;

    Bitboard* attacksPtr = table;

    for (Square s = SQ_A1; s <= SQ_H8; s += EAST) {
        // Board edges NOT relevant to `s`: edges the ray reaches anyway.
        Bitboard edges = ((Rank1BB | Rank8BB) & ~rank_bb(s)) |
                         ((FileABB | FileHBB) & ~file_bb(s));

        Magic& m = magics[s];
        m.mask  = sliding_attack(pt, s, 0) & ~edges;
        m.shift = 64 - popcount(m.mask);

        // Contiguous placement in the pool: this square's block starts right
        // after the previous square's block.
        m.attacks = (s == SQ_A1) ? table : magics[s - 1].attacks + (1 << popcount(magics[s - 1].mask));
        attacksPtr = m.attacks;

        // Enumerate every occupancy subset of the mask via carry-rippler.
        Bitboard b = 0;
        int size = 0;
        do {
            occupancy[size] = b;
            reference[size] = sliding_attack(pt, s, b);
            size++;
            b = (b - m.mask) & m.mask;
        } while (b);

        // Reseed per square (by rank) for reliable, deterministic convergence.
        PRNG rng(seeds[rank_of(s)]);

        // Search for a magic that maps subsets collision-free.
        for (int i = 0; i < size;) {
            for (m.magic = 0; popcount((m.magic * m.mask) >> 56) < 6;)
                m.magic = rng.sparse_rand();

            // Verify: on a fresh epoch, no two distinct references collide.
            ++cnt;
            for (i = 0; i < size; ++i) {
                unsigned idx = m.index(occupancy[i]);

                if (epoch[idx] < cnt) {
                    epoch[idx] = cnt;
                    attacksPtr[idx] = reference[i];
                } else if (attacksPtr[idx] != reference[i]) {
                    break;  // collision — try another magic
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Leaper attack computation for knight/king via offset steps + wrap masking.
// ---------------------------------------------------------------------------
Bitboard leaper_attacks(Square s, const int steps[], int nSteps) {
    Bitboard attacks = 0;
    for (int i = 0; i < nSteps; ++i) {
        Square to = Square(int(s) + steps[i]);
        // Reject wraps: destination must be on-board and within Chebyshev
        // distance 2 (knight) of the source.
        if (is_ok(to) && sq_distance(s, to) <= 2)
            attacks |= sq_bb(to);
    }
    return attacks;
}

}  // namespace

// ---------------------------------------------------------------------------
// Attacks::init — populate all tables. Must run before any attack query and
// before Bitboards::init().
// ---------------------------------------------------------------------------
void Attacks::init() {
    // Pawn captures.
    for (Square s = SQ_A1; s <= SQ_H8; s += EAST) {
        Bitboard b = sq_bb(s);
        PawnAttacks[WHITE][s] = pawn_attacks_bb<WHITE>(b);
        PawnAttacks[BLACK][s] = pawn_attacks_bb<BLACK>(b);
    }

    // Knight and king leaper steps.
    const int knightSteps[8] = {17, 15, 10, 6, -17, -15, -10, -6};
    const int kingSteps[8]   = {9, 8, 7, 1, -9, -8, -7, -1};

    for (Square s = SQ_A1; s <= SQ_H8; s += EAST) {
        PseudoAttacks[KNIGHT][s] = leaper_attacks(s, knightSteps, 8);
        PseudoAttacks[KING][s]   = leaper_attacks(s, kingSteps, 8);
    }

    // Slider magics.
    init_magics(ROOK, RookTable, RookMagics);
    init_magics(BISHOP, BishopTable, BishopMagics);

    // Pseudo (empty-board) slider attacks.
    for (Square s = SQ_A1; s <= SQ_H8; s += EAST) {
        PseudoAttacks[BISHOP][s] = bishop_attacks(s, 0);
        PseudoAttacks[ROOK][s]   = rook_attacks(s, 0);
        PseudoAttacks[QUEEN][s]  = PseudoAttacks[BISHOP][s] | PseudoAttacks[ROOK][s];
    }
}

}  // namespace engine

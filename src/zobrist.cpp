// zobrist.cpp — Zobrist key table definitions and deterministic init.
//
// Fills every table with values from a fixed-seed xorshift64 PRNG so hashes are
// reproducible across runs. psq[NO_PIECE][*] is left zero (never XOR-ed in).

#include "zobrist.hpp"

namespace engine {

namespace Zobrist {

Key psq[PIECE_NB][SQUARE_NB];
Key enpassant[FILE_NB];
Key castling[CASTLING_RIGHT_NB];
Key side;

namespace {

// Simple deterministic xorshift64 PRNG with a hardcoded seed.
class PRNG {
    uint64_t s;

   public:
    explicit PRNG(uint64_t seed) : s(seed) {}

    uint64_t rand64() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
};

}  // namespace

void init() {
    PRNG rng(1070372ULL);

    // Leave psq[NO_PIECE][*] == 0; fill all real pieces.
    for (int pc = 0; pc < PIECE_NB; ++pc)
        for (Square s = SQ_A1; s <= SQ_H8; s += EAST) {
            if (pc == NO_PIECE)
                psq[pc][s] = 0;
            else
                psq[pc][s] = rng.rand64();
        }

    for (File f = FILE_A; f < FILE_NB; f = File(f + 1))
        enpassant[f] = rng.rand64();

    for (int cr = 0; cr < CASTLING_RIGHT_NB; ++cr)
        castling[cr] = rng.rand64();

    side = rng.rand64();
}

}  // namespace Zobrist

}  // namespace engine

// zobrist.hpp — Zobrist hashing keys for transposition/repetition detection.
//
// Keys are random 64-bit values XOR-ed incrementally as the position changes.
// Zobrist::init() fills the tables deterministically (fixed PRNG seed) so hashes
// are reproducible across runs.
#pragma once

#include "types.hpp"

namespace engine {

namespace Zobrist {

extern Key psq[PIECE_NB][SQUARE_NB];   // piece on square
extern Key enpassant[FILE_NB];         // en-passant file (when a capture is possible)
extern Key castling[CASTLING_RIGHT_NB];// castling-rights mask
extern Key side;                       // side-to-move is black

void init();

}  // namespace Zobrist

}  // namespace engine

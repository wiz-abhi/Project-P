// book.hpp — Polyglot opening-book support.
//
// Provides:
//   - polyglot_key(pos): the standard Polyglot Zobrist hash of a position
//     (independent of the engine's own internal Zobrist hashing).
//   - Book: a reader for Polyglot ".bin" files (16-byte big-endian entries),
//     with binary-search probing and legal-move translation.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "types.hpp"

namespace engine {

class Position;

// Standard Polyglot 64-bit hash key for a position.
uint64_t polyglot_key(const Position& pos);

class Book {
   public:
    // Load a Polyglot .bin file. Returns true on success. Replaces any book
    // previously loaded. An empty/failed load leaves the book unloaded.
    bool load(const std::string& path);

    // Is a (non-empty) book currently loaded?
    bool loaded() const { return !entries_.empty(); }

    // Probe the book for pos. Returns a legal engine Move drawn from the book,
    // or Move::none() if the position is not in the book (or no book entry
    // translates to a legal move). If pickBest, the highest-weighted entry is
    // chosen; otherwise a weighted-random entry is chosen (PRNG seeded from the
    // polyglot key, so probing is deterministic for a given position).
    Move probe(const Position& pos, bool pickBest) const;

   private:
    struct Entry {
        uint64_t key;
        uint16_t move;
        uint16_t weight;
        uint32_t learn;
    };

    std::vector<Entry> entries_;  // sorted ascending by key
};

}  // namespace engine

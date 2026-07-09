/*
  Wizard — Polyglot opening-book support.

  Wizard is a derivative of Stockfish 15.1 and is distributed under the GNU
  General Public License version 3 (see NOTICE and the license header in the
  Stockfish source files).

  This module adds native Polyglot ".bin" opening-book support:
    - polyglot_key(pos): the standard Polyglot Zobrist hash of a position
      (independent of Stockfish's own internal Zobrist key).
    - PolyglotBook: a reader for Polyglot books (16-byte big-endian entries)
      with binary-search probing and legal-move translation.
*/

#ifndef BOOK_H_INCLUDED
#define BOOK_H_INCLUDED

#include <cstdint>
#include <string>
#include <vector>

#include "types.h"

namespace Stockfish {

class Position;

// Standard Polyglot 64-bit hash key for a position.
uint64_t polyglot_key(const Position& pos);

class PolyglotBook {
 public:
  // Load a Polyglot .bin file. Returns true on success, replacing any book
  // previously loaded. An empty path or a failed load leaves the book unloaded.
  bool load(const std::string& path);

  bool loaded() const { return !entries.empty(); }
  void clear() { entries.clear(); }

  // Probe the book for pos. Returns a legal Move drawn from the book, or
  // MOVE_NONE if the position is not in the book (or no entry translates to a
  // legal move). If pickBest, the highest-weighted entry is chosen; otherwise a
  // weighted-random entry is chosen (PRNG seeded from the Polyglot key, so the
  // choice is deterministic for a given position).
  Move probe(const Position& pos, bool pickBest) const;

 private:
  struct Entry {
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;
  };

  std::vector<Entry> entries;  // sorted ascending by key (Polyglot spec)
};

// Global book instance, loaded via the "BookFile" UCI option and probed from
// the UCI "go" handler when "OwnBook" is enabled.
extern PolyglotBook Book;

}  // namespace Stockfish

#endif  // #ifndef BOOK_H_INCLUDED

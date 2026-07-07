// movegen.hpp — Legal / pseudo-legal move generation interface.
//
// generate<Type>() appends moves to a raw buffer and returns the new end pointer.
// MoveList<Type> wraps it for range-based iteration. ExtMove pairs a move with a
// score slot used by move ordering.
#pragma once

#include "position.hpp"
#include "types.hpp"

namespace engine {

enum GenType {
    CAPTURES,      // captures + queen promotions
    QUIETS,        // non-captures, non-promotions (and under-promotions handled in ALL)
    QUIET_CHECKS,  // quiet moves that give check
    EVASIONS,      // legal replies while in check (king moves, blocks, captures)
    NON_EVASIONS,  // all pseudo-legal moves when not in check
    LEGAL          // strictly legal moves (filtered)
};

struct ExtMove {
    Move move;
    int  value;   // move-ordering score

    operator Move() const { return move; }
    void operator=(Move m) { move = m; }
    // Prevent implicit conversion from ExtMove -> bool via Move::operator bool
    operator float() const = delete;
};

inline bool operator<(const ExtMove& a, const ExtMove& b) { return a.value < b.value; }

template <GenType Type>
ExtMove* generate(const Position& pos, ExtMove* moveList);

// Range-adaptor over generate<Type>() with an on-stack buffer.
template <GenType Type>
struct MoveList {
    explicit MoveList(const Position& pos) : last(generate<Type>(pos, moveList)) {}
    const ExtMove* begin() const { return moveList; }
    const ExtMove* end() const { return last; }
    size_t size() const { return last - moveList; }
    bool contains(Move m) const {
        for (const ExtMove* it = moveList; it != last; ++it)
            if (it->move == m) return true;
        return false;
    }

   private:
    ExtMove moveList[MAX_MOVES];
    ExtMove* last;
};

}  // namespace engine

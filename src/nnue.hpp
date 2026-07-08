// nnue.hpp — Classic HalfKP / SFNNv1 NNUE evaluation (256x2-32-32-1).
//
// Two evaluation paths are provided:
//
//   * Full-refresh: NNUE::evaluate(pos) recomputes both accumulators from
//     scratch every call, so it is a PURE function of the Position (fully
//     thread-safe, no per-thread state). Kept as the correctness reference and
//     as a fallback for the leaf/qsearch entry points.
//
//   * Incremental ("UE" of NNUE): an Accumulator holds the two per-perspective
//     256-wide feature sums. The search maintains a stack of accumulators and
//     derives each child from its parent by applying only the feature deltas of
//     the move just played (add/sub of the moved/captured/promoted pieces), or
//     refreshing a single perspective when its king moves. This recovers the
//     ~4x cost of rebuilding both accumulators at every node.
#pragma once

#include <cstdint>
#include <string>

#include "position.hpp"
#include "types.hpp"

namespace engine {

namespace NNUE {

// Number of neurons per perspective in the feature transformer output.
constexpr int kAccDimensions = 256;

// An incrementally-updated pair of feature-transformer accumulators, one per
// perspective. acc[P][i] == ft_bias[i] + Σ over all non-king pieces of
// ft_weight[make_index(P, king_sq(P), piece, piece_sq)][i].
//
// `computed[P]` records whether that perspective's sum currently reflects the
// position it belongs to (set by refresh/update; used only for debugging/asserts
// — the search always keeps both perspectives valid).
struct Accumulator {
    alignas(64) int16_t acc[COLOR_NB][kAccDimensions];
    bool computed[COLOR_NB] = {false, false};
};

// Parse a HalfKP .nnue file into the static network arrays. Returns true on
// success (and sets loaded()==true). On failure returns false and leaves the
// previously loaded state intact iff parsing fails before overwriting.
bool  load(const std::string& path);

// Whether a network is currently loaded and usable.
bool  loaded();

// Side-to-move-relative evaluation in centipawns (full refresh). Requires
// loaded()==true. Pure function of the Position.
Value evaluate(const Position& pos);

// ---- Incremental API ------------------------------------------------------

// Recompute perspective C of `acc` from scratch from the given position.
void  refresh_perspective(Accumulator& acc, const Position& pos, Color c);

// Recompute both perspectives of `acc` from scratch from the given position.
void  refresh(Accumulator& acc, const Position& pos);

// Run the 512->32->32->1 network over an already-built accumulator, returning
// the side-to-move-relative score in centipawns. `stm` is the side to move in
// the position the accumulator describes.
Value evaluate(const Accumulator& acc, Color stm);

// Derive `child` from `parent` by applying the single move `m`. MUST be called
// on the PRE-MOVE position `before` (the parent position, immediately BEFORE
// pos.do_move(m)). Handles quiet moves, captures, en-passant, promotions,
// castling and king moves (single-perspective refresh) exactly.
void  update(Accumulator& child, const Accumulator& parent,
             const Position& before, Move m);

}  // namespace NNUE

}  // namespace engine

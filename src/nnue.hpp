// nnue.hpp — NNUE evaluation supporting TWO architectures, detected at load:
//
//   * Classic HalfKP / SFNNv1 (256x2-32-32-1), version 0x7AF32F16.
//     41024 inputs (king-square × non-king piece-square), 180°-rotation
//     orientation for Black. Kept as the fallback architecture.
//
//   * Modern HalfKAv2_hm / SFNNv4 (SF15-era), version 0x7AF32F20.
//     22528 inputs (32 king buckets on files e-h with horizontal mirroring ×
//     11 piece planes × 64 squares — BOTH kings are features), feature
//     transformer 22528->1024 per perspective with 8 PSQT buckets (i32), and
//     8 bucketed layer stacks of 1024->(15+1)->32->1 with a skip connection
//     from the 16th fc_0 neuron. The FT activation is the pairwise clipped
//     multiplication out[j] = clip(acc[j])·clip(acc[j+512])/128 (512 outputs
//     per perspective).
//
// Two evaluation paths are provided for either architecture:
//
//   * Full-refresh: NNUE::evaluate(pos) recomputes both accumulators from
//     scratch every call, so it is a PURE function of the Position (fully
//     thread-safe, no per-thread state). Kept as the correctness reference and
//     as a fallback for the leaf/qsearch entry points.
//
//   * Incremental ("UE" of NNUE): an Accumulator holds the two per-perspective
//     feature sums (+ PSQT sums for the modern arch). The search maintains a
//     stack of accumulators and derives each child from its parent by applying
//     only the feature deltas of the move just played, or refreshing a single
//     perspective when its own king moves.
#pragma once

#include <cstdint>
#include <string>

#include "position.hpp"
#include "types.hpp"

namespace engine {

namespace NNUE {

// Feature-transformer widths per perspective. The Accumulator is sized for the
// larger (modern) architecture; the HalfKP path only uses the first 256 lanes.
constexpr int kAccDimensionsHalfKP = 256;
constexpr int kAccDimensionsV4     = 1024;
constexpr int kPsqtBuckets         = 8;

// An incrementally-updated pair of feature-transformer accumulators, one per
// perspective, plus (modern arch only) the per-perspective PSQT bucket sums.
//
// `computed[P]` records whether that perspective's sum currently reflects the
// position it belongs to (set by refresh/update; used only for debugging/asserts
// — the search always keeps both perspectives valid).
struct Accumulator {
    alignas(64) int16_t acc[COLOR_NB][kAccDimensionsV4];
    int32_t psqt[COLOR_NB][kPsqtBuckets];
    bool computed[COLOR_NB] = {false, false};
};

// Parse a .nnue file into the static network arrays, detecting the architecture
// from the header version. Returns true on success (and sets loaded()==true).
// If the file fails to load/parse and NO network is currently loaded, attempts
// to fall back to the bundled HalfKP net so the engine is never netless.
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

// Run the layer stack over an already-built accumulator, returning the
// side-to-move-relative score in centipawns. `pos` must be the position the
// accumulator describes (used for the side to move and, on the modern arch,
// the material-count bucket selection).
Value evaluate(const Accumulator& acc, const Position& pos);

// Derive `child` from `parent` by applying the single move `m`. MUST be called
// on the PRE-MOVE position `before` (the parent position, immediately BEFORE
// pos.do_move(m)). Handles quiet moves, captures, en-passant, promotions,
// castling and king moves (single-perspective refresh) exactly.
void  update(Accumulator& child, const Accumulator& parent,
             const Position& before, Move m);

}  // namespace NNUE

}  // namespace engine

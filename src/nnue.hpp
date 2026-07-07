// nnue.hpp — Classic HalfKP / SFNNv1 NNUE evaluation (256x2-32-32-1).
//
// Full-refresh evaluator: NNUE::evaluate(pos) is a PURE function of the Position
// (it recomputes both accumulators from scratch every call), so it is fully
// thread-safe and requires no per-thread state.
#pragma once

#include <string>

#include "position.hpp"
#include "types.hpp"

namespace engine {

namespace NNUE {

// Parse a HalfKP .nnue file into the static network arrays. Returns true on
// success (and sets loaded()==true). On failure returns false and leaves the
// previously loaded state intact iff parsing fails before overwriting.
bool  load(const std::string& path);

// Whether a network is currently loaded and usable.
bool  loaded();

// Side-to-move-relative evaluation in centipawns. Requires loaded()==true.
Value evaluate(const Position& pos);

}  // namespace NNUE

}  // namespace engine

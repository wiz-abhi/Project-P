// eval.hpp — Static position evaluation.
//
// evaluate() returns a score in centipawns from the SIDE-TO-MOVE's point of view
// (positive = good for the player to move). Phase 2 ships a tapered hand-crafted
// evaluation (material + piece-square tables); later phases extend it and add NNUE.
#pragma once

#include "position.hpp"
#include "types.hpp"

namespace engine {

namespace Eval {

void  init();                          // initialize PSQT / precomputed tables
Value evaluate(const Position& pos);   // side-to-move relative score

}  // namespace Eval

}  // namespace engine

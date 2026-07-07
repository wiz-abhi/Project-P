// search.hpp — Search driver: iterative deepening + alpha-beta + quiescence.
//
// Phase 2 is single-threaded and includes a transposition table, MVV-LVA capture
// ordering, killer/history heuristics, and time management. Later phases layer on
// PVS, aspiration windows, null-move pruning, LMR/LMP, futility, and SMP threads.
#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "position.hpp"
#include "types.hpp"

namespace engine {

// Parsed "go" parameters.
struct SearchLimits {
    int64_t nodes    = 0;             // node cap (0 = none)
    int     depth    = 0;             // fixed depth (0 = none)
    int     movetime = 0;             // fixed ms this move (0 = none)
    int     time[COLOR_NB] = {0, 0};  // wtime / btime
    int     inc[COLOR_NB]  = {0, 0};  // winc / binc
    int     movestogo = 0;
    bool    infinite  = false;
    std::vector<Move> searchmoves;    // restrict root to these (empty = all)
};

// Time budget derived from SearchLimits for the current move.
class TimeManager {
   public:
    void init(const SearchLimits& limits, Color us, int ply);
    int64_t optimum() const { return optimumMs; }
    int64_t maximum() const { return maximumMs; }
    int64_t elapsed() const;   // ms since init()
    void    start();

   private:
    int64_t startTime = 0;
    int64_t optimumMs = 0;
    int64_t maximumMs = 0;
};

namespace Search {

// Global stop flag: set by the UCI "stop"/"quit" handler or by the time manager.
extern std::atomic<bool> stop;

// Aggregate search statistics for UCI info output.
struct Stats {
    int64_t nodes = 0;
    int     seldepth = 0;
};
extern Stats stats;

void init();    // one-time setup
void clear();   // reset TT + heuristic tables (called on ucinewgame)

// Run iterative deepening on pos under limits. Prints UCI "info" lines each
// iteration and a final "bestmove". Returns the best move found.
Move think(Position& pos, const SearchLimits& limits);

}  // namespace Search

}  // namespace engine

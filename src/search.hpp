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
    bool    ponder    = false;        // "go ponder": search now, defer the clock
    std::vector<Move> searchmoves;    // restrict root to these (empty = all)
};

// Time budget derived from SearchLimits for the current move.
class TimeManager {
   public:
    void init(const SearchLimits& limits, Color us, int ply);
    // optimum()/maximum() are read on the search thread(s); a concurrent
    // ponderhit() on the UCI thread rewrites them via restart(). Acquire loads
    // here pair with the release store at the end of compute_budget() so that a
    // reader seeing a freshly-finite budget also sees the freshly-reset clock
    // origin (never a stale startTime → never a spurious early stop).
    int64_t optimum() const { return optimumMs.load(std::memory_order_acquire); }
    int64_t maximum() const { return maximumMs.load(std::memory_order_acquire); }
    int64_t elapsed() const;   // ms since the clock origin
    void    start();

    // Restart the clock at NOW and (re)derive the budget from the limits/color/ply
    // captured at init() time. Called from Search::ponderhit() to begin the timed
    // phase the instant the opponent plays the predicted move.
    void    restart();

   private:
    // Compute optimum/maximum from the stored limits + us + ply, assuming the
    // clock origin (startTime) is already set. Shared by init() and restart().
    void    compute_budget();

    std::atomic<int64_t> startTime{0};
    std::atomic<int64_t> optimumMs{0};
    std::atomic<int64_t> maximumMs{0};

    // Snapshot of the parameters init() was called with, so restart() can recompute
    // the budget from the same wtime/btime/winc/binc after a ponderhit.
    SearchLimits savedLimits;
    Color        savedUs  = WHITE;
    int          savedPly = 0;
};

namespace Search {

// Global stop flag: set by the UCI "stop"/"quit" handler or by the time manager.
extern std::atomic<bool> stop;

// Pondering flag: set when a search was launched with "go ponder". While true the
// search ignores the clock (searches like "infinite") and never terminates on the
// time budget. The UCI "ponderhit" handler flips it false via ponderhit(), which
// also starts the clock from that instant.
extern std::atomic<bool> pondering;

// Transition from pondering to a normal timed search: clears the pondering flag,
// resets the clock origin to NOW, and (re)computes the optimum/maximum time
// budget from the limits the "go ponder" command carried. The already-running
// search thread then begins honouring the clock. Safe to call only while a
// ponder search is in flight.
void ponderhit();

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

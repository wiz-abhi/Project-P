// search.cpp — Modern search stack: iterative deepening + PVS negamax + qsearch.
//
// Lazy SMP multithreaded search on top of the Phase-3 single-threaded core.
// On top of the Phase-2 core (transposition table, MVV-LVA capture ordering,
// killer/history heuristics, check extension, mate-distance pruning, time
// management) this layers on the full modern search stack, following well-known
// Stockfish/CPW formulations:
//
//   * Principal Variation Search (PVS) with null-window re-searches
//   * Aspiration windows in iterative deepening
//   * Null-move pruning (with zugzwang guard + verification at high depth)
//   * Reverse futility / static null-move pruning
//   * Futility pruning of quiet moves at low depth
//   * Late Move Reductions (log-based) with re-search on fail-high
//   * Late Move Pruning (movecount-based)
//   * SEE-based capture ordering and SEE pruning
//   * Countermove heuristic + butterfly history + capture history
//
// Parallelism uses the Lazy SMP model: N worker threads each run their own
// iterative deepening on a private copy of the root position, all sharing the
// (lockless) transposition table. Search-local state — node/seldepth counters,
// killers, history/counter-move/capture-history tables, the search stack,
// triangular PV, StateInfo pool and the Position itself — is PER-THREAD, held in
// struct Thread. Only the TT and the global Search::stop flag are shared.
//
// See search.hpp for the public interface.

#include "search.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "eval.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "position.hpp"
#include "tt.hpp"
#include "types.hpp"
#include "uci.hpp"

namespace engine {

// "Use NNUE" toggle (defined in eval.cpp). When true and a net is loaded, the
// search uses its incrementally-updated accumulator stack for static eval.
extern bool UseNNUE;

// Move Overhead (ms) shared with the UCI option handler; TimeManager subtracts it.
extern int MoveOverheadMs;

namespace Search {

std::atomic<bool> stop{false};
std::atomic<bool> pondering{false};   // true while searching under "go ponder"
Stats             stats;   // aggregate (summed) stats for UCI reporting

// Number of search threads (Lazy SMP workers). Set by the UCI "Threads" option
// via set_threads(); defaults to 1 (single-threaded, behaviourally identical to
// the previous engine).
int Threads = 1;

void set_threads(int n) { Threads = std::max(1, n); }

}  // namespace Search

// ---------------------------------------------------------------------------
// Root position description, stashed by the UCI layer so each worker can build
// its own Position (with an intact StateInfo history for repetition / 50-move
// detection) from the root FEN + game moves. Set from uci.cpp before think().
// ---------------------------------------------------------------------------
namespace Search {

struct RootInfo {
    std::string fen;
    std::vector<Move> moves;
};

RootInfo rootInfo;

void set_root(const std::string& fen, const std::vector<Move>& moves) {
    rootInfo.fen   = fen;
    rootInfo.moves = moves;
}

}  // namespace Search

// ---------------------------------------------------------------------------
// Clock helpers
// ---------------------------------------------------------------------------
namespace {

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

void TimeManager::start() {
    startTime.store(now_ms(), std::memory_order_release);
}

int64_t TimeManager::elapsed() const {
    return now_ms() - startTime.load(std::memory_order_acquire);
}

void TimeManager::init(const SearchLimits& limits, Color us, int ply) {
    start();

    // Remember what we were asked for so a later ponderhit can re-derive the
    // budget from the same clock parameters (measured from the ponderhit instant).
    savedLimits = limits;
    savedUs     = us;
    savedPly    = ply;

    compute_budget();
}

void TimeManager::restart() {
    // Reset the clock origin to NOW, then recompute the budget from the stored
    // limits. Called on ponderhit(); the running search sees a fresh clock.
    start();
    compute_budget();
}

void TimeManager::compute_budget() {
    const SearchLimits& limits = savedLimits;
    const Color         us     = savedUs;

    // Publish the budget with the right ordering: write optimum first, then
    // maximum with release, because search-thread readers gate on maximum()
    // (acquire). A reader that sees a freshly-finite maximum is then guaranteed to
    // also see the matching optimum and the startTime that start() released just
    // before — so a ponderhit transition can never produce a stale early stop.
    auto publish = [this](int64_t opt, int64_t mx) {
        optimumMs.store(opt, std::memory_order_relaxed);
        maximumMs.store(mx, std::memory_order_release);
    };

    // While pondering, the clock does not run: search unbounded (like "infinite").
    // ponderhit() clears Search::pondering and calls restart() to recompute a real
    // budget from this same point.
    if (Search::pondering.load(std::memory_order_relaxed)) {
        publish(INT64_MAX / 4, INT64_MAX / 4);
        return;
    }

    const int64_t overhead = MoveOverheadMs;

    // Fixed / unlimited modes: rely on depth/node/stop rather than the clock.
    if (limits.movetime > 0) {
        int64_t mt = std::max<int64_t>(1, limits.movetime - overhead);
        publish(mt, mt);
        return;
    }
    if (limits.depth > 0 || limits.nodes > 0 || limits.infinite ||
        (limits.time[us] == 0 && limits.inc[us] == 0)) {
        // No usable clock info -> effectively unlimited (a depth/node/stop
        // limit is expected to terminate the search).
        publish(INT64_MAX / 4, INT64_MAX / 4);
        return;
    }

    // Allocation from remaining time + increment, hardened for bullet where
    // fixed per-move latency (network + bridge) can flag us in long games.
    const int64_t time = limits.time[us];
    const int64_t inc  = limits.inc[us];

    // Moves-to-go horizon. Bullet games routinely run 60-80 plies, so a sudden-
    // death game must budget for many more moves than the classic "30" or it
    // starves the endgame (the exact cause of our flag losses). Assume ~40 of
    // our own moves remain; cap a provided movestogo the same way.
    const int64_t mtg = limits.movestogo > 0 ? std::min<int64_t>(limits.movestogo, 40) : 40;

    // Time safely usable, after reserving the transmission overhead for THIS move.
    const int64_t remaining = std::max<int64_t>(1, time - overhead);

    // Optimum: an even slice of the remaining clock plus most of the increment
    // (self-scaling: as remaining shrinks each move, so does the slice).
    int64_t opt = remaining / mtg + int64_t(inc * 0.75);
    // Maximum for one move: a small multiple of optimum and a small fraction of
    // the clock, so no single fail-low can dump the whole clock.
    int64_t mx = std::min<int64_t>(opt * 3, remaining / 4);

    // Low-time safety net: when the clock is short, move almost instantly and let
    // the increment carry us. This is what stops the endgame flagging in bullet.
    if (remaining < 20 * overhead) {                       // ~8s at Overhead 400
        const int64_t floorSpend = inc > 0 ? int64_t(inc * 0.5) : remaining / 30;
        opt = std::min(opt, floorSpend);
        mx  = std::min<int64_t>(mx, inc > 0 ? inc : remaining / 15);
    }

    opt = std::clamp<int64_t>(opt, 1, remaining);
    mx  = std::clamp<int64_t>(mx, opt, remaining);

    publish(opt, mx);
}

// ---------------------------------------------------------------------------
// Search internals
// ---------------------------------------------------------------------------
namespace {

using namespace Search;

// Shared (read-only after init) search configuration for the current "go".
TimeManager  Time;        // owned/read by the main thread's time management
SearchLimits Limits;
Color        rootColor;

// Late Move Reduction table: reductions[depth][moveCount]. Read-only during search.
int   Reductions[MAX_PLY + 1][MAX_MOVES];

constexpr int64_t CheckEveryNodes = 2048;

// Piece values for MVV-LVA / delta pruning, indexed by PieceType.
constexpr Value PieceValueOrder[PIECE_TYPE_NB] = {
    0, PawnValue, KnightValue, BishopValue, RookValue, QueenValue, 0, 0};

void init_reductions() {
    for (int d = 0; d <= MAX_PLY; ++d)
        for (int mc = 0; mc < MAX_MOVES; ++mc) {
            if (d == 0 || mc == 0) {
                Reductions[d][mc] = 0;
                continue;
            }
            double r = 0.75 + std::log(double(d)) * std::log(double(mc)) / 2.25;
            Reductions[d][mc] = int(r);
        }
}

// Adjust a mate score for storing into / reading out of the TT. Inside the TT
// mate scores are relative to the node; at the node we need them relative to the
// root, so we shift by ply.
Value value_to_tt(Value v, int ply) {
    if (v >= VALUE_MATE_IN_MAX_PLY)  return v + ply;
    if (v <= VALUE_MATED_IN_MAX_PLY) return v - ply;
    return v;
}

Value value_from_tt(Value v, int ply) {
    if (v == VALUE_NONE) return VALUE_NONE;
    if (v >= VALUE_MATE_IN_MAX_PLY)  return v - ply;
    if (v <= VALUE_MATED_IN_MAX_PLY) return v + ply;
    return v;
}

// MVV-LVA score for a capture / promotion.
int mvv_lva(const Position& pos, Move m) {
    PieceType victim   = type_of(pos.piece_on(m.to_sq()));
    PieceType attacker = type_of(pos.moved_piece(m));
    int score = 0;
    if (m.type_of() == EN_PASSANT)
        victim = PAWN;
    if (m.type_of() == PROMOTION)
        score += PieceValueOrder[m.promotion_type()];
    return score + PieceValueOrder[victim] * 16 - PieceValueOrder[attacker];
}

// Generic "gravity" history update: keep the value bounded around +-Max.
void update_stat(int& entry, int bonus, int Max) {
    bonus = std::clamp(bonus, -Max, Max);
    entry += bonus - entry * std::abs(bonus) / Max;
}

int history_bonus(int depth) {
    return std::min(2 * depth * depth + 16 * depth, 1200);
}

// ---------------------------------------------------------------------------
// Per-thread search state. One Thread per Lazy-SMP worker. All formerly-global
// mutable search state lives here so workers never touch each other's data (the
// TT is the only shared, deliberately-lockless structure).
// ---------------------------------------------------------------------------
struct Thread {
    int  id = 0;               // 0 == main thread
    bool isMain() const { return id == 0; }

    // Own board, rebuilt from the root FEN + game moves.
    Position  pos;
    StateInfo rootStates[MAX_PLY + 4 + 512];  // history chain for the root line
    int       rootStateCount = 0;

    // Node / seldepth counters (summed across threads for reporting).
    int64_t nodes    = 0;
    int     seldepth = 0;

    // Per-thread sticky abort flag (mirrors the global stop).
    bool stopped = false;

    // Heuristic tables (per-thread; deliberately not shared).
    Move killers[MAX_PLY + 2][2];
    int  history[COLOR_NB][SQUARE_NB][SQUARE_NB];           // butterfly [c][from][to]
    int  captureHistory[PIECE_NB][SQUARE_NB][PIECE_TYPE_NB]; // [piece][to][captured pt]
    Move counterMoves[PIECE_NB][SQUARE_NB];                // [prev piece][prev to]

    // Triangular PV table.
    Move pvTable[MAX_PLY + 1][MAX_PLY + 1];
    int  pvLength[MAX_PLY + 1];

    // StateInfo pool used during search, one per ply.
    StateInfo states[MAX_PLY + 4];

    // Incrementally-updated NNUE accumulator stack, one slot per ply. accStack[0]
    // is refreshed from the root; each child slot is derived from its parent via
    // NNUE::update just before the corresponding do_move (see search/qsearch).
    // qsearch may descend to MAX_PLY, so keep a little slack.
    NNUE::Accumulator accStack[MAX_PLY + 5];

    // Whether the incremental NNUE path is active for this search (net loaded and
    // the UCI "Use NNUE" option enabled). Cached at new_search() time.
    bool nnueActive = false;

    // Static evaluation at `ply`, using the incremental accumulator when active,
    // otherwise the hand-crafted / full-refresh path. Assumes accStack[ply] is
    // valid (it is on entry to every node once the root is refreshed).
    Value eval_at(int ply) {
        if (nnueActive)
            return NNUE::evaluate(accStack[ply], pos);
        return Eval::evaluate(pos);
    }

    // Per-ply stack info threaded through the recursion. ssStore has one extra
    // leading element so ss[-1] (read for the countermove / null-move guard at
    // ply 0) addresses a valid, zero-initialised sentinel rather than UB.
    struct Stack {
        Move  currentMove = Move::none();
        Piece movedPiece  = NO_PIECE;
        int   staticEval  = 0;
    };
    Stack  ssStore[MAX_PLY + 5];
    Stack* ss = ssStore + 1;

    // Best result produced by this thread's iterative deepening.
    Move  bestMove    = Move::none();
    Move  ponderMove  = Move::none();   // 2nd PV move of the last completed depth
    Value bestValue   = VALUE_ZERO;
    int   completedDepth = 0;

    void clear_tables() {
        std::memset(killers, 0, sizeof(killers));
        std::memset(history, 0, sizeof(history));
        std::memset(captureHistory, 0, sizeof(captureHistory));
        std::memset(counterMoves, 0, sizeof(counterMoves));
    }

    // Rebuild pos from the stashed root FEN + game moves so the StateInfo chain
    // (and thus repetition / 50-move detection) matches the real game history.
    void setup_root() {
        rootStateCount = 0;
        pos.set(rootInfo.fen, &rootStates[rootStateCount++]);
        for (Move m : rootInfo.moves) {
            if (rootStateCount >= int(sizeof(rootStates) / sizeof(rootStates[0])))
                break;
            pos.do_move(m, rootStates[rootStateCount++]);
        }
    }

    void new_search() {
        nodes    = 0;
        seldepth = 0;
        stopped  = false;
        bestMove = Move::none();
        ponderMove = Move::none();
        bestValue = VALUE_ZERO;
        completedDepth = 0;
        nnueActive = UseNNUE && NNUE::loaded();
        std::memset(killers, 0, sizeof(killers));
        // Reset the sentinel (ss[-1]) plus every real ply slot.
        ss[-1].currentMove = Move::none();
        ss[-1].movedPiece  = NO_PIECE;
        ss[-1].staticEval  = VALUE_NONE;
        for (int i = 0; i < MAX_PLY + 4; ++i) {
            ss[i].currentMove = Move::none();
            ss[i].movedPiece  = NO_PIECE;
            ss[i].staticEval  = VALUE_NONE;
        }
    }

    // -- history helpers (per-thread) --------------------------------------
    void update_history(Color c, Move m, int bonus) {
        update_stat(history[c][m.from_sq()][m.to_sq()], bonus, 16384);
    }
    void update_capture_history(Piece pc, Move m, PieceType captured, int bonus) {
        update_stat(captureHistory[pc][m.to_sq()][captured], bonus, 16384);
    }

    // Only the main thread consults the wall clock / node limit; helpers stop
    // solely off the global stop flag (flipped by the main thread).
    void check_time() {
        if ((nodes & (CheckEveryNodes - 1)) != 0)
            return;
        if (stop.load(std::memory_order_relaxed)) {
            stopped = true;
            return;
        }
        if (!isMain())
            return;
        // While pondering the clock/node budget is not in force; only an explicit
        // global stop (handled above) or a ponderhit (clears pondering) ends it.
        if (pondering.load(std::memory_order_relaxed))
            return;
        if (Limits.nodes > 0 && total_nodes_estimate() >= Limits.nodes) {
            stopped = true;
            stop.store(true, std::memory_order_relaxed);
            return;
        }
        if (Time.maximum() < INT64_MAX / 4 && Time.elapsed() >= Time.maximum()) {
            stopped = true;
            stop.store(true, std::memory_order_relaxed);
        }
    }

    // Estimate total nodes for the node-limit check (see definition below).
    int64_t total_nodes_estimate() const;

    Value qsearch(Value alpha, Value beta, int ply);
    Value search(Value alpha, Value beta, int depth, int ply, bool cutNode);

    std::string pv_string() {
        std::string s;
        for (int i = 0; i < pvLength[0]; ++i) {
            if (i) s += ' ';
            s += move_to_uci(pvTable[0][i]);
        }
        return s;
    }
};

// All live worker threads for the current search (index 0 is the main thread).
// Used only for summing node counts; never mutated during the parallel section
// except each thread writing its own counters.
std::vector<std::unique_ptr<Thread>> Workers;

int64_t sum_nodes() {
    int64_t n = 0;
    for (const auto& t : Workers)
        n += t->nodes;
    return n;
}
int max_seldepth() {
    int d = 0;
    for (const auto& t : Workers)
        d = std::max(d, t->seldepth);
    return d;
}

int64_t Thread::total_nodes_estimate() const { return sum_nodes(); }

// ---------------------------------------------------------------------------
// Quiescence search: only captures + queen promotions (plus full evasions when
// in check). No stand-pat while in check.
// ---------------------------------------------------------------------------
Value Thread::qsearch(Value alpha, Value beta, int ply) {
    ++nodes;
    if (ply > seldepth)
        seldepth = ply;

    check_time();
    if (stopped)
        return VALUE_ZERO;

    if (pos.is_draw(ply))
        return VALUE_DRAW;

    if (ply >= MAX_PLY)
        return eval_at(ply);

    const bool inCheck = pos.checkers();
    Value bestValue;

    if (inCheck) {
        bestValue = -VALUE_INFINITE;  // must find an evasion
    } else {
        bestValue = eval_at(ply);
        if (bestValue >= beta)
            return bestValue;
        if (bestValue > alpha)
            alpha = bestValue;
    }

    // In check: search every legal evasion. Otherwise: captures + Q-promos only.
    if (inCheck) {
        MoveList<LEGAL> moves(pos);
        if (moves.size() == 0)
            return mated_in(ply);  // checkmate

        // Order evasions by MVV-LVA (captures first, then quiets).
        ExtMove ordered[MAX_MOVES];
        int n = 0;
        for (const ExtMove& em : moves) {
            ordered[n].move  = em.move;
            ordered[n].value = pos.capture(em.move) ? mvv_lva(pos, em.move) : -1;
            ++n;
        }
        std::stable_sort(ordered, ordered + n,
                         [](const ExtMove& a, const ExtMove& b) { return a.value > b.value; });

        for (int i = 0; i < n; ++i) {
            Move m = ordered[i].move;
            if (nnueActive)
                NNUE::update(accStack[ply + 1], accStack[ply], pos, m);
            pos.do_move(m, states[ply]);
            Value v = -qsearch(-beta, -alpha, ply + 1);
            pos.undo_move(m);
            if (stopped)
                return VALUE_ZERO;
            if (v > bestValue) {
                bestValue = v;
                if (v > alpha) {
                    alpha = v;
                    if (v >= beta)
                        break;
                }
            }
        }
        return bestValue;
    }

    // Not in check: captures + queen promotions.
    MoveList<CAPTURES> caps(pos);
    ExtMove ordered[MAX_MOVES];
    int n = 0;
    for (const ExtMove& em : caps) {
        if (!pos.legal(em.move))
            continue;
        ordered[n].move  = em.move;
        ordered[n].value = mvv_lva(pos, em.move);
        ++n;
    }
    std::stable_sort(ordered, ordered + n,
                     [](const ExtMove& a, const ExtMove& b) { return a.value > b.value; });

    for (int i = 0; i < n; ++i) {
        Move m = ordered[i].move;

        // Delta pruning: skip captures that cannot raise alpha even with a margin.
        PieceType victim = (m.type_of() == EN_PASSANT) ? PAWN
                                                       : type_of(pos.piece_on(m.to_sq()));
        Value gain = PieceValueOrder[victim];
        if (m.type_of() == PROMOTION)
            gain += QueenValue - PawnValue;
        if (bestValue + gain + 200 < alpha)
            continue;

        // SEE pruning: skip clearly losing captures in quiescence.
        if (!pos.see_ge(m, -100))
            continue;

        if (nnueActive)
            NNUE::update(accStack[ply + 1], accStack[ply], pos, m);
        pos.do_move(m, states[ply]);
        Value v = -qsearch(-beta, -alpha, ply + 1);
        pos.undo_move(m);
        if (stopped)
            return VALUE_ZERO;

        if (v > bestValue) {
            bestValue = v;
            if (v > alpha) {
                alpha = v;
                if (v >= beta)
                    break;
            }
        }
    }
    return bestValue;
}

// ---------------------------------------------------------------------------
// Negamax alpha-beta with a transposition table (PVS + modern pruning).
// ---------------------------------------------------------------------------
Value Thread::search(Value alpha, Value beta, int depth, int ply, bool cutNode) {
    const bool rootNode = (ply == 0);
    const bool pvNode   = (beta - alpha > 1);
    pvLength[ply] = ply;

    if (depth <= 0)
        return qsearch(alpha, beta, ply);

    ++nodes;
    if (ply > seldepth)
        seldepth = ply;

    check_time();
    if (stopped)
        return VALUE_ZERO;

    if (!rootNode) {
        if (pos.is_draw(ply))
            return VALUE_DRAW;
        if (ply >= MAX_PLY)
            return eval_at(ply);

        // Mate-distance pruning: bound the window by the best/worst possible mate.
        alpha = std::max(mated_in(ply), alpha);
        beta  = std::min(mate_in(ply + 1), beta);
        if (alpha >= beta)
            return alpha;
    }

    // TT probe.
    bool ttHit = false;
    TTEntry* tte = TT.probe(pos.key(), ttHit);
    Value ttValue = ttHit ? value_from_tt(tte->value(), ply) : VALUE_NONE;
    Move  ttMove  = ttHit ? tte->move() : Move::none();
    Bound ttBound = ttHit ? tte->bound() : BOUND_NONE;

    // TT cutoff (not in PV nodes to preserve an accurate PV / avoid instability).
    if (!pvNode && ttHit && tte->depth() >= depth && ttValue != VALUE_NONE) {
        if (ttBound == BOUND_EXACT ||
            (ttBound == BOUND_LOWER && ttValue >= beta) ||
            (ttBound == BOUND_UPPER && ttValue <= alpha))
            return ttValue;
    }

    const bool inCheck = pos.checkers();

    // Static evaluation (with TT-eval reuse). Not meaningful while in check.
    Value staticEval;
    Value eval;
    if (inCheck) {
        staticEval = eval = VALUE_NONE;
    } else if (ttHit && tte->eval() != VALUE_NONE) {
        staticEval = eval = tte->eval();
        // Use the TT value as a better positional estimate when the bound allows.
        if (ttValue != VALUE_NONE &&
            (ttBound & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttValue;
    } else {
        staticEval = eval = eval_at(ply);
    }

    ss[ply].staticEval = staticEval;

    // Are we improving relative to two plies ago? (used to soften pruning)
    const bool improving =
        !inCheck && ply >= 2 && ss[ply - 2].staticEval != VALUE_NONE &&
        staticEval > ss[ply - 2].staticEval;

    const Color us  = pos.side_to_move();
    const bool  nonPawnMat = pos.non_pawn_material(us) > 0;

    // -----------------------------------------------------------------------
    // Pre-move-loop pruning (only outside PV / not in check / not near mate).
    // -----------------------------------------------------------------------
    if (!pvNode && !inCheck && std::abs(beta) < VALUE_MATE_IN_MAX_PLY) {

        // Reverse futility / static null-move pruning.
        if (depth <= 8 &&
            eval - 80 * (depth - (improving ? 1 : 0)) >= beta &&
            eval < VALUE_MATE_IN_MAX_PLY)
            return eval;

        // Null-move pruning: give the opponent a free move; if we still fail
        // high, prune. Guard against zugzwang via non-pawn material.
        if (depth >= 3 && eval >= beta && nonPawnMat &&
            ss[ply - 1].currentMove != Move::null()) {
            int R = 3 + depth / 3 + std::min((eval - beta) / 200, 3);
            ss[ply].currentMove = Move::null();
            ss[ply].movedPiece  = NO_PIECE;

            // Null move: board unchanged, so the child accumulator equals the
            // parent's (feature indices depend only on king squares + pieces,
            // not on side-to-move). Copy it forward.
            if (nnueActive)
                accStack[ply + 1] = accStack[ply];

            pos.do_null_move(states[ply]);
            Value nullValue =
                -search(-beta, -beta + 1, depth - R, ply + 1, !cutNode);
            pos.undo_null_move();

            if (stopped)
                return VALUE_ZERO;

            if (nullValue >= beta) {
                // Do not return unproven mate scores.
                if (nullValue >= VALUE_MATE_IN_MAX_PLY)
                    nullValue = beta;

                if (depth < 10)
                    return nullValue;

                // Verification search at high depth to guard against zugzwang.
                Value v = search(beta - 1, beta, depth - R, ply, false);
                if (v >= beta)
                    return nullValue;
            }
        }
    }

    // Generate and order moves.
    MoveList<LEGAL> moves(pos);
    if (moves.size() == 0)
        return inCheck ? mated_in(ply) : VALUE_DRAW;  // mate or stalemate

    // Countermove from the previous move.
    Move counter = Move::none();
    if (ss[ply - 1].currentMove.is_ok())
        counter = counterMoves[ss[ply - 1].movedPiece]
                              [ss[ply - 1].currentMove.to_sq()];

    // Score moves for ordering.
    ExtMove ordered[MAX_MOVES];
    int n = 0;
    for (const ExtMove& em : moves) {
        Move m = em.move;
        int score;
        const bool isCap = pos.capture(m) || m.type_of() == PROMOTION;
        if (m == ttMove) {
            score = 1 << 30;
        } else if (isCap) {
            // Winning/equal captures rank above quiets; losing captures below.
            int capScore = mvv_lva(pos, m);
            PieceType captured = (m.type_of() == EN_PASSANT)
                                     ? PAWN
                                     : type_of(pos.piece_on(m.to_sq()));
            capScore += captureHistory[pos.moved_piece(m)][m.to_sq()][captured] / 16;
            if (pos.see_ge(m, -50))
                score = (1 << 28) + capScore;   // good capture
            else
                score = (1 << 18) + capScore;   // bad capture (below quiets)
        } else if (m == killers[ply][0]) {
            score = (1 << 27);
        } else if (m == killers[ply][1]) {
            score = (1 << 27) - 1;
        } else if (m == counter) {
            score = (1 << 26);
        } else {
            score = (1 << 20) + history[us][m.from_sq()][m.to_sq()];
        }
        ordered[n].move  = m;
        ordered[n].value = score;
        ++n;
    }
    std::stable_sort(ordered, ordered + n,
                     [](const ExtMove& a, const ExtMove& b) { return a.value > b.value; });

    // searchmoves restriction at the root.
    const bool restrictRoot = rootNode && !Limits.searchmoves.empty();

    Value bestValue = -VALUE_INFINITE;
    Move  bestMove  = Move::none();
    Bound bound     = BOUND_UPPER;
    int   moveCount = 0;

    // Quiet moves tried (for history malus on the ones that failed).
    Move  quietsTried[64];
    int   quietCount = 0;

    const bool ttMoveIsCapture =
        ttMove != Move::none() &&
        (pos.capture(ttMove) || ttMove.type_of() == PROMOTION);

    for (int i = 0; i < n; ++i) {
        Move m = ordered[i].move;

        if (restrictRoot &&
            std::find(Limits.searchmoves.begin(), Limits.searchmoves.end(), m) ==
                Limits.searchmoves.end())
            continue;

        ++moveCount;
        const bool isCapture = pos.capture(m) || m.type_of() == PROMOTION;
        const bool givesCheck = pos.gives_check(m);
        const bool isQuiet = !isCapture;

        // -------------------------------------------------------------------
        // Move-loop pruning for quiet moves (skip late/hopeless moves).
        // -------------------------------------------------------------------
        if (!rootNode && !inCheck && isQuiet &&
            bestValue > VALUE_MATED_IN_MAX_PLY) {

            // Late Move Pruning: after enough quiets at low depth, skip the rest.
            int lmpThreshold = (3 + depth * depth) / (improving ? 1 : 2);
            if (!pvNode && depth <= 8 && moveCount >= lmpThreshold && !givesCheck) {
                continue;
            }

            // Futility pruning: quiet move that cannot raise alpha at low depth.
            if (!pvNode && depth <= 6 && !givesCheck &&
                staticEval + 120 + 90 * depth <= alpha) {
                continue;
            }

            // SEE pruning for quiets that lose material.
            if (depth <= 6 && !givesCheck && !pos.see_ge(m, -25 * depth * depth)) {
                continue;
            }
        }

        if (isQuiet && quietCount < 64)
            quietsTried[quietCount++] = m;

        ss[ply].currentMove = m;
        ss[ply].movedPiece  = pos.moved_piece(m);

        // Check extension.
        int extension = (givesCheck) ? 1 : 0;
        int newDepth  = depth - 1 + extension;

        // Derive the child accumulator from the parent + this move (on the
        // pre-move board), then make the move.
        if (nnueActive)
            NNUE::update(accStack[ply + 1], accStack[ply], pos, m);
        pos.do_move(m, states[ply], givesCheck);

        Value v = -VALUE_INFINITE;
        bool doFullSearch;

        // -------------------------------------------------------------------
        // Late Move Reductions: reduce late quiet moves, re-search on fail-high.
        // -------------------------------------------------------------------
        if (depth >= 3 && moveCount > 1 + (rootNode ? 1 : 0) &&
            (isQuiet || !pos.see_ge(m, VALUE_ZERO))) {
            int r = Reductions[std::min(depth, MAX_PLY)][std::min(moveCount, MAX_MOVES - 1)];

            // Reduce more for cut nodes and non-PV; less for killers / good history.
            if (cutNode)          r += 1;
            if (!pvNode)          r += 1;
            if (!improving)       r += 1;
            if (ttMoveIsCapture)  r += 1;

            if (isQuiet) {
                int h = history[us][m.from_sq()][m.to_sq()];
                r -= h / 6000;
            }

            // Search diversity for Lazy SMP: helper threads perturb the reduction
            // slightly (by thread id) so they explore different subtrees than the
            // main thread and each other, which improves TT sharing. The main
            // thread (id 0) is unchanged.
            if (id > 0 && isQuiet && ((moveCount + id) & 3) == 0)
                r += 1;

            r = std::max(0, std::min(r, newDepth - 1));

            int reducedDepth = newDepth - r;

            // Reduced, null-window search.
            v = -search(-alpha - 1, -alpha, reducedDepth, ply + 1, true);

            // If the reduced search beat alpha, re-search at full depth.
            doFullSearch = (v > alpha && r > 0);
        } else {
            doFullSearch = !pvNode || moveCount > 1;
        }

        if (doFullSearch) {
            // Full-depth null-window search (PVS for non-first moves).
            v = -search(-alpha - 1, -alpha, newDepth, ply + 1, !cutNode);
        }

        // Principal Variation Search: first move, or a fail-high inside the
        // window at a PV node, gets a full-window (re-)search.
        if (pvNode && (moveCount == 1 || (v > alpha && v < beta))) {
            v = -search(-beta, -alpha, newDepth, ply + 1, false);
        }

        pos.undo_move(m);

        if (stopped)
            return VALUE_ZERO;

        if (v > bestValue) {
            bestValue = v;
            bestMove  = m;

            if (v > alpha) {
                alpha = v;
                bound = BOUND_EXACT;

                // Update triangular PV.
                pvTable[ply][ply] = m;
                for (int k = ply + 1; k < pvLength[ply + 1]; ++k)
                    pvTable[ply][k] = pvTable[ply + 1][k];
                pvLength[ply] = pvLength[ply + 1];

                if (v >= beta) {
                    bound = BOUND_LOWER;

                    // Update killers / history / countermove on a beta cutoff.
                    if (!isCapture) {
                        if (killers[ply][0] != m) {
                            killers[ply][1] = killers[ply][0];
                            killers[ply][0] = m;
                        }
                        int bonus = history_bonus(depth);
                        update_history(us, m, bonus);

                        // Countermove.
                        if (ss[ply - 1].currentMove.is_ok())
                            counterMoves[ss[ply - 1].movedPiece]
                                        [ss[ply - 1].currentMove.to_sq()] = m;

                        // History malus for the other quiets we tried.
                        for (int q = 0; q < quietCount; ++q) {
                            Move qm = quietsTried[q];
                            if (qm != m)
                                update_history(us, qm, -bonus);
                        }
                    } else {
                        PieceType captured =
                            (m.type_of() == EN_PASSANT)
                                ? PAWN
                                : type_of(pos.piece_on(m.to_sq()));
                        update_capture_history(pos.moved_piece(m), m, captured,
                                               history_bonus(depth));
                    }
                    break;
                }
            }
        }
    }

    // If restrictRoot filtered everything (empty), fall back to a normal result.
    if (moveCount == 0)
        return inCheck ? mated_in(ply) : VALUE_DRAW;

    // TT store.
    if (!stopped)
        tte->save(pos.key(), value_to_tt(bestValue, ply),
                  pvNode, bound, depth, bestMove,
                  staticEval, TT.generation());

    return bestValue;
}

// ---------------------------------------------------------------------------
// Iterative deepening loop, run by every worker thread. The main thread
// (id == 0) additionally prints "info" lines and owns time management; helper
// threads run silently and diversify via a per-thread starting-depth stagger.
// ---------------------------------------------------------------------------
void iterative_deepening(Thread& th) {
    const int maxDepth = Limits.depth > 0 ? std::min(Limits.depth, MAX_PLY - 1)
                                          : MAX_PLY - 1;

    // Seed the root accumulator once (the root position is fixed for the whole
    // search); every node then derives its accumulator incrementally.
    if (th.nnueActive)
        NNUE::refresh(th.accStack[0], th.pos);

    Move  bestMove  = Move::none();
    Value bestValue = VALUE_ZERO;

    // Diversify helper threads: stagger the first depth so they don't all march
    // in lockstep on identical iterations (odd helpers skip depth 1).
    int startDepth = 1;
    if (th.id > 0)
        startDepth = 1 + (th.id % 2);

    for (int depth = startDepth; depth <= maxDepth; ++depth) {
        // -------------------------------------------------------------------
        // Aspiration windows (from depth >= 4): search around the last score
        // with a small window; widen exponentially on fail-low/high.
        // -------------------------------------------------------------------
        Value alpha = -VALUE_INFINITE;
        Value beta  = VALUE_INFINITE;
        Value delta = VALUE_INFINITE;
        Value v;

        if (depth >= 4 && std::abs(bestValue) < VALUE_MATE_IN_MAX_PLY) {
            // Helper threads use a slightly wider aspiration window for diversity.
            delta = Value(18 + (th.id % 4) * 4);
            alpha = std::max<Value>(bestValue - delta, -VALUE_INFINITE);
            beta  = std::min<Value>(bestValue + delta,  VALUE_INFINITE);
        }

        while (true) {
            v = th.search(alpha, beta, depth, 0, false);

            if (th.stopped)
                break;

            if (v <= alpha) {
                beta  = (alpha + beta) / 2;
                alpha = std::max<Value>(v - delta, -VALUE_INFINITE);
            } else if (v >= beta) {
                beta = std::min<Value>(v + delta, VALUE_INFINITE);
            } else {
                break;  // score inside the window
            }

            delta += delta / 2 + 5;  // widen exponentially
            if (delta > 2000) {      // give up on aspiration, go full-width
                alpha = -VALUE_INFINITE;
                beta  = VALUE_INFINITE;
            }
        }

        if (th.stopped) {
            // Depth incomplete: keep the previous iteration's best move if we
            // have one; otherwise take whatever the PV produced.
            if (bestMove == Move::none() && th.pvLength[0] > 0)
                bestMove = th.pvTable[0][0];
            break;
        }

        bestValue = v;
        if (th.pvLength[0] > 0)
            bestMove = th.pvTable[0][0];

        th.bestMove       = bestMove;
        th.bestValue      = bestValue;
        th.completedDepth = depth;
        // Capture the ponder move (2nd PV move) from THIS completed iteration, so
        // it stays consistent with bestMove even if a later iteration is aborted
        // mid-flight (which can leave pvTable[0] partly overwritten).
        th.ponderMove =
            (th.pvLength[0] >= 2 && th.pvTable[0][0] == bestMove)
                ? th.pvTable[0][1]
                : Move::none();

        // Only the main thread reports and manages the clock.
        if (th.isMain()) {
            int64_t elapsed = std::max<int64_t>(1, Time.elapsed());
            int64_t totalNodes = sum_nodes();
            int64_t nps     = totalNodes * 1000 / elapsed;

            std::cout << "info depth " << depth
                      << " seldepth " << max_seldepth()
                      << " score " << UCI::value(v)
                      << " nodes " << totalNodes
                      << " nps " << nps
                      << " time " << elapsed
                      << " pv " << th.pv_string()
                      << std::endl;

            // While pondering the clock is not running and we must not emit a
            // bestmove: never terminate on time / mate / node budget. If the
            // position is already solved (a mate found, or we would run out of
            // depth), idle here until the GUI sends ponderhit (clears pondering)
            // or stop (sets the global stop flag) rather than falling out of the
            // loop and printing bestmove prematurely.
            if (pondering.load(std::memory_order_relaxed)) {
                bool solved = std::abs(v) >= VALUE_MATE_IN_MAX_PLY ||
                              (Limits.nodes > 0 && totalNodes >= Limits.nodes) ||
                              depth >= maxDepth;
                if (solved) {
                    while (pondering.load(std::memory_order_relaxed) &&
                           !stop.load(std::memory_order_relaxed))
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                // If ponderhit arrived we fall through to keep deepening under the
                // now-live clock; if stop arrived the next iteration's search will
                // observe it and return. Either way, do not stop here.
                continue;
            }

            // Stop if we found a forced mate or exhausted the optimum time budget.
            if (Time.maximum() < INT64_MAX / 4 && Time.elapsed() >= Time.optimum()) {
                stop.store(true, std::memory_order_relaxed);
                break;
            }
            if (std::abs(v) >= VALUE_MATE_IN_MAX_PLY) {
                stop.store(true, std::memory_order_relaxed);
                break;
            }
            if (Limits.nodes > 0 && totalNodes >= Limits.nodes) {
                stop.store(true, std::memory_order_relaxed);
                break;
            }
        } else {
            // Helper threads still honour a mate find / global stop. Do not break
            // on a mate find while pondering (that would let a helper exit and,
            // once the pool joins on stop, is harmless — but keep them alive so
            // they keep sharing TT until pondering ends or stop is set).
            if (!pondering.load(std::memory_order_relaxed) &&
                std::abs(v) >= VALUE_MATE_IN_MAX_PLY)
                break;
            if (stop.load(std::memory_order_relaxed))
                break;
        }
    }

    th.bestMove  = bestMove;
    th.bestValue = bestValue;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public Search interface
// ---------------------------------------------------------------------------
namespace Search {

void init() {
    init_reductions();
    clear();
}

void clear() {
    TT.clear();
    // Heuristic tables now live per-thread and are reset at the start of every
    // search; nothing global to clear here beyond the shared TT.
}

// Transition from a "go ponder" search to a normal timed search. The GUI calls
// this (via the UCI "ponderhit" command) when the opponent actually played the
// move we were pondering on. We do NOT relaunch the search — the same worker
// thread(s) keep running — we simply:
//   1. clear the pondering flag (so check_time / the ID loop honour the clock),
//   2. reset the clock origin to NOW and recompute the budget from the wtime/
//      btime/winc/binc the "go ponder" carried (so time is measured from here).
// If no ponder search is in flight this is a harmless no-op.
void ponderhit() {
    if (!pondering.load(std::memory_order_relaxed))
        return;
    pondering.store(false, std::memory_order_relaxed);
    // Recompute optimum/maximum with pondering now false → a real timed budget,
    // measured from this instant.
    Time.restart();
}

Move think(Position& pos, const SearchLimits& limits) {
    Limits    = limits;
    rootColor = pos.side_to_move();

    // If the UCI layer did not stash a root description (e.g. a direct call),
    // fall back to the current position's FEN with no prior move history.
    if (rootInfo.fen.empty())
        rootInfo.fen = pos.fen();

    stop.store(false, std::memory_order_relaxed);
    // Establish the pondering state from the launch command BEFORE Time.init(),
    // which consults it to decide between an infinite (ponder) and a real budget.
    pondering.store(limits.ponder, std::memory_order_relaxed);
    stats.nodes    = 0;
    stats.seldepth = 0;

    Time.init(limits, rootColor, pos.game_ply());
    TT.new_search();

    const int numThreads = std::max(1, Threads);

    // Build one Thread per worker; each rebuilds its own Position from the root.
    Workers.clear();
    Workers.reserve(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        auto th = std::make_unique<Thread>();
        th->id = i;
        th->clear_tables();
        th->new_search();
        th->setup_root();
        Workers.push_back(std::move(th));
    }

    // Launch helper threads (1..N-1); the main thread runs worker 0 inline so it
    // can print / manage time on the calling (search) thread.
    std::vector<std::thread> pool;
    pool.reserve(numThreads > 0 ? numThreads - 1 : 0);
    for (int i = 1; i < numThreads; ++i)
        pool.emplace_back([i]() { iterative_deepening(*Workers[i]); });

    iterative_deepening(*Workers[0]);

    // Main thread finished (time/limit/mate): signal helpers and join cleanly.
    stop.store(true, std::memory_order_relaxed);
    for (auto& t : pool)
        if (t.joinable())
            t.join();

    // Aggregate reporting stats.
    stats.nodes    = sum_nodes();
    stats.seldepth = max_seldepth();

    // Pick the best result across threads: prefer the greater completed depth,
    // then the higher score. This lets a helper that finished deeper override
    // the main thread's move.
    Thread* best = Workers[0].get();
    for (int i = 1; i < numThreads; ++i) {
        Thread* t = Workers[i].get();
        if (t->bestMove == Move::none())
            continue;
        if (best->bestMove == Move::none() ||
            t->completedDepth > best->completedDepth ||
            (t->completedDepth == best->completedDepth &&
             t->bestValue > best->bestValue))
            best = t;
    }

    Move bestMove = best->bestMove;

    // Ponder move: the 2nd move of the chosen line's principal variation — the
    // reply we expect, so the GUI can ponder on it next. Taken from the last
    // COMPLETED iteration (captured in th.ponderMove) so it stays consistent with
    // bestMove even when the final iteration was aborted mid-flight.
    Move ponderMove = (best->bestMove != Move::none()) ? best->ponderMove
                                                       : Move::none();

    if (bestMove == Move::none()) {
        // Fallback: pick any legal move so we never emit an illegal bestmove.
        MoveList<LEGAL> moves(pos);
        if (moves.size() > 0)
            bestMove = moves.begin()->move;
        ponderMove = Move::none();  // no reliable PV behind a fallback move
    }

    std::cout << "bestmove " << move_to_uci(bestMove);
    if (ponderMove != Move::none())
        std::cout << " ponder " << move_to_uci(ponderMove);
    std::cout << std::endl;

    // Clear the pondering flag now the search is fully done, so it never leaks
    // into a subsequent non-ponder "go" (defensive; the UCI layer also resets it).
    pondering.store(false, std::memory_order_relaxed);

    // Free per-thread state (large tables) once the move is chosen.
    Workers.clear();
    return bestMove;
}

}  // namespace Search

}  // namespace engine

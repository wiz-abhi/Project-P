// uci.cpp — UCI command loop + perft self-test driver.
//
//   engine perft                 run the built-in perft suite (PASS/FAIL)
//   engine perft "<fen>" <depth> perft divide for one position
//   (no perft arg)               enter the UCI stdin command loop

#include "uci.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "attacks.hpp"
#include "bitboard.hpp"
#include "book.hpp"
#include "eval.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "position.hpp"
#include "search.hpp"
#include "tt.hpp"
#include "types.hpp"

namespace engine {

// Move Overhead (ms), consumed by TimeManager. Defined here, declared extern in
// search.cpp.
int MoveOverheadMs = 30;

// NNUE toggle (defined in eval.cpp) and the current EvalFile path.
extern bool UseNNUE;
namespace { std::string EvalFilePath = "nets/nn-halfkp.nnue"; }

namespace Search {
// Defined in search.cpp. Declared here (rather than in the frozen search.hpp) so
// the UCI layer can configure the number of Lazy-SMP worker threads and hand the
// search the root position description for per-thread reconstruction.
void set_threads(int n);
void set_root(const std::string& fen, const std::vector<Move>& moves);
}  // namespace Search

namespace {

constexpr char EngineName[]   = "ClientEngine";
constexpr char EngineAuthor[] = "Client";
constexpr char StartFEN[]     =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ---------------------------------------------------------------------------
// Perft driver (preserved verbatim in behavior from the original main.cpp)
// ---------------------------------------------------------------------------
uint64_t perft(Position& pos, int depth) {
    if (depth == 0)
        return 1;

    MoveList<LEGAL> moves(pos);
    if (depth == 1)
        return moves.size();

    uint64_t nodes = 0;
    StateInfo st;
    for (const ExtMove& em : moves) {
        pos.do_move(em.move, st);
        nodes += perft(pos, depth - 1);
        pos.undo_move(em.move);
    }
    return nodes;
}

uint64_t divide(Position& pos, int depth) {
    uint64_t total = 0;
    StateInfo st;
    for (const ExtMove& em : MoveList<LEGAL>(pos)) {
        pos.do_move(em.move, st);
        uint64_t n = depth <= 1 ? 1 : perft(pos, depth - 1);
        pos.undo_move(em.move);
        total += n;
        std::cout << move_to_uci(em.move) << ": " << n << '\n';
    }
    std::cout << "\nNodes searched: " << total << '\n';
    return total;
}

struct Case {
    const char*           fen;
    const char*           name;
    std::vector<uint64_t> expected;  // index 0 -> depth 1, etc.
};

const std::vector<Case> Suite = {
    {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     "startpos",
     {20, 400, 8902, 197281, 4865609}},
    {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     "kiwipete",
     {48, 2039, 97862, 4085603}},
    {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     "position 3",
     {14, 191, 2812, 43238, 674624}},
    {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     "position 4",
     {6, 264, 9467, 422333}},
    {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     "position 5",
     {44, 1486, 62379, 2103487}},
    {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     "position 6",
     {46, 2079, 89890, 3894594}},
};

int run_suite() {
    bool     allPass    = true;
    uint64_t grandNodes = 0;
    double   grandSecs  = 0.0;

    for (const Case& c : Suite) {
        std::cout << "\n=== " << c.name << " ===\n" << c.fen << '\n';
        StateInfo si;
        Position  pos;
        pos.set(c.fen, &si);

        for (size_t i = 0; i < c.expected.size(); ++i) {
            int      depth = int(i) + 1;
            uint64_t exp   = c.expected[i];

            auto     t0  = std::chrono::steady_clock::now();
            uint64_t got = perft(pos, depth);
            auto     t1  = std::chrono::steady_clock::now();

            double secs = std::chrono::duration<double>(t1 - t0).count();
            grandNodes += got;
            grandSecs += secs;

            bool ok = (got == exp);
            allPass = allPass && ok;

            std::cout << "  depth " << depth << "  nodes " << std::setw(10)
                      << got << "  expected " << std::setw(10) << exp << "  "
                      << (ok ? "PASS" : "FAIL");
            if (!ok)
                std::cout << "  (diff " << (int64_t(got) - int64_t(exp)) << ")";
            std::cout << '\n';
        }
    }

    double nps = grandSecs > 0 ? grandNodes / grandSecs : 0;
    std::cout << "\n================ SUMMARY ================\n";
    std::cout << (allPass ? "ALL POSITIONS PASS" : "SOME POSITIONS FAILED")
              << '\n';
    std::cout << "total nodes: " << grandNodes << "   time: " << std::fixed
              << std::setprecision(3) << grandSecs << "s   ~" << uint64_t(nps)
              << " nodes/sec\n";

    return allPass ? 0 : 1;
}

void perft_usage() {
    std::cout
        << "usage:\n"
        << "  engine perft                    run the perft self-test suite\n"
        << "  engine perft \"<fen>\" <depth>    perft divide for one position\n";
}

int perft_command(int argc, char** argv) {
    if (argc == 2)
        return run_suite();

    if (argc >= 4) {
        std::string fen   = argv[2];
        int         depth = std::atoi(argv[3]);
        StateInfo   si;
        Position    pos;
        pos.set(fen, &si);
        std::cout << "perft divide  depth " << depth << "\n" << fen << "\n\n";
        divide(pos, depth);
        return 0;
    }

    perft_usage();
    return 0;
}

// ---------------------------------------------------------------------------
// UCI loop state
// ---------------------------------------------------------------------------
Position               rootPos;
std::deque<StateInfo>  rootStates;
std::thread            searchThread;

// Polyglot opening book. Enabled via the "OwnBook" option; the file is loaded
// via the "BookFile" option. When enabled and the current root position is in
// the book, "go" plays a book move directly and skips the search.
//
// Accessed via book() (a function-local static) so it is constructed on first
// use rather than at static-init time, and never runs a static destructor at
// process exit.
Book& book() {
    static Book* b = new Book();
    return *b;
}
bool                   ownBook     = false;
std::string            bookFilePath = "book/openings.bin";

// Root position description for the search: the FEN of the "position" base plus
// the game moves applied on top. Each Lazy-SMP worker rebuilds its own Position
// from this so repetition / 50-move history is intact per-thread.
std::string            rootFEN;
std::vector<Move>      rootMoves;

void join_search() {
    if (searchThread.joinable())
        searchThread.join();
}

// position [startpos | fen <fen>] [moves <m1> ...]
void set_position(std::istringstream& is) {
    std::string token, fen;
    is >> token;

    if (token == "startpos") {
        fen = StartFEN;
        is >> token;  // consume potential "moves"
    } else if (token == "fen") {
        while (is >> token && token != "moves")
            fen += token + " ";
    } else {
        return;
    }

    rootStates.clear();
    rootStates.emplace_back();
    rootPos.set(fen, &rootStates.back());

    // Record the base FEN (after Position::set has normalised it) and the game
    // moves so the search can rebuild a per-thread Position with intact history.
    rootFEN = rootPos.fen();
    rootMoves.clear();

    // Apply the move list (token is "moves" or already the first move).
    if (token == "moves") {
        while (is >> token) {
            Move m = UCI::to_move(rootPos, token);
            if (m == Move::none())
                break;
            rootMoves.push_back(m);
            rootStates.emplace_back();
            rootPos.do_move(m, rootStates.back());
        }
    }
}

// go ...
void go(std::istringstream& is) {
    join_search();

    SearchLimits limits;
    std::string  token;

    while (is >> token) {
        if (token == "searchmoves") {
            while (is >> token) {
                Move m = UCI::to_move(rootPos, token);
                if (m != Move::none())
                    limits.searchmoves.push_back(m);
            }
        } else if (token == "ponder")    limits.ponder = true;
        else if (token == "wtime")       is >> limits.time[WHITE];
        else if (token == "btime")       is >> limits.time[BLACK];
        else if (token == "winc")        is >> limits.inc[WHITE];
        else if (token == "binc")        is >> limits.inc[BLACK];
        else if (token == "movestogo")   is >> limits.movestogo;
        else if (token == "depth")       is >> limits.depth;
        else if (token == "nodes")       is >> limits.nodes;
        else if (token == "movetime")    is >> limits.movetime;
        else if (token == "infinite")    limits.infinite = true;
    }

    // Opening book: if enabled and the current position is in the book, play a
    // book move immediately and skip the search. Never short-circuit a "go
    // ponder": pondering must actually search (on the opponent's clock) and must
    // not emit a bestmove until ponderhit/stop.
    if (!limits.ponder && ownBook && book().loaded()) {
        Move bm = book().probe(rootPos, /*pickBest=*/false);
        if (bm != Move::none()) {
            std::string uci = move_to_uci(bm);
            std::cout << "info depth 0 score cp 0 pv " << uci << "\n"
                      << "bestmove " << uci << std::endl;
            return;
        }
    }

    Search::stop = false;
    Search::set_root(rootFEN, rootMoves);
    searchThread = std::thread([limits]() { Search::think(rootPos, limits); });
}

void setoption(std::istringstream& is) {
    std::string token, name, value;
    is >> token;  // "name"

    while (is >> token && token != "value")
        name += (name.empty() ? "" : " ") + token;
    while (is >> token)
        value += (value.empty() ? "" : " ") + token;

    if (name == "Hash") {
        TT.resize(size_t(std::atoi(value.c_str())));
    } else if (name == "Move Overhead") {
        MoveOverheadMs = std::atoi(value.c_str());
    } else if (name == "Threads") {
        int n = std::atoi(value.c_str());
        if (n < 1)   n = 1;
        if (n > 256) n = 256;
        Search::set_threads(n);
    } else if (name == "EvalFile") {
        EvalFilePath = value;
        if (NNUE::load(EvalFilePath))
            std::cout << "info string EvalFile loaded: " << EvalFilePath << "\n";
        else
            std::cout << "info string EvalFile load FAILED: " << EvalFilePath << "\n";
    } else if (name == "Use NNUE") {
        UseNNUE = (value == "true" || value == "1" || value == "True");
    } else if (name == "OwnBook") {
        ownBook = (value == "true" || value == "1" || value == "True");
    } else if (name == "BookFile") {
        bookFilePath = value;
        if (book().load(bookFilePath))
            std::cout << "info string BookFile loaded: " << bookFilePath
                      << "\n";
        else
            std::cout << "info string BookFile load FAILED: " << bookFilePath
                      << "\n";
    }
}

void uci_loop_stdin() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string        token;
        is >> token;

        if (token == "uci") {
            std::cout << "id name " << EngineName << "\n"
                      << "id author " << EngineAuthor << "\n"
                      << "option name Hash type spin default 16 min 1 max 65536\n"
                      << "option name Threads type spin default 1 min 1 max 256\n"
                      << "option name Move Overhead type spin default 30 min 0 max 5000\n"
                      << "option name EvalFile type string default nets/nn-halfkp.nnue\n"
                      << "option name Use NNUE type check default true\n"
                      << "option name OwnBook type check default false\n"
                      << "option name BookFile type string default book/openings.bin\n"
                      << "uciok" << std::endl;
        } else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (token == "setoption") {
            setoption(is);
        } else if (token == "ucinewgame") {
            join_search();
            Search::clear();
        } else if (token == "position") {
            join_search();
            set_position(is);
        } else if (token == "go") {
            go(is);
        } else if (token == "ponderhit") {
            // Opponent played the move we were pondering: hand the running search
            // its live clock. Do NOT join/relaunch — the same thread keeps going
            // and prints bestmove when the (now-started) time budget is spent.
            Search::ponderhit();
        } else if (token == "stop") {
            Search::stop = true;
            join_search();
        } else if (token == "eval") {
            // Debug: print raw NNUE value for the current root position.
            if (NNUE::loaded())
                std::cout << "nnue " << NNUE::evaluate(rootPos) << std::endl;
            else
                std::cout << "nnue (not loaded)" << std::endl;
        } else if (token == "d" || token == "print") {
            std::cout << rootPos.to_string() << std::endl;
        } else if (token == "quit" || token == "exit") {
            Search::stop = true;
            join_search();
            break;
        }
    }
    // EOF on stdin: let any in-progress search finish naturally rather than
    // truncating it (matches a batch-piped session). "quit" above force-stops.
    join_search();
}

}  // namespace

// ---------------------------------------------------------------------------
// Public UCI interface
// ---------------------------------------------------------------------------
namespace UCI {

void loop(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "perft") {
        std::exit(perft_command(argc, argv));
    }

    // Flush stdout on every insertion so each UCI response (notably
    // "bestmove ...") reaches the GUI immediately instead of sitting in a
    // block buffer — important when stdout is a pipe.
    std::cout << std::unitbuf;

    // Default: initialize a start position then enter the stdin loop.
    rootStates.clear();
    rootStates.emplace_back();
    rootPos.set(StartFEN, &rootStates.back());
    uci_loop_stdin();
}

Move to_move(const Position& pos, const std::string& str) {
    for (const ExtMove& em : MoveList<LEGAL>(pos))
        if (str == move_to_uci(em.move))
            return em.move;
    return Move::none();
}

std::string value(Value v) {
    std::ostringstream ss;
    if (std::abs(v) >= VALUE_MATE_IN_MAX_PLY && std::abs(v) <= VALUE_MATE) {
        int n = (v > 0) ? (VALUE_MATE - v + 1) / 2 : -(VALUE_MATE + v) / 2;
        ss << "mate " << n;
    } else {
        ss << "cp " << v;
    }
    return ss.str();
}

}  // namespace UCI

}  // namespace engine

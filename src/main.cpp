// main.cpp — engine entry point.
//
// Initializes the engine's global tables, then hands control to the UCI loop,
// which also dispatches the "perft" self-test subcommand.

#include "attacks.hpp"
#include "bitboard.hpp"
#include "eval.hpp"
#include "position.hpp"
#include "search.hpp"
#include "tt.hpp"
#include "uci.hpp"

using namespace engine;

int main(int argc, char** argv) {
    Attacks::init();
    Bitboards::init();
    Position::init();
    Eval::init();
    Search::init();
    TT.resize(16);

    UCI::loop(argc, argv);
    return 0;
}

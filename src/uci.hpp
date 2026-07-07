// uci.hpp — Universal Chess Interface command loop.
//
// Parses UCI commands from stdin and drives the engine. Also handles auxiliary
// subcommands ("perft", "bench") when passed as argv or typed.
#pragma once

#include <string>

#include "types.hpp"

namespace engine {

class Position;

namespace UCI {

void loop(int argc, char** argv);          // main read-eval loop
Move to_move(const Position& pos, const std::string& str);   // parse a UCI move string
std::string value(Value v);                 // format a score as UCI "cp"/"mate" token

}  // namespace UCI

}  // namespace engine

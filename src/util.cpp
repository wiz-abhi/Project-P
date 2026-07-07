// util.cpp — Free string-formatting helpers declared at the bottom of types.hpp.

#include <string>

#include "types.hpp"

namespace engine {

std::string square_to_string(Square s) {
    if (s == SQ_NONE || !is_ok(s))
        return "-";
    std::string str;
    str += char('a' + file_of(s));
    str += char('1' + rank_of(s));
    return str;
}

std::string move_to_uci(Move m) {
    if (m == Move::none())
        return "(none)";
    if (m == Move::null())
        return "0000";

    Square from = m.from_sq();
    Square to   = m.to_sq();

    std::string str = square_to_string(from) + square_to_string(to);

    if (m.type_of() == PROMOTION) {
        const char pieceLetters[] = " pnbrqk";  // index by PieceType
        str += pieceLetters[m.promotion_type()];
    }

    return str;
}

}  // namespace engine

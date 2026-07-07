// types.hpp — Core type definitions shared across the entire engine.
//
// This header is the fixed "contract": every other module builds against the
// types, enums and helpers declared here. Keep it dependency-free and stable.
#pragma once

#include <cstdint>
#include <string>

namespace engine {

// ---------------------------------------------------------------------------
// Basic scalar aliases
// ---------------------------------------------------------------------------
using Bitboard = uint64_t;   // one bit per square (a1 = bit 0 ... h8 = bit 63)
using Key      = uint64_t;   // Zobrist hash key
using Value    = int32_t;    // evaluation / search score in centipawns

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------
enum Color : int { WHITE, BLACK, COLOR_NB = 2 };

constexpr Color operator~(Color c) { return Color(c ^ BLACK); }  // flip color

// ---------------------------------------------------------------------------
// Piece types and pieces
// ---------------------------------------------------------------------------
enum PieceType : int {
    NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    ALL_PIECES = 0,
    PIECE_TYPE_NB = 8
};

enum Piece : int {
    NO_PIECE,
    W_PAWN = PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = PAWN + 8, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    PIECE_NB = 16
};

constexpr Piece make_piece(Color c, PieceType pt) { return Piece((c << 3) + pt); }
constexpr PieceType type_of(Piece pc) { return PieceType(pc & 7); }
constexpr Color color_of(Piece pc) { return Color(pc >> 3); }
constexpr bool is_ok(Piece pc) { return pc != NO_PIECE; }

// ---------------------------------------------------------------------------
// Squares, files, ranks
// ---------------------------------------------------------------------------
enum Square : int {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE,
    SQUARE_NB = 64
};

enum File : int { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NB };
enum Rank : int { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NB };

constexpr Square make_square(File f, Rank r) { return Square((r << 3) + f); }
constexpr File file_of(Square s) { return File(s & 7); }
constexpr Rank rank_of(Square s) { return Rank(s >> 3); }
constexpr bool is_ok(Square s) { return s >= SQ_A1 && s <= SQ_H8; }

// Relative square/rank from a color's perspective (black is mirrored vertically)
constexpr Square relative_square(Color c, Square s) { return Square(s ^ (c * 56)); }
constexpr Rank   relative_rank(Color c, Rank r)     { return Rank(r ^ (c * 7)); }
constexpr Rank   relative_rank(Color c, Square s)   { return relative_rank(c, rank_of(s)); }

// ---------------------------------------------------------------------------
// Directions (as square offsets). Only ever apply via shift helpers in
// bitboard.hpp to stay wrap-safe; raw arithmetic on edge squares is unsafe.
// ---------------------------------------------------------------------------
enum Direction : int {
    NORTH =  8,
    EAST  =  1,
    SOUTH = -NORTH,
    WEST  = -EAST,
    NORTH_EAST = NORTH + EAST,
    NORTH_WEST = NORTH + WEST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST
};

constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
constexpr Square& operator+=(Square& s, Direction d) { return s = s + d; }
constexpr Square& operator-=(Square& s, Direction d) { return s = s - d; }

// ---------------------------------------------------------------------------
// Castling rights (bitmask)
// ---------------------------------------------------------------------------
enum CastlingRights : int {
    NO_CASTLING,
    WHITE_OO  = 1,
    WHITE_OOO = 2,
    BLACK_OO  = 4,
    BLACK_OOO = 8,

    KING_SIDE      = WHITE_OO  | BLACK_OO,
    QUEEN_SIDE     = WHITE_OOO | BLACK_OOO,
    WHITE_CASTLING = WHITE_OO  | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO  | BLACK_OOO,
    ANY_CASTLING   = WHITE_CASTLING | BLACK_CASTLING,
    CASTLING_RIGHT_NB = 16
};

// ---------------------------------------------------------------------------
// Move encoding — 16 bits:
//   bits  0-5 : destination square
//   bits  6-11: origin square
//   bits 12-13: promotion piece type - KNIGHT (0=N, 1=B, 2=R, 3=Q)
//   bits 14-15: move type flag (see MoveType)
// A "none"/"null" move is distinguished by from==to.
// ---------------------------------------------------------------------------
enum MoveType : int {
    NORMAL,
    PROMOTION  = 1 << 14,
    EN_PASSANT = 2 << 14,
    CASTLING   = 3 << 14
};

class Move {
   public:
    Move() = default;
    constexpr explicit Move(uint16_t d) : data(d) {}
    constexpr Move(Square from, Square to) : data((from << 6) + to) {}

    template <MoveType T>
    static constexpr Move make(Square from, Square to, PieceType pt = KNIGHT) {
        return Move(T + ((pt - KNIGHT) << 12) + (from << 6) + to);
    }

    constexpr Square from_sq() const  { return Square((data >> 6) & 0x3F); }
    constexpr Square to_sq() const    { return Square(data & 0x3F); }
    constexpr int    from_to() const  { return data & 0xFFF; }
    constexpr MoveType type_of() const { return MoveType(data & (3 << 14)); }
    constexpr PieceType promotion_type() const { return PieceType(((data >> 12) & 3) + KNIGHT); }

    constexpr bool is_ok() const { return none().data != data && null().data != data; }

    static constexpr Move null() { return Move(65); }  // b1->b1, otherwise invalid
    static constexpr Move none() { return Move(0);  }   // a1->a1

    constexpr bool operator==(const Move& m) const { return data == m.data; }
    constexpr bool operator!=(const Move& m) const { return data != m.data; }
    constexpr explicit operator bool() const { return data != 0; }

    constexpr uint16_t raw() const { return data; }

   private:
    uint16_t data{0};
};

// ---------------------------------------------------------------------------
// Value / score constants
// ---------------------------------------------------------------------------
constexpr Value VALUE_ZERO     = 0;
constexpr Value VALUE_DRAW      = 0;
constexpr Value VALUE_NONE      = 32002;
constexpr Value VALUE_INFINITE  = 32001;
constexpr Value VALUE_MATE      = 32000;
constexpr Value VALUE_MATE_IN_MAX_PLY  = VALUE_MATE - 256;
constexpr Value VALUE_MATED_IN_MAX_PLY = -VALUE_MATE_IN_MAX_PLY;

constexpr int MAX_PLY   = 246;
constexpr int MAX_MOVES = 256;   // upper bound on legal moves in any position

// Baseline piece values (midgame), used by move ordering / SEE / material.
constexpr Value PawnValue   = 100;
constexpr Value KnightValue = 320;
constexpr Value BishopValue = 330;
constexpr Value RookValue   = 500;
constexpr Value QueenValue  = 900;

constexpr Value mate_in(int ply)  { return  VALUE_MATE - ply; }
constexpr Value mated_in(int ply) { return -VALUE_MATE + ply; }

// ---------------------------------------------------------------------------
// String helpers (definitions live in a .cpp / util module)
// ---------------------------------------------------------------------------
std::string square_to_string(Square s);   // e.g. "e4"
std::string move_to_uci(Move m);          // long algebraic, e.g. "e2e4", "e7e8q"

}  // namespace engine

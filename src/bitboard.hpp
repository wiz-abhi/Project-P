// bitboard.hpp — Bitboard type, constants and bit-manipulation helpers.
//
// A Bitboard is a 64-bit word; bit i (i = square index) is set when that square
// is occupied. All engine board logic is expressed as set operations on these.
#pragma once

#include "types.hpp"

namespace engine {

// ---------------------------------------------------------------------------
// Constant masks
// ---------------------------------------------------------------------------
constexpr Bitboard FileABB = 0x0101010101010101ULL;
constexpr Bitboard FileBBB = FileABB << 1;
constexpr Bitboard FileCBB = FileABB << 2;
constexpr Bitboard FileDBB = FileABB << 3;
constexpr Bitboard FileEBB = FileABB << 4;
constexpr Bitboard FileFBB = FileABB << 5;
constexpr Bitboard FileGBB = FileABB << 6;
constexpr Bitboard FileHBB = FileABB << 7;

constexpr Bitboard Rank1BB = 0xFF;
constexpr Bitboard Rank2BB = Rank1BB << (8 * 1);
constexpr Bitboard Rank3BB = Rank1BB << (8 * 2);
constexpr Bitboard Rank4BB = Rank1BB << (8 * 3);
constexpr Bitboard Rank5BB = Rank1BB << (8 * 4);
constexpr Bitboard Rank6BB = Rank1BB << (8 * 5);
constexpr Bitboard Rank7BB = Rank1BB << (8 * 6);
constexpr Bitboard Rank8BB = Rank1BB << (8 * 7);

constexpr Bitboard AllSquares  = ~Bitboard(0);
constexpr Bitboard DarkSquares = 0xAA55AA55AA55AA55ULL;

// ---------------------------------------------------------------------------
// Runtime tables (populated by Bitboards::init(), defined in bitboard.cpp)
// ---------------------------------------------------------------------------
extern Bitboard SquareBB[SQUARE_NB];               // single-bit board per square
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];   // squares strictly between (exclusive), aligned only
extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];      // full line through two aligned squares
extern uint8_t  SquareDistance[SQUARE_NB][SQUARE_NB];

namespace Bitboards {
void init();                 // must be called once at startup (after Attacks::init())
std::string pretty(Bitboard b);   // ASCII diagram for debugging
}  // namespace Bitboards

// ---------------------------------------------------------------------------
// Square <-> bitboard
// ---------------------------------------------------------------------------
inline Bitboard square_bb(Square s) { return SquareBB[s]; }

inline Bitboard  operator&(Bitboard b, Square s) { return b & square_bb(s); }
inline Bitboard  operator|(Bitboard b, Square s) { return b | square_bb(s); }
inline Bitboard  operator^(Bitboard b, Square s) { return b ^ square_bb(s); }
inline Bitboard& operator|=(Bitboard& b, Square s) { return b |= square_bb(s); }
inline Bitboard& operator^=(Bitboard& b, Square s) { return b ^= square_bb(s); }

constexpr Bitboard file_bb(File f) { return FileABB << f; }
constexpr Bitboard file_bb(Square s) { return FileABB << file_of(s); }
constexpr Bitboard rank_bb(Rank r) { return Rank1BB << (8 * r); }
constexpr Bitboard rank_bb(Square s) { return Rank1BB << (8 * rank_of(s)); }

// ---------------------------------------------------------------------------
// Wrap-safe directional shifts of a whole bitboard
// ---------------------------------------------------------------------------
template <Direction D>
constexpr Bitboard shift(Bitboard b) {
    return D == NORTH        ? b << 8
         : D == SOUTH        ? b >> 8
         : D == NORTH + NORTH ? b << 16
         : D == SOUTH + SOUTH ? b >> 16
         : D == EAST         ? (b & ~FileHBB) << 1
         : D == WEST         ? (b & ~FileABB) >> 1
         : D == NORTH_EAST   ? (b & ~FileHBB) << 9
         : D == NORTH_WEST   ? (b & ~FileABB) << 7
         : D == SOUTH_EAST   ? (b & ~FileHBB) >> 7
         : D == SOUTH_WEST   ? (b & ~FileABB) >> 9
         : 0;
}

// Pawn attack spans for a set of pawns of the given color
template <Color C>
constexpr Bitboard pawn_attacks_bb(Bitboard b) {
    return C == WHITE ? shift<NORTH_WEST>(b) | shift<NORTH_EAST>(b)
                      : shift<SOUTH_WEST>(b) | shift<SOUTH_EAST>(b);
}

// ---------------------------------------------------------------------------
// Population count / bit scan (use GCC/Clang builtins — 64-bit fast paths)
// ---------------------------------------------------------------------------
inline int popcount(Bitboard b) { return __builtin_popcountll(b); }

inline Square lsb(Bitboard b) { return Square(__builtin_ctzll(b)); }  // b != 0
inline Square msb(Bitboard b) { return Square(63 ^ __builtin_clzll(b)); }  // b != 0

inline Square pop_lsb(Bitboard& b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

inline bool more_than_one(Bitboard b) { return b & (b - 1); }

inline Square frontmost_sq(Color c, Bitboard b) { return c == WHITE ? msb(b) : lsb(b); }

// ---------------------------------------------------------------------------
// Geometry helpers (backed by runtime tables)
// ---------------------------------------------------------------------------
inline int distance(Square a, Square b) { return SquareDistance[a][b]; }

template <typename T = Square>
inline int distance(Square a, Square b);
template <>
inline int distance<File>(Square a, Square b) { return std::abs(file_of(a) - file_of(b)); }
template <>
inline int distance<Rank>(Square a, Square b) { return std::abs(rank_of(a) - rank_of(b)); }

inline Bitboard between_bb(Square a, Square b) { return BetweenBB[a][b]; }
inline Bitboard line_bb(Square a, Square b) { return LineBB[a][b]; }

// Are three squares collinear (on a shared file/rank/diagonal)?
inline bool aligned(Square a, Square b, Square c) { return line_bb(a, b) & c; }

}  // namespace engine

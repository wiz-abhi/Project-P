// position.hpp — The board representation and make/unmake move machinery.
//
// Position holds bitboards + a mailbox array and an incrementally-updated
// Zobrist key. Irreversible state needed to undo a move lives in StateInfo,
// which forms a linked list so do_move/undo_move are O(1).
#pragma once

#include <string>

#include "attacks.hpp"
#include "bitboard.hpp"
#include "types.hpp"

namespace engine {

// Per-move irreversible state. A new StateInfo is pushed by do_move.
struct StateInfo {
    // Copied from the previous state (part of the reversible-hash / rule data).
    Key      pawnKey;
    Value    nonPawnMaterial[COLOR_NB];
    int      castlingRights;
    int      rule50;          // halfmove clock for the 50-move rule
    int      pliesFromNull;
    Square   epSquare;        // en-passant target square or SQ_NONE

    // Recomputed by do_move.
    Key      key;             // full Zobrist key of the resulting position
    Bitboard checkersBB;      // pieces giving check to the side to move
    Piece    capturedPiece;
    Bitboard blockersForKing[COLOR_NB];  // own pieces pinned to own king
    Bitboard pinners[COLOR_NB];          // enemy sliders pinning them
    Bitboard checkSquares[PIECE_TYPE_NB];// squares from which each pt gives check
    int      repetition;      // 0 none; >0 draw within search; <0 earlier in game

    StateInfo* previous;
};

class Position {
   public:
    static void init();  // one-time global init (Zobrist etc.)

    Position() = default;
    Position(const Position&) = delete;
    Position& operator=(const Position&) = delete;

    // Setup / FEN
    Position& set(const std::string& fen, StateInfo* si);
    std::string fen() const;

    // Piece / occupancy queries
    Piece    piece_on(Square s) const { return board[s]; }
    bool     empty(Square s) const { return board[s] == NO_PIECE; }
    Bitboard pieces() const { return byTypeBB[ALL_PIECES]; }
    Bitboard pieces(PieceType pt) const { return byTypeBB[pt]; }
    Bitboard pieces(PieceType p1, PieceType p2) const { return byTypeBB[p1] | byTypeBB[p2]; }
    Bitboard pieces(Color c) const { return byColorBB[c]; }
    Bitboard pieces(Color c, PieceType pt) const { return byColorBB[c] & byTypeBB[pt]; }
    Bitboard pieces(Color c, PieceType p1, PieceType p2) const { return pieces(c) & (byTypeBB[p1] | byTypeBB[p2]); }
    template <typename... Pts>
    Bitboard pieces(Color c, Pts... pts) const { return pieces(c) & (byTypeBB[pts] | ...); }

    int      count(Piece pc) const { return pieceCount[pc]; }
    template <PieceType Pt> int count(Color c) const { return pieceCount[make_piece(c, Pt)]; }
    template <PieceType Pt> int count() const { return count<Pt>(WHITE) + count<Pt>(BLACK); }
    Square   square(Color c, PieceType pt) const { return lsb(pieces(c, pt)); }  // for singletons (king)
    Square   king_square(Color c) const { return square(c, KING); }

    // Side, clocks, castling
    Color    side_to_move() const { return sideToMove; }
    int      game_ply() const { return gamePly; }
    int      rule50_count() const { return st->rule50; }
    Square   ep_square() const { return st->epSquare; }
    int      castling_rights() const { return st->castlingRights; }
    bool     can_castle(CastlingRights cr) const { return st->castlingRights & cr; }
    bool     castling_impeded(CastlingRights cr) const;
    Square   castling_rook_square(CastlingRights cr) const;

    // Checks / pins / attacks
    Bitboard checkers() const { return st->checkersBB; }
    Bitboard blockers_for_king(Color c) const { return st->blockersForKing[c]; }
    Bitboard pinners(Color c) const { return st->pinners[c]; }
    Bitboard check_squares(PieceType pt) const { return st->checkSquares[pt]; }
    Bitboard attackers_to(Square s) const { return attackers_to(s, pieces()); }
    Bitboard attackers_to(Square s, Bitboard occ) const;
    bool     is_attacked_by(Color c, Square s) const;  // does side c attack square s

    // Move properties
    bool     legal(Move m) const;
    bool     pseudo_legal(Move m) const;
    bool     capture(Move m) const;
    bool     capture_stage(Move m) const;   // capture or promotion (for qsearch)
    bool     gives_check(Move m) const;
    Piece    moved_piece(Move m) const { return board[m.from_sq()]; }
    Piece    captured_piece() const { return st->capturedPiece; }

    // Make / unmake
    void     do_move(Move m, StateInfo& newSt);
    void     do_move(Move m, StateInfo& newSt, bool givesCheck);
    void     undo_move(Move m);
    void     do_null_move(StateInfo& newSt);
    void     undo_null_move();

    // Static Exchange Evaluation: is the see value of m >= threshold?
    bool     see_ge(Move m, Value threshold = VALUE_ZERO) const;

    // Hashing / draw detection
    Key      key() const { return st->key; }
    Key      pawn_key() const { return st->pawnKey; }
    bool     is_draw(int ply) const;             // repetition or 50-move
    bool     has_repeated() const;
    bool     upcoming_repetition(int ply) const; // cuckoo / early repetition test
    Value    non_pawn_material(Color c) const { return st->nonPawnMaterial[c]; }
    Value    non_pawn_material() const { return non_pawn_material(WHITE) + non_pawn_material(BLACK); }

    // Phase / misc
    int      game_phase() const;   // 0 (endgame) .. 24 (opening), for tapered eval
    bool     checkers_exist() const { return st->checkersBB != 0; }

    // Debug
    bool     pos_is_ok() const;
    std::string to_string() const;   // ASCII board

   private:
    // Internal helpers used by do/undo (defined in position.cpp).
    void put_piece(Piece pc, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
    template <bool Do> void do_castling(Color us, Square from, Square& to, Square& rfrom, Square& rto);
    void set_castling_right(Color c, Square rfrom);
    void set_state() const;
    void set_check_info() const;
    Bitboard slider_blockers(Bitboard sliders, Square s, Bitboard& pinners) const;

    // Board data
    Piece    board[SQUARE_NB];
    Bitboard byTypeBB[PIECE_TYPE_NB];
    Bitboard byColorBB[COLOR_NB];
    int      pieceCount[PIECE_NB];
    int      castlingRightsMask[SQUARE_NB];
    Square   castlingRookSquare[CASTLING_RIGHT_NB];
    Bitboard castlingPath[CASTLING_RIGHT_NB];
    Color    sideToMove;
    int      gamePly;
    StateInfo* st;
};

}  // namespace engine

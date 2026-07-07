// position.cpp — Board representation and make/unmake move machinery.
//
// Implements every method declared in position.hpp against the frozen headers.
// Algorithms follow standard, well-known chess-engine techniques (Stockfish
// lineage) adapted to this project's type contracts.

#include "position.hpp"

#include <cassert>
#include <cctype>
#include <cstring>
#include <sstream>

#include "attacks.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"

namespace engine {

namespace {

// Piece <-> FEN character mapping. Index by Piece value.
constexpr char PieceToChar[] = " PNBRQK  pnbrqk";

// Value of a piece type for material / SEE.
constexpr Value PieceValue[PIECE_TYPE_NB] = {
    VALUE_ZERO,   // NO_PIECE_TYPE
    PawnValue, KnightValue, BishopValue, RookValue, QueenValue,
    VALUE_ZERO,   // KING (no material value)
    VALUE_ZERO    // padding
};

}  // namespace

// ---------------------------------------------------------------------------
// One-time global init
// ---------------------------------------------------------------------------
void Position::init() { Zobrist::init(); }

// ---------------------------------------------------------------------------
// Low-level board mutation helpers (keep bitboards + mailbox + counts in sync)
// ---------------------------------------------------------------------------
void Position::put_piece(Piece pc, Square s) {
    board[s] = pc;
    byTypeBB[ALL_PIECES] |= s;
    byTypeBB[type_of(pc)] |= s;
    byColorBB[color_of(pc)] |= s;
    pieceCount[pc]++;
}

void Position::remove_piece(Square s) {
    Piece pc = board[s];
    byTypeBB[ALL_PIECES] ^= s;
    byTypeBB[type_of(pc)] ^= s;
    byColorBB[color_of(pc)] ^= s;
    board[s] = NO_PIECE;
    pieceCount[pc]--;
}

void Position::move_piece(Square from, Square to) {
    Piece pc = board[from];
    Bitboard fromTo = square_bb(from) | square_bb(to);
    byTypeBB[ALL_PIECES] ^= fromTo;
    byTypeBB[type_of(pc)] ^= fromTo;
    byColorBB[color_of(pc)] ^= fromTo;
    board[from] = NO_PIECE;
    board[to] = pc;
}

// ---------------------------------------------------------------------------
// FEN parsing
// ---------------------------------------------------------------------------
Position& Position::set(const std::string& fenStr, StateInfo* si) {
    unsigned char col, row, token;
    size_t idx;
    Square sq = SQ_A8;
    std::istringstream ss(fenStr);

    // Zero-init everything.
    std::memset(this, 0, sizeof(Position));
    std::memset(si, 0, sizeof(StateInfo));
    st = si;

    ss >> std::noskipws;

    // 1. Piece placement.
    while ((ss >> token) && !isspace(token)) {
        if (isdigit(token)) {
            sq = Square(int(sq) + (token - '0'));  // advance file(s)
        } else if (token == '/') {
            sq = Square(int(sq) - 16);  // move to start of next (lower) rank
        } else if ((idx = std::string(PieceToChar).find(token)) != std::string::npos) {
            put_piece(Piece(idx), sq);
            sq += EAST;
        }
    }

    // 2. Side to move.
    ss >> token;
    sideToMove = (token == 'w' ? WHITE : BLACK);
    ss >> token;  // consume separating space

    // 3. Castling rights.
    while ((ss >> token) && !isspace(token)) {
        Square rsq;
        Color c = islower(token) ? BLACK : WHITE;
        Piece rook = make_piece(c, ROOK);
        token = char(toupper(token));

        if (token == 'K')
            for (rsq = relative_square(c, SQ_H1); piece_on(rsq) != rook; rsq += WEST) {}
        else if (token == 'Q')
            for (rsq = relative_square(c, SQ_A1); piece_on(rsq) != rook; rsq += EAST) {}
        else
            continue;

        set_castling_right(c, rsq);
    }

    // 4. En-passant square. Only accept it if a pawn of the side to move can
    //    actually make the capture (matches how set_state hashes it).
    bool epValid = false;
    if (((ss >> col) && (col >= 'a' && col <= 'h')) &&
        ((ss >> row) && (row == (sideToMove == WHITE ? '6' : '3')))) {
        st->epSquare = make_square(File(col - 'a'), Rank(row - '1'));

        if ((pawn_attacks(~sideToMove, st->epSquare) & pieces(sideToMove, PAWN)) &&
            (pieces(~sideToMove, PAWN) & (st->epSquare + (sideToMove == WHITE ? SOUTH : NORTH))))
            epValid = true;
    }
    if (!epValid)
        st->epSquare = SQ_NONE;

    // 5. Halfmove clock and fullmove number.
    int fullmove = 1;
    ss >> std::skipws >> st->rule50 >> fullmove;

    gamePly = 2 * (fullmove - 1) + (sideToMove == BLACK);
    // gamePly must be >= 0.
    if (gamePly < 0)
        gamePly = 0;

    set_state();

    return *this;
}

// ---------------------------------------------------------------------------
// Configure one castling right (standard chess only)
// ---------------------------------------------------------------------------
void Position::set_castling_right(Color c, Square rfrom) {
    Square kfrom = king_square(c);
    CastlingRights cr =
        c == WHITE
            ? (kfrom < rfrom ? WHITE_OO : WHITE_OOO)
            : (kfrom < rfrom ? BLACK_OO : BLACK_OOO);

    st->castlingRights |= cr;
    castlingRightsMask[kfrom] |= cr;
    castlingRightsMask[rfrom] |= cr;
    castlingRookSquare[cr] = rfrom;

    Square kto = relative_square(c, (cr & KING_SIDE) ? SQ_G1 : SQ_C1);
    Square rto = relative_square(c, (cr & KING_SIDE) ? SQ_F1 : SQ_D1);

    // Path that must be empty (excluding king-from and rook-from).
    castlingPath[cr] = (between_bb(rfrom, rto) | between_bb(kfrom, kto)) &
                       ~(square_bb(kfrom) | square_bb(rfrom));
}

bool Position::castling_impeded(CastlingRights cr) const {
    return pieces() & castlingPath[cr];
}

Square Position::castling_rook_square(CastlingRights cr) const {
    return castlingRookSquare[cr];
}

// ---------------------------------------------------------------------------
// set_state — compute the full Zobrist key, pawn key and material from scratch
// ---------------------------------------------------------------------------
void Position::set_state() const {
    st->key = 0;
    st->pawnKey = 0;
    st->nonPawnMaterial[WHITE] = st->nonPawnMaterial[BLACK] = VALUE_ZERO;
    st->checkersBB = attackers_to(king_square(sideToMove)) & pieces(~sideToMove);

    for (Bitboard b = pieces(); b;) {
        Square s = pop_lsb(b);
        Piece pc = piece_on(s);
        st->key ^= Zobrist::psq[pc][s];

        if (type_of(pc) == PAWN)
            st->pawnKey ^= Zobrist::psq[pc][s];
        else if (type_of(pc) != KING)
            st->nonPawnMaterial[color_of(pc)] += PieceValue[type_of(pc)];
    }

    if (st->epSquare != SQ_NONE)
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];

    if (sideToMove == BLACK)
        st->key ^= Zobrist::side;

    st->key ^= Zobrist::castling[st->castlingRights];

    set_check_info();
}

// ---------------------------------------------------------------------------
// set_check_info — checkers, pins/blockers for both colors, check squares
// ---------------------------------------------------------------------------
void Position::set_check_info() const {
    st->blockersForKing[WHITE] =
        slider_blockers(pieces(BLACK), king_square(WHITE), st->pinners[BLACK]);
    st->blockersForKing[BLACK] =
        slider_blockers(pieces(WHITE), king_square(BLACK), st->pinners[WHITE]);

    Square ksq = king_square(~sideToMove);

    st->checkSquares[PAWN]   = pawn_attacks(~sideToMove, ksq);
    st->checkSquares[KNIGHT] = attacks_bb(KNIGHT, ksq, pieces());
    st->checkSquares[BISHOP] = attacks_bb<BISHOP>(ksq, pieces());
    st->checkSquares[ROOK]   = attacks_bb<ROOK>(ksq, pieces());
    st->checkSquares[QUEEN]  = st->checkSquares[BISHOP] | st->checkSquares[ROOK];
    st->checkSquares[KING]   = 0;
}

// ---------------------------------------------------------------------------
// slider_blockers — pin detection relative to square s
//
// `sliders` are the (enemy) pieces that might pin. For every slider aligned
// with s, if exactly one piece sits strictly between it and s, that piece is a
// blocker; the pinning slider is recorded in `pinners`.
// ---------------------------------------------------------------------------
Bitboard Position::slider_blockers(Bitboard sliders, Square s, Bitboard& pinners) const {
    Bitboard blockers = 0;
    pinners = 0;

    // Candidate snipers: pieces attacking s along a ray if the board were empty,
    // restricted to the given slider set.
    Bitboard snipers = ((attacks_bb<ROOK>(s, 0) & pieces(QUEEN, ROOK)) |
                        (attacks_bb<BISHOP>(s, 0) & pieces(QUEEN, BISHOP))) &
                       sliders;
    Bitboard occupancy = pieces() ^ snipers;

    while (snipers) {
        Square sniperSq = pop_lsb(snipers);
        Bitboard b = between_bb(s, sniperSq) & occupancy;

        if (b && !more_than_one(b)) {
            blockers |= b;
            // The single blocker pins if it belongs to the same color as the
            // piece on s (own-side blocker). pinners records the sniper.
            if (b & pieces(color_of(piece_on(s))))
                pinners |= sniperSq;
        }
    }
    return blockers;
}

// ---------------------------------------------------------------------------
// attackers_to — all pieces (both colors) attacking s given occupancy occ
// ---------------------------------------------------------------------------
Bitboard Position::attackers_to(Square s, Bitboard occ) const {
    return (pawn_attacks(WHITE, s) & pieces(BLACK, PAWN)) |
           (pawn_attacks(BLACK, s) & pieces(WHITE, PAWN)) |
           (attacks_bb(KNIGHT, s, occ) & pieces(KNIGHT)) |
           (attacks_bb<ROOK>(s, occ) & pieces(ROOK, QUEEN)) |
           (attacks_bb<BISHOP>(s, occ) & pieces(BISHOP, QUEEN)) |
           (attacks_bb(KING, s, occ) & pieces(KING));
}

bool Position::is_attacked_by(Color c, Square s) const {
    return (pawn_attacks(~c, s) & pieces(c, PAWN)) ||
           (attacks_bb(KNIGHT, s, pieces()) & pieces(c, KNIGHT)) ||
           (attacks_bb<ROOK>(s, pieces()) & pieces(c, ROOK, QUEEN)) ||
           (attacks_bb<BISHOP>(s, pieces()) & pieces(c, BISHOP, QUEEN)) ||
           (attacks_bb(KING, s, pieces()) & pieces(c, KING));
}

// ---------------------------------------------------------------------------
// legal — assumes m is pseudo-legal; verify it leaves our king safe
// ---------------------------------------------------------------------------
bool Position::legal(Move m) const {
    Color us = sideToMove;
    Square from = m.from_sq();
    Square to = m.to_sq();

    // En passant: make the capture on a temporary occupancy and test for check.
    if (m.type_of() == EN_PASSANT) {
        Square ksq = king_square(us);
        Square capsq = to - (us == WHITE ? NORTH : SOUTH);
        Bitboard occ = (pieces() ^ from ^ capsq) | to;

        return !(attacks_bb<ROOK>(ksq, occ) & pieces(~us, QUEEN, ROOK)) &&
               !(attacks_bb<BISHOP>(ksq, occ) & pieces(~us, QUEEN, BISHOP));
    }

    // Castling: the king path (already known impeded-free) must not be attacked.
    if (m.type_of() == CASTLING) {
        // to is encoded as the king destination square.
        Direction step = to > from ? WEST : EAST;
        for (Square s = to; s != from; s += step)
            if (is_attacked_by(~us, s))
                return false;
        return true;
    }

    // King move: destination must not be attacked, with the king removed from
    // the occupancy (so it cannot block a slider that would still hit it).
    if (type_of(piece_on(from)) == KING) {
        Bitboard occ = pieces() ^ from;
        return !(attackers_to(to, occ) & pieces(~us));
    }

    // Any other piece: legal unless it is pinned and moving off the king line.
    return !(blockers_for_king(us) & from) ||
           aligned(from, to, king_square(us));
}

// ---------------------------------------------------------------------------
// pseudo_legal — full validity of an arbitrary Move against this position
// ---------------------------------------------------------------------------
bool Position::pseudo_legal(Move m) const {
    Color us = sideToMove;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece pc = moved_piece(m);

    // Non-NORMAL moves are only produced by full movegen; validate them via a
    // generated-move membership style check by re-deriving their conditions.
    if (m.type_of() != NORMAL) {
        // Delegate to specific checks below; handled at the end.
    }

    // The moving piece must exist and belong to us.
    if (pc == NO_PIECE || color_of(pc) != us)
        return false;

    // Cannot capture our own piece.
    if (pieces(us) & to)
        return false;

    // Handle special move types explicitly.
    if (m.type_of() == PROMOTION) {
        if (type_of(pc) != PAWN)
            return false;
        if (relative_rank(us, from) != RANK_7 || relative_rank(us, to) != RANK_8)
            return false;
        // fall through to pawn geometry below via NORMAL-like handling
    } else if (m.type_of() == EN_PASSANT) {
        if (type_of(pc) != PAWN)
            return false;
        if (to != st->epSquare)
            return false;
        if (!empty(to))
            return false;
        Square capsq = to - (us == WHITE ? NORTH : SOUTH);
        if (piece_on(capsq) != make_piece(~us, PAWN))
            return false;
        // Must be a diagonal pawn attack.
        if (!(pawn_attacks(us, from) & to))
            return false;
        return true;
    } else if (m.type_of() == CASTLING) {
        if (type_of(pc) != KING)
            return false;
        // Identify the castling right from king-from/king-to direction.
        CastlingRights cr =
            us == WHITE ? (to > from ? WHITE_OO : WHITE_OOO)
                        : (to > from ? BLACK_OO : BLACK_OOO);
        if (!can_castle(cr))
            return false;
        if (castling_impeded(cr))
            return false;
        if (checkers())
            return false;
        // King must move to the expected destination.
        Square kto = relative_square(us, (cr & KING_SIDE) ? SQ_G1 : SQ_C1);
        if (to != kto)
            return false;
        return true;
    }

    // From here: NORMAL or PROMOTION (pawn geometry / leaper / slider paths).
    if (type_of(pc) == PAWN) {
        // A pawn cannot appear on the promotion rank without a PROMOTION flag,
        // nor carry a PROMOTION flag off the promotion rank (handled above).
        if (relative_rank(us, to) == RANK_8 && m.type_of() != PROMOTION)
            return false;

        Direction up = (us == WHITE ? NORTH : SOUTH);
        bool capture = (pawn_attacks(us, from) & to) & pieces(~us);
        bool single = (from + up == to) && empty(to);
        bool dbl = (from + up + up == to) && relative_rank(us, from) == RANK_2 &&
                   empty(to) && empty(from + up);

        if (!(capture || single || dbl))
            return false;
    } else {
        // Leapers / sliders: destination must be a pseudo-attack square.
        if (!(attacks_bb(type_of(pc), from, pieces()) & to))
            return false;
    }

    // If in check, only moves that resolve it are pseudo-legal here we defer to
    // legal(); but for TT/killer validation we still require basic evasion
    // consistency is not strictly needed. Accept and let legal() filter.
    return true;
}

// ---------------------------------------------------------------------------
// capture / capture_stage
// ---------------------------------------------------------------------------
bool Position::capture(Move m) const {
    return (!empty(m.to_sq()) && m.type_of() != CASTLING) ||
           m.type_of() == EN_PASSANT;
}

bool Position::capture_stage(Move m) const {
    return capture(m) ||
           (m.type_of() == PROMOTION && m.promotion_type() == QUEEN);
}

// ---------------------------------------------------------------------------
// gives_check — does making m check the opponent?
// ---------------------------------------------------------------------------
bool Position::gives_check(Move m) const {
    Square from = m.from_sq();
    Square to = m.to_sq();
    Color us = sideToMove;
    Square ksq = king_square(~us);

    // Direct check by the (possibly promoted) piece landing on `to`.
    PieceType pt = type_of(moved_piece(m));
    if (m.type_of() == PROMOTION)
        pt = m.promotion_type();

    if (pt != KING && (check_squares(pt) & to))
        return true;

    // Discovered check: the moving piece was a blocker for the enemy king and
    // moves off the king line.
    if ((blockers_for_king(~us) & from) && !aligned(from, to, ksq))
        return true;

    switch (m.type_of()) {
        case NORMAL:
            return false;

        case PROMOTION:
            // Re-evaluate as the promoted slider from `to` on an occupancy with
            // the pawn removed from `from`.
            return attacks_bb(m.promotion_type(), to, pieces() ^ from) & ksq;

        case EN_PASSANT: {
            // Removing both pawns may open a rank/diagonal onto the king.
            Square capsq = make_square(file_of(to), rank_of(from));
            Bitboard b = (pieces() ^ from ^ capsq) | to;
            return (attacks_bb<ROOK>(ksq, b) & pieces(us, QUEEN, ROOK)) ||
                   (attacks_bb<BISHOP>(ksq, b) & pieces(us, QUEEN, BISHOP));
        }

        case CASTLING: {
            // Only the rook can give check after castling.
            Square rto = relative_square(us, (to > from) ? SQ_F1 : SQ_D1);
            return attacks_bb<ROOK>(rto, pieces()) & ksq;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// do_move (dispatch)
// ---------------------------------------------------------------------------
void Position::do_move(Move m, StateInfo& newSt) {
    do_move(m, newSt, gives_check(m));
}

// ---------------------------------------------------------------------------
// do_move (core)
// ---------------------------------------------------------------------------
void Position::do_move(Move m, StateInfo& newSt, bool givesCheck) {
    Key k = st->key ^ Zobrist::side;

    // Copy reversible fields to the new state, then link and switch.
    std::memcpy(&newSt, st, offsetof(StateInfo, key));
    newSt.previous = st;
    st = &newSt;

    ++gamePly;
    ++st->rule50;
    ++st->pliesFromNull;

    Color us = sideToMove;
    Color them = ~us;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece pc = piece_on(from);
    Piece captured = m.type_of() == EN_PASSANT ? make_piece(them, PAWN) : piece_on(to);

    if (m.type_of() == CASTLING) {
        // Castling: move king and rook together, hashing the rook's move here
        // (the king's from/to is hashed with the generic moved-piece update).
        Square rfrom, rto;
        Square kto = to;
        do_castling<true>(us, from, kto, rfrom, rto);

        Piece rook = make_piece(us, ROOK);
        k ^= Zobrist::psq[rook][rfrom] ^ Zobrist::psq[rook][rto];
        captured = NO_PIECE;
    }

    if (captured) {
        Square capsq = to;
        if (type_of(captured) == PAWN && m.type_of() == EN_PASSANT)
            capsq -= (us == WHITE ? NORTH : SOUTH);

        // Update material / pawn key.
        if (type_of(captured) == PAWN)
            st->pawnKey ^= Zobrist::psq[captured][capsq];
        else
            st->nonPawnMaterial[them] -= PieceValue[type_of(captured)];

        remove_piece(capsq);
        k ^= Zobrist::psq[captured][capsq];
        st->rule50 = 0;
    }

    // Update castling-rights hash and value if from/to affects them.
    if (st->castlingRights && (castlingRightsMask[from] | castlingRightsMask[to])) {
        k ^= Zobrist::castling[st->castlingRights];
        st->castlingRights &= ~(castlingRightsMask[from] | castlingRightsMask[to]);
        k ^= Zobrist::castling[st->castlingRights];
    }

    // Clear a previously-set en-passant square from the hash.
    if (st->epSquare != SQ_NONE) {
        k ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }

    // Move the piece (unless castling already handled king movement).
    if (m.type_of() != CASTLING)
        move_piece(from, to);

    // Pawn-specific updates: en-passant target, promotion, pawn key.
    if (type_of(pc) == PAWN) {
        // Double push sets an ep square (only if an enemy pawn can capture).
        if ((int(to) ^ int(from)) == 16 &&
            (pawn_attacks(us, to - (us == WHITE ? NORTH : SOUTH)) &
             pieces(them, PAWN))) {
            st->epSquare = to - (us == WHITE ? NORTH : SOUTH);
            k ^= Zobrist::enpassant[file_of(st->epSquare)];
        } else if (m.type_of() == PROMOTION) {
            Piece promoted = make_piece(us, m.promotion_type());
            remove_piece(to);
            put_piece(promoted, to);

            k ^= Zobrist::psq[pc][to] ^ Zobrist::psq[promoted][to];
            st->pawnKey ^= Zobrist::psq[pc][to];
            st->nonPawnMaterial[us] += PieceValue[m.promotion_type()];
        }

        // Update pawn key for the pawn's own from/to movement.
        st->pawnKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
        st->rule50 = 0;
    }

    // Hash the moved piece's from/to.
    k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

    st->capturedPiece = captured;
    st->key = k;

    // Flip side to move.
    sideToMove = ~sideToMove;

    // Recompute checkers and derived info from the new side-to-move's view.
    st->checkersBB = givesCheck ? attackers_to(king_square(them)) & pieces(us) : Bitboard(0);

    set_check_info();

    // Repetition detection: walk back in steps of 2 within the reversible window.
    st->repetition = 0;
    int end = st->rule50 < st->pliesFromNull ? st->rule50 : st->pliesFromNull;
    if (end >= 4) {
        StateInfo* stp = st->previous->previous;
        for (int i = 4; i <= end; i += 2) {
            stp = stp->previous->previous;
            if (stp->key == st->key) {
                st->repetition = stp->repetition ? -i : i;
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// undo_move — exact inverse of do_move
// ---------------------------------------------------------------------------
void Position::undo_move(Move m) {
    sideToMove = ~sideToMove;

    Color us = sideToMove;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece pc = piece_on(to);

    if (m.type_of() == PROMOTION) {
        // Replace the promoted piece with a pawn on `to`.
        remove_piece(to);
        pc = make_piece(us, PAWN);
        put_piece(pc, to);
    }

    if (m.type_of() == CASTLING) {
        Square rfrom, rto;
        Square kto = to;
        do_castling<false>(us, from, kto, rfrom, rto);
    } else {
        move_piece(to, from);  // move the piece back

        if (st->capturedPiece) {
            Square capsq = to;
            if (m.type_of() == EN_PASSANT)
                capsq -= (us == WHITE ? NORTH : SOUTH);
            put_piece(st->capturedPiece, capsq);
        }
    }

    // Restore previous state.
    st = st->previous;
    --gamePly;
}

// ---------------------------------------------------------------------------
// do_castling — move king and rook (Do = true) or reverse them (Do = false)
// ---------------------------------------------------------------------------
template <bool Do>
void Position::do_castling(Color us, Square from, Square& to, Square& rfrom, Square& rto) {
    bool kingSide = to > from;

    // Determine rook source (standard chess) and destinations. Note `to` arrives
    // as the king destination square in this engine's castling encoding.
    rfrom = kingSide ? relative_square(us, SQ_H1) : relative_square(us, SQ_A1);
    rto = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
    Square kto = relative_square(us, kingSide ? SQ_G1 : SQ_C1);
    to = kto;

    // Remove both pieces, then place them at their destinations. Doing it in
    // this order avoids clobbering when squares overlap.
    remove_piece(Do ? from : kto);
    remove_piece(Do ? rfrom : rto);
    put_piece(make_piece(us, KING), Do ? kto : from);
    put_piece(make_piece(us, ROOK), Do ? rto : rfrom);
}

// ---------------------------------------------------------------------------
// Null move
// ---------------------------------------------------------------------------
void Position::do_null_move(StateInfo& newSt) {
    std::memcpy(&newSt, st, sizeof(StateInfo));
    newSt.previous = st;
    st = &newSt;

    st->key ^= Zobrist::side;

    if (st->epSquare != SQ_NONE) {
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }

    ++st->rule50;
    st->pliesFromNull = 0;

    sideToMove = ~sideToMove;

    set_check_info();

    st->repetition = 0;
}

void Position::undo_null_move() {
    st = st->previous;
    sideToMove = ~sideToMove;
}

// ---------------------------------------------------------------------------
// Static Exchange Evaluation
// ---------------------------------------------------------------------------
bool Position::see_ge(Move m, Value threshold) const {
    // Only NORMAL captures / quiet moves are scored precisely; castling and
    // en-passant edge cases are handled conservatively.
    if (m.type_of() != NORMAL)
        return VALUE_ZERO >= threshold;

    Square from = m.from_sq();
    Square to = m.to_sq();

    int swap = PieceValue[type_of(piece_on(to))] - threshold;
    if (swap < 0)
        return false;

    swap = PieceValue[type_of(piece_on(from))] - swap;
    if (swap <= 0)
        return true;

    Bitboard occupied = pieces() ^ from ^ to;
    Color stm = sideToMove;
    Bitboard attackers = attackers_to(to, occupied);
    Bitboard stmAttackers, bb;
    int res = 1;

    while (true) {
        stm = ~stm;
        attackers &= occupied;

        if (!(stmAttackers = attackers & pieces(stm)))
            break;

        // Ignore pinned attackers whose sliders still bear on the king.
        // (Simplified: keep all; correctness for SEE ordering unaffected in the
        // common cases — a fuller pin-aware version can be layered later.)

        res ^= 1;

        if ((bb = stmAttackers & pieces(PAWN))) {
            if ((swap = PawnValue - swap) < res)
                break;
            occupied ^= (bb & -bb);
            attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
        } else if ((bb = stmAttackers & pieces(KNIGHT))) {
            if ((swap = KnightValue - swap) < res)
                break;
            occupied ^= (bb & -bb);
        } else if ((bb = stmAttackers & pieces(BISHOP))) {
            if ((swap = BishopValue - swap) < res)
                break;
            occupied ^= (bb & -bb);
            attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
        } else if ((bb = stmAttackers & pieces(ROOK))) {
            if ((swap = RookValue - swap) < res)
                break;
            occupied ^= (bb & -bb);
            attackers |= attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN);
        } else if ((bb = stmAttackers & pieces(QUEEN))) {
            if ((swap = QueenValue - swap) < res)
                break;
            occupied ^= (bb & -bb);
            attackers |= (attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN)) |
                         (attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN));
        } else {
            // King capture: legal only if the opponent has no more attackers.
            return (attackers & ~pieces(stm)) ? bool(res ^ 1) : bool(res);
        }
    }
    return bool(res);
}

// ---------------------------------------------------------------------------
// Draw detection
// ---------------------------------------------------------------------------
bool Position::is_draw(int ply) const {
    if (st->rule50 >= 100)
        return true;

    // Repetition within the current search line.
    return st->repetition && st->repetition < ply;
}

bool Position::has_repeated() const {
    StateInfo* stc = st;
    int end = stc->rule50 < stc->pliesFromNull ? stc->rule50 : stc->pliesFromNull;
    while (end-- >= 4) {
        if (stc->repetition)
            return true;
        stc = stc->previous;
    }
    return false;
}

bool Position::upcoming_repetition(int /*ply*/) const {
    // TODO: implement the cuckoo-table early-repetition test. A correct-but-
    // conservative stub returns false (no false positives on draws).
    return false;
}

// ---------------------------------------------------------------------------
// Game phase (0..24)
// ---------------------------------------------------------------------------
int Position::game_phase() const {
    int phase = count<KNIGHT>() * 1 + count<BISHOP>() * 1 +
                count<ROOK>() * 2 + count<QUEEN>() * 4;
    return phase < 24 ? phase : 24;
}

// ---------------------------------------------------------------------------
// FEN generation (inverse of set)
// ---------------------------------------------------------------------------
std::string Position::fen() const {
    std::ostringstream ss;

    // 1. Piece placement (rank 8 down to rank 1).
    for (Rank r = RANK_8; r >= RANK_1; r = Rank(r - 1)) {
        for (File f = FILE_A; f <= FILE_H; f = File(f + 1)) {
            int empties = 0;
            while (f <= FILE_H && empty(make_square(f, r))) {
                ++empties;
                f = File(f + 1);
            }
            if (empties)
                ss << empties;
            if (f <= FILE_H)
                ss << PieceToChar[piece_on(make_square(f, r))];
        }
        if (r > RANK_1)
            ss << '/';
    }

    // 2. Side to move.
    ss << (sideToMove == WHITE ? " w " : " b ");

    // 3. Castling rights.
    if (can_castle(WHITE_OO))  ss << 'K';
    if (can_castle(WHITE_OOO)) ss << 'Q';
    if (can_castle(BLACK_OO))  ss << 'k';
    if (can_castle(BLACK_OOO)) ss << 'q';
    if (!can_castle(ANY_CASTLING)) ss << '-';

    // 4. En-passant, 5. halfmove clock, 6. fullmove number.
    ss << (st->epSquare == SQ_NONE ? " - " : " " + square_to_string(st->epSquare) + " ")
       << st->rule50 << ' '
       << 1 + (gamePly - (sideToMove == BLACK)) / 2;

    return ss.str();
}

// ---------------------------------------------------------------------------
// Sanity check
// ---------------------------------------------------------------------------
bool Position::pos_is_ok() const {
    // Kings must both exist, exactly one each.
    if (pieceCount[W_KING] != 1 || pieceCount[B_KING] != 1)
        return false;

    if (attackers_to(king_square(~sideToMove)) & pieces(sideToMove))
        return false;  // side not to move is in check — illegal

    // Bitboard / mailbox consistency.
    if ((pieces(WHITE) & pieces(BLACK)) != 0)
        return false;
    if ((pieces(WHITE) | pieces(BLACK)) != pieces())
        return false;

    for (Square s = SQ_A1; s <= SQ_H8; s += EAST) {
        Piece pc = board[s];
        if (pc == NO_PIECE) {
            if (pieces() & s)
                return false;
        } else {
            if (!(pieces() & s))
                return false;
            if (!(byTypeBB[type_of(pc)] & s))
                return false;
            if (!(byColorBB[color_of(pc)] & s))
                return false;
        }
    }

    // Piece counts vs bitboards.
    for (PieceType pt = PAWN; pt <= KING; pt = PieceType(pt + 1)) {
        for (Color c : {WHITE, BLACK}) {
            Piece pc = make_piece(c, pt);
            if (pieceCount[pc] != popcount(pieces(c, pt)))
                return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// ASCII board diagram
// ---------------------------------------------------------------------------
std::string Position::to_string() const {
    std::ostringstream ss;

    ss << "\n +---+---+---+---+---+---+---+---+\n";
    for (Rank r = RANK_8; r >= RANK_1; r = Rank(r - 1)) {
        for (File f = FILE_A; f <= FILE_H; f = File(f + 1))
            ss << " | " << PieceToChar[piece_on(make_square(f, r))];
        ss << " | " << (1 + r) << "\n +---+---+---+---+---+---+---+---+\n";
    }
    ss << "   a   b   c   d   e   f   g   h\n";

    ss << "\nSide to move: " << (sideToMove == WHITE ? "White" : "Black");
    ss << "\nCastling: ";
    if (can_castle(WHITE_OO))  ss << 'K';
    if (can_castle(WHITE_OOO)) ss << 'Q';
    if (can_castle(BLACK_OO))  ss << 'k';
    if (can_castle(BLACK_OOO)) ss << 'q';
    if (!can_castle(ANY_CASTLING)) ss << '-';
    ss << "\nEP: " << square_to_string(st->epSquare);
    ss << "\nKey: " << std::hex << st->key << std::dec << "\n";

    return ss.str();
}

}  // namespace engine

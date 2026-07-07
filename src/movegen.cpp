// movegen.cpp — Legal / pseudo-legal move generation.
//
// Structure follows the Stockfish idiom: a color+gentype templated set of
// helpers append pseudo-legal moves to a raw buffer; generate<LEGAL> filters
// them through Position::legal. The CAPTURES / QUIETS split partitions the
// non-evasion move set with no overlaps and no omissions, and EVASIONS emits
// only check-resolving moves (required because Position::legal only checks that
// a move does not expose the king, not that it resolves an existing check).

#include "movegen.hpp"

#include "attacks.hpp"
#include "bitboard.hpp"
#include "position.hpp"
#include "types.hpp"

namespace engine {

namespace {

// Emit the four promotion moves for a pawn moving from->to, splitting by
// GenType so each promotion appears exactly once across CAPTURES/QUIETS.
//   - CAPTURES / EVASIONS / NON_EVASIONS : queen promotion (and, for a full
//     set, all promotion types are needed — queen goes with captures-like sets)
//   - QUIETS : rook, bishop, knight under-promotions
// To keep the partition simple and correct we emit:
//   CAPTURES:  QUEEN + KNIGHT + BISHOP + ROOK   -> NO. See below.
//
// The invariant that matters is: NON_EVASIONS = CAPTURES ∪ QUIETS with every
// promotion piece appearing exactly once. We achieve that by:
//   CAPTURES : QUEEN only
//   QUIETS   : KNIGHT, BISHOP, ROOK
// and NON_EVASIONS / EVASIONS emit all four. Because NON_EVASIONS is built by
// running the capture pass then the quiet pass, the union is complete.
template <GenType Type, Direction D, bool Enemy>
ExtMove* make_promotions(ExtMove* moveList, Square to) {
    constexpr bool All =
        Type == EVASIONS || Type == NON_EVASIONS || Type == LEGAL;

    Square from = to - D;

    if (Type == CAPTURES || All)
        *moveList++ = Move::make<PROMOTION>(from, to, QUEEN);

    if (Type == QUIETS || All) {
        *moveList++ = Move::make<PROMOTION>(from, to, ROOK);
        *moveList++ = Move::make<PROMOTION>(from, to, BISHOP);
        *moveList++ = Move::make<PROMOTION>(from, to, KNIGHT);
    }

    (void)from;
    return moveList;
}

template <Color Us, GenType Type>
ExtMove* generate_pawn_moves(const Position& pos, ExtMove* moveList, Bitboard target) {
    constexpr Color     Them     = ~Us;
    constexpr Bitboard  TRank7BB = (Us == WHITE ? Rank7BB : Rank2BB);
    constexpr Bitboard  TRank3BB = (Us == WHITE ? Rank3BB : Rank6BB);
    constexpr Direction Up       = (Us == WHITE ? NORTH : SOUTH);
    constexpr Direction UpRight  = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
    constexpr Direction UpLeft   = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

    const Bitboard emptySquares = ~pos.pieces();
    const Bitboard enemies      = (Type == EVASIONS) ? pos.checkers()
                                                     : pos.pieces(Them);

    Bitboard pawnsOn7    = pos.pieces(Us, PAWN) & TRank7BB;
    Bitboard pawnsNotOn7 = pos.pieces(Us, PAWN) & ~TRank7BB;

    // Single and double pushes, no promotions.
    if (Type != CAPTURES) {
        Bitboard b1 = shift<Up>(pawnsNotOn7) & emptySquares;
        Bitboard b2 = shift<Up>(b1 & TRank3BB) & emptySquares;

        if (Type == EVASIONS) {  // consider only blocking squares
            b1 &= target;
            b2 &= target;
        }

        while (b1) {
            Square to = pop_lsb(b1);
            *moveList++ = Move(to - Up, to);
        }
        while (b2) {
            Square to = pop_lsb(b2);
            *moveList++ = Move(to - Up - Up, to);
        }
    }

    // Promotions and under-promotions.
    if (pawnsOn7) {
        Bitboard b1 = shift<UpRight>(pawnsOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsOn7) & enemies;
        Bitboard b3 = shift<Up>(pawnsOn7) & emptySquares;

        if (Type == EVASIONS) b3 &= target;

        while (b1) moveList = make_promotions<Type, UpRight, true>(moveList, pop_lsb(b1));
        while (b2) moveList = make_promotions<Type, UpLeft, true>(moveList, pop_lsb(b2));
        while (b3) moveList = make_promotions<Type, Up, false>(moveList, pop_lsb(b3));
    }

    // Standard and en-passant captures.
    if (Type == CAPTURES || Type == EVASIONS || Type == NON_EVASIONS || Type == LEGAL) {
        Bitboard b1 = shift<UpRight>(pawnsNotOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsNotOn7) & enemies;

        while (b1) {
            Square to = pop_lsb(b1);
            *moveList++ = Move(to - UpRight, to);
        }
        while (b2) {
            Square to = pop_lsb(b2);
            *moveList++ = Move(to - UpLeft, to);
        }

        if (pos.ep_square() != SQ_NONE) {
            Square ep = pos.ep_square();

            // In EVASIONS, en-passant only helps if the captured pawn is the
            // checker (i.e. the pawn just moved and gives check).
            if (Type == EVASIONS && !(target & (ep - Up)))
                ; // captured pawn not the checker -> skip
            else {
                Bitboard b = pawnsNotOn7 & pawn_attacks(Them, ep);
                while (b) {
                    Square from = pop_lsb(b);
                    *moveList++ = Move::make<EN_PASSANT>(from, ep);
                }
            }
        }
    }

    return moveList;
}

template <Color Us, PieceType Pt>
ExtMove* generate_moves(const Position& pos, ExtMove* moveList, Bitboard target) {
    static_assert(Pt != KING && Pt != PAWN, "unsupported piece type");

    Bitboard bb = pos.pieces(Us, Pt);
    while (bb) {
        Square from = pop_lsb(bb);
        Bitboard b = attacks_bb<Pt>(from, pos.pieces()) & target;
        while (b) {
            Square to = pop_lsb(b);
            *moveList++ = Move(from, to);
        }
    }
    return moveList;
}

// Non-templated slider dispatch avoided; use explicit Pt for the leaper knight.
template <Color Us>
ExtMove* generate_knight_moves(const Position& pos, ExtMove* moveList, Bitboard target) {
    Bitboard bb = pos.pieces(Us, KNIGHT);
    while (bb) {
        Square from = pop_lsb(bb);
        Bitboard b = attacks_bb(KNIGHT, from, pos.pieces()) & target;
        while (b) {
            Square to = pop_lsb(b);
            *moveList++ = Move(from, to);
        }
    }
    return moveList;
}

template <Color Us, GenType Type>
ExtMove* generate_all(const Position& pos, ExtMove* moveList) {
    static_assert(Type != LEGAL, "unsupported");
    constexpr bool Checks = Type == QUIET_CHECKS;

    const Square ksq = pos.king_square(Us);
    Bitboard target;

    // Skip generating non-king moves when in double check.
    if (Type != EVASIONS || !more_than_one(pos.checkers())) {
        target = Type == EVASIONS     ? between_bb(ksq, lsb(pos.checkers())) | pos.checkers()
               : Type == NON_EVASIONS ? ~pos.pieces(Us)
               : Type == CAPTURES     ? pos.pieces(~Us)
                                      : ~pos.pieces();  // QUIETS, QUIET_CHECKS

        moveList = generate_pawn_moves<Us, Type>(pos, moveList, target);
        moveList = generate_knight_moves<Us>(pos, moveList, target);
        moveList = generate_moves<Us, BISHOP>(pos, moveList, target);
        moveList = generate_moves<Us, ROOK>(pos, moveList, target);
        moveList = generate_moves<Us, QUEEN>(pos, moveList, target);
    }

    // King moves.
    if (!Checks || pos.blockers_for_king(~Us) & ksq) {
        Bitboard b = attacks_bb(KING, ksq, pos.pieces());
        b &= (Type == EVASIONS)     ? ~pos.pieces(Us)
           : (Type == CAPTURES)     ? pos.pieces(~Us)
           : (Type == NON_EVASIONS) ? ~pos.pieces(Us)
                                    : ~pos.pieces();  // QUIETS, QUIET_CHECKS

        while (b) {
            Square to = pop_lsb(b);
            *moveList++ = Move(ksq, to);
        }

        // Castling (quiet moves only, and never while in check).
        if (Type == QUIETS || Type == NON_EVASIONS) {
            constexpr CastlingRights OO  = (Us == WHITE ? WHITE_OO : BLACK_OO);
            constexpr CastlingRights OOO = (Us == WHITE ? WHITE_OOO : BLACK_OOO);
            for (CastlingRights cr : {OO, OOO})
                if (pos.can_castle(cr) && !pos.castling_impeded(cr)) {
                    Square kto = relative_square(
                        Us, (cr & KING_SIDE) ? SQ_G1 : SQ_C1);
                    *moveList++ = Move::make<CASTLING>(ksq, kto);
                }
        }
    }

    return moveList;
}

}  // namespace

template <GenType Type>
ExtMove* generate(const Position& pos, ExtMove* moveList) {
    static_assert(Type != LEGAL, "LEGAL handled separately");

    const Color us = pos.side_to_move();
    return us == WHITE ? generate_all<WHITE, Type>(pos, moveList)
                       : generate_all<BLACK, Type>(pos, moveList);
}

// Explicit instantiations for the non-LEGAL gen types.
template ExtMove* generate<CAPTURES>(const Position&, ExtMove*);
template ExtMove* generate<QUIETS>(const Position&, ExtMove*);
template ExtMove* generate<QUIET_CHECKS>(const Position&, ExtMove*);
template ExtMove* generate<EVASIONS>(const Position&, ExtMove*);
template ExtMove* generate<NON_EVASIONS>(const Position&, ExtMove*);

// LEGAL — generate the appropriate pseudo-legal set and filter through legal().
template <>
ExtMove* generate<LEGAL>(const Position& pos, ExtMove* moveList) {
    const Color    us       = pos.side_to_move();
    const Bitboard pinned   = pos.blockers_for_king(us) & pos.pieces(us);
    const Square   ksq      = pos.king_square(us);
    ExtMove*       cur      = moveList;

    moveList = pos.checkers() ? generate<EVASIONS>(pos, moveList)
                              : generate<NON_EVASIONS>(pos, moveList);

    while (cur != moveList) {
        Move m = cur->move;
        // Only moves that could leave the king in check need the full legality
        // test; the rest are already legal. (This is an optimization; testing
        // every move would also be correct.)
        if (((pinned & m.from_sq()) || m.from_sq() == ksq ||
             m.type_of() == EN_PASSANT) &&
            !pos.legal(m))
            *cur = *(--moveList);
        else
            ++cur;
    }

    return moveList;
}

}  // namespace engine

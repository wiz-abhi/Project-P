// nnue.cpp — Classic HalfKP / SFNNv1 NNUE (256x2-32-32-1), full-refresh, scalar.
//
// Architecture (Stockfish "nn-…​.nnue" SFNNv1, HalfKP feature set):
//   Feature transformer : 41024 -> 256  (per-perspective accumulator)
//   Layer 0 (affine)    : 512 (=256*2) -> 32   + ClippedReLU
//   Layer 1 (affine)    : 32 -> 32              + ClippedReLU
//   Output (affine)     : 32 -> 1
//
// evaluate() rebuilds both accumulators from scratch (full refresh) each call,
// so it is a pure function of the Position and needs no per-thread state.

#include "nnue.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <new>
#include <vector>

#include "bitboard.hpp"
#include "position.hpp"
#include "types.hpp"

// Restrict this translation unit to SSE (<=16-byte vector moves). Under
// `-march=native` on Windows/MinGW the compiler otherwise emits 32-byte-aligned
// AVX moves (vmovdqa/vmovaps) on stack spill slots, but the Windows x64 ABI only
// guarantees 16-byte stack alignment, so those accesses fault intermittently
// (0xC0000005) depending on the launching process's stack alignment. Any 16-byte
// SSE aligned move is always satisfied by the ABI's 16-byte guarantee (and the
// entry points additionally carry force_align_arg_pointer), so this removes the
// hazard deterministically. This is the scalar reference evaluator; the rest of
// the engine keeps full AVX, and the eval is not the search bottleneck.
// Windows-only: MinGW's 16-byte stack alignment can fault on AVX stack spills
// (0xC0000005). Linux/glibc realigns correctly, and there the command-line
// -march=native target must NOT be overridden here — a no-avx TU clashes with the
// fortified always_inline memcpy (built for the native target), failing the build.
#if defined(_WIN32) && defined(__GNUC__)
#pragma GCC target("no-avx,no-avx2")
#endif

namespace engine {
namespace NNUE {

namespace {

// ---------------------------------------------------------------------------
// Architecture constants
// ---------------------------------------------------------------------------
constexpr int      kHalfDimensions = 256;
constexpr int      kInputDimensions = 41024;
constexpr uint32_t kExpectedVersion = 0x7AF32F16u;

constexpr int L0_IN  = kHalfDimensions * 2;  // 512
constexpr int L0_OUT = 32;
constexpr int L1_IN  = 32;
constexpr int L1_OUT = 32;
constexpr int L2_IN  = 32;
constexpr int L2_OUT = 1;

constexpr int WEIGHT_SCALE_BITS = 6;
// Final output divisor. The SFNNv1 reference constant is 16, but this particular
// net's raw output range is ~2x hotter than the reference nets the search was
// tuned against; dividing by 16 left NNUE evals ~2-3x larger than the
// hand-crafted eval, which wrecked search interaction (aspiration windows,
// futility/delta pruning and mate thresholds all assume ~HCE-scale scores) and
// made NNUE strictly weaker despite correct feature indexing. Scaling the output
// down to HCE-comparable magnitudes is what makes NNUE clearly beat HCE.
constexpr int FV_SCALE          = 32;

// ---------------------------------------------------------------------------
// HalfKP piece-square index bases.  PS_END == 641 == 41024 / 64.
// ---------------------------------------------------------------------------
enum {
    PS_W_PAWN   = 1,
    PS_B_PAWN   = 1 * 64 + 1,
    PS_W_KNIGHT = 2 * 64 + 1,
    PS_B_KNIGHT = 3 * 64 + 1,
    PS_W_BISHOP = 4 * 64 + 1,
    PS_B_BISHOP = 5 * 64 + 1,
    PS_W_ROOK   = 6 * 64 + 1,
    PS_B_ROOK   = 7 * 64 + 1,
    PS_W_QUEEN  = 8 * 64 + 1,
    PS_B_QUEEN  = 9 * 64 + 1,
    PS_END      = 10 * 64 + 1   // 641
};

// PieceSquareIndex[perspective][piece] : maps a Piece (16-valued enum) to its
// PS_* base for the given perspective.  From the perspective's point of view,
// its own pieces use the PS_W_* bases and the enemy's use PS_B_* (i.e. the two
// tables are the W/B swap of each other).  Kings (index 6/14) are 0 (unused —
// kings are never features).
int PieceSquareIndex[COLOR_NB][PIECE_NB];

void init_piece_square_index() {
    for (int c = 0; c < COLOR_NB; ++c)
        for (int p = 0; p < PIECE_NB; ++p)
            PieceSquareIndex[c][p] = 0;

    // WHITE perspective: white pieces -> PS_W_*, black pieces -> PS_B_*.
    PieceSquareIndex[WHITE][W_PAWN]   = PS_W_PAWN;
    PieceSquareIndex[WHITE][W_KNIGHT] = PS_W_KNIGHT;
    PieceSquareIndex[WHITE][W_BISHOP] = PS_W_BISHOP;
    PieceSquareIndex[WHITE][W_ROOK]   = PS_W_ROOK;
    PieceSquareIndex[WHITE][W_QUEEN]  = PS_W_QUEEN;
    PieceSquareIndex[WHITE][B_PAWN]   = PS_B_PAWN;
    PieceSquareIndex[WHITE][B_KNIGHT] = PS_B_KNIGHT;
    PieceSquareIndex[WHITE][B_BISHOP] = PS_B_BISHOP;
    PieceSquareIndex[WHITE][B_ROOK]   = PS_B_ROOK;
    PieceSquareIndex[WHITE][B_QUEEN]  = PS_B_QUEEN;

    // BLACK perspective: black pieces -> PS_W_* (own), white pieces -> PS_B_*.
    PieceSquareIndex[BLACK][B_PAWN]   = PS_W_PAWN;
    PieceSquareIndex[BLACK][B_KNIGHT] = PS_W_KNIGHT;
    PieceSquareIndex[BLACK][B_BISHOP] = PS_W_BISHOP;
    PieceSquareIndex[BLACK][B_ROOK]   = PS_W_ROOK;
    PieceSquareIndex[BLACK][B_QUEEN]  = PS_W_QUEEN;
    PieceSquareIndex[BLACK][W_PAWN]   = PS_B_PAWN;
    PieceSquareIndex[BLACK][W_KNIGHT] = PS_B_KNIGHT;
    PieceSquareIndex[BLACK][W_BISHOP] = PS_B_BISHOP;
    PieceSquareIndex[BLACK][W_ROOK]   = PS_B_ROOK;
    PieceSquareIndex[BLACK][W_QUEEN]  = PS_B_QUEEN;
}

// ---------------------------------------------------------------------------
// Network weights (static, loaded once).
// ---------------------------------------------------------------------------
// Feature transformer. All hot buffers are 64-byte aligned so the auto-
// vectorizer (AVX2 under -march=native) may use aligned loads safely.
alignas(64) int16_t ft_bias[kHalfDimensions];
// Feature-major: ft_weight[f * kHalfDimensions + i]. Backed by an over-aligned
// heap allocation (std::vector only guarantees 16-byte alignment, which is not
// enough for 32-byte AVX aligned moves).
int16_t* ft_weight = nullptr;   // size = kInputDimensions * kHalfDimensions

// Layer 0 (512 -> 32), output-major weights.
alignas(64) int32_t l0_bias[L0_OUT];
alignas(64) int8_t  l0_weight[L0_OUT * L0_IN];

// Layer 1 (32 -> 32), output-major.
alignas(64) int32_t l1_bias[L1_OUT];
alignas(64) int8_t  l1_weight[L1_OUT * L1_IN];

// Output (32 -> 1), output-major.
alignas(64) int32_t out_bias[L2_OUT];
alignas(64) int8_t  out_weight[L2_OUT * L2_IN];

bool     g_loaded = false;
uint32_t g_version = 0, g_ft_hash = 0, g_net_hash = 0, g_arch_hash = 0;

// ---------------------------------------------------------------------------
// Sequential little-endian reader over an in-memory buffer.
// ---------------------------------------------------------------------------
struct Reader {
    const uint8_t* data;
    size_t         size;
    size_t         pos = 0;
    bool           ok = true;

    bool need(size_t n) {
        if (pos + n > size) { ok = false; return false; }
        return true;
    }
    uint32_t u32() {
        if (!need(4)) return 0;
        uint32_t v;
        std::memcpy(&v, data + pos, 4);
        pos += 4;
        return v;  // host is little-endian (x86-64)
    }
    int16_t i16() {
        if (!need(2)) return 0;
        int16_t v;
        std::memcpy(&v, data + pos, 2);
        pos += 2;
        return v;
    }
    int32_t i32() {
        if (!need(4)) return 0;
        int32_t v;
        std::memcpy(&v, data + pos, 4);
        pos += 4;
        return v;
    }
    int8_t i8() {
        if (!need(1)) return 0;
        return static_cast<int8_t>(data[pos++]);
    }
    void skip(size_t n) {
        if (need(n)) pos += n;
    }
};

inline int clamp127(int x) {
    return x < 0 ? 0 : (x > 127 ? 127 : x);
}

// orient(perspective, sq): 180-degree rotation for Black (classic HalfKP).
inline int orient(Color perspective, Square s) {
    return perspective == WHITE ? int(s) : int(s) ^ 63;
}

// HalfKP feature index for (perspective, that perspective's king square, a
// non-king piece `pc` on square `s`). This is the single source of truth shared
// by the full-refresh path and the incremental path, so the two can never drift.
//   index = PS_END * orient(king) + PieceSquareIndex[perspective][pc] + orient(s)
inline int make_index(Color perspective, Square ksq, Piece pc, Square s) {
    const int koffset = PS_END * orient(perspective, ksq);
    return koffset + PieceSquareIndex[perspective][pc] + orient(perspective, s);
}

}  // namespace

// On Windows/MinGW the x64 ABI only guarantees 16-byte stack alignment, but
// -march=native lets the auto-vectorizer emit 32-byte-aligned AVX moves
// (vmovdqa/vmovaps) on stack spill slots. When the runtime stack is 16- but not
// 32-byte aligned this faults (intermittent 0xC0000005). Forcing the prologue to
// realign the stack pointer on the NNUE entry points makes those AVX accesses
// safe regardless of the caller's stack alignment.
#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
#define NNUE_STACK_ALIGN __attribute__((force_align_arg_pointer))
#else
#define NNUE_STACK_ALIGN
#endif

// ---------------------------------------------------------------------------
// load() — read the whole file, parse sequentially, assert bytes consumed.
// ---------------------------------------------------------------------------
NNUE_STACK_ALIGN bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;

    std::streamsize fsize = f.tellg();
    if (fsize <= 0) return false;
    f.seekg(0, std::ios::beg);

    // Raw malloc buffer (not std::vector) so this TU can carry a `no-avx` target
    // attribute without tripping over always_inline AVX code in std::allocator.
    const size_t bufSize = static_cast<size_t>(fsize);
    uint8_t* raw = static_cast<uint8_t*>(std::malloc(bufSize));
    if (!raw) return false;
    if (!f.read(reinterpret_cast<char*>(raw), fsize)) { std::free(raw); return false; }

    Reader r{raw, bufSize};

    // ---- Header ----
    uint32_t version   = r.u32();
    uint32_t arch_hash = r.u32();
    uint32_t arch_size = r.u32();
    size_t   arch_off  = r.pos;
    r.skip(arch_size);   // description string

    if (version != kExpectedVersion) {
        std::cerr << "NNUE: unexpected version 0x" << std::hex << version
                  << " (expected 0x" << kExpectedVersion << ")" << std::dec
                  << "\n";
        // Continue anyway — we only warn — but many nets will be incompatible.
    }

    // ---- Feature Transformer ----
    size_t   ft_off  = r.pos;
    uint32_t ft_hash = r.u32();
    for (int i = 0; i < kHalfDimensions; ++i)
        ft_bias[i] = r.i16();

    const size_t ftCount = static_cast<size_t>(kInputDimensions) * kHalfDimensions;
    if (ft_weight == nullptr) {
        // 64-byte aligned allocation, allocated once and never freed (lives for
        // the whole process). Size is a multiple of 64 bytes already.
        void* mem = ::operator new(ftCount * sizeof(int16_t),
                                   std::align_val_t(64));
        ft_weight = static_cast<int16_t*>(mem);
    }
    for (size_t i = 0; i < ftCount; ++i)
        ft_weight[i] = r.i16();

    // ---- Network ----
    size_t   net_off  = r.pos;
    uint32_t net_hash = r.u32();

    for (int o = 0; o < L0_OUT; ++o) l0_bias[o] = r.i32();
    for (int i = 0; i < L0_OUT * L0_IN; ++i) l0_weight[i] = r.i8();

    for (int o = 0; o < L1_OUT; ++o) l1_bias[o] = r.i32();
    for (int i = 0; i < L1_OUT * L1_IN; ++i) l1_weight[i] = r.i8();

    for (int o = 0; o < L2_OUT; ++o) out_bias[o] = r.i32();
    for (int i = 0; i < L2_OUT * L2_IN; ++i) out_weight[i] = r.i8();

    if (!r.ok) {
        std::cerr << "NNUE: file too small / truncated while parsing "
                  << path << "\n";
        g_loaded = false;
        std::free(raw);
        return false;
    }

    size_t consumed = r.pos;
    if (consumed != bufSize) {
        std::cerr << "NNUE: LAYOUT ERROR bytes consumed = " << consumed
                  << " / " << bufSize << " (mismatch)\n"
                  << "  arch@"  << arch_off << " size " << arch_size
                  << "  ft@"    << ft_off
                  << "  net@"   << net_off << "\n";
        g_loaded = false;
        assert(consumed == bufSize && "NNUE layout mismatch");
        std::free(raw);
        return false;
    }

    init_piece_square_index();

    g_version  = version;
    g_arch_hash = arch_hash;
    g_ft_hash  = ft_hash;
    g_net_hash = net_hash;
    g_loaded   = true;

    std::cout << "NNUE loaded: " << path
              << ", bytes consumed = " << consumed << " / " << bufSize
              << "\n";
    std::printf("NNUE version=0x%08X ft_hash=0x%08X net_hash=0x%08X arch_hash=0x%08X\n",
                version, ft_hash, net_hash, arch_hash);
    std::fflush(stdout);

    std::free(raw);
    return true;
}

bool loaded() { return g_loaded; }

namespace {

// ---------------------------------------------------------------------------
// network_forward() — the shared 512->32->32->1 inference over two clipped
// per-perspective accumulators (side-to-move first). Both the full-refresh path
// and the incremental path funnel through here, so their arithmetic is byte-for-
// byte identical. `acc_stm` / `acc_nstm` are the raw int accumulator rows for
// the side to move and the other side respectively.
// ---------------------------------------------------------------------------
template <typename Acc>
inline Value network_forward(const Acc* acc_stm, const Acc* acc_nstm) {
    // Clipped 512-wide input: side-to-move perspective first.
    alignas(64) uint8_t input[L0_IN];
    for (int i = 0; i < kHalfDimensions; ++i)
        input[i] = static_cast<uint8_t>(clamp127(int(acc_stm[i])));
    for (int i = 0; i < kHalfDimensions; ++i)
        input[kHalfDimensions + i] = static_cast<uint8_t>(clamp127(int(acc_nstm[i])));

    // Layer 0 (512 -> 32) + ClippedReLU.
    alignas(64) uint8_t h0[L0_OUT];
    for (int o = 0; o < L0_OUT; ++o) {
        int32_t s = l0_bias[o];
        const int8_t* w = &l0_weight[static_cast<size_t>(o) * L0_IN];
        for (int i = 0; i < L0_IN; ++i)
            s += int(input[i]) * int(w[i]);
        h0[o] = static_cast<uint8_t>(clamp127(s >> WEIGHT_SCALE_BITS));
    }

    // Layer 1 (32 -> 32) + ClippedReLU.
    alignas(64) uint8_t h1[L1_OUT];
    for (int o = 0; o < L1_OUT; ++o) {
        int32_t s = l1_bias[o];
        const int8_t* w = &l1_weight[static_cast<size_t>(o) * L1_IN];
        for (int i = 0; i < L1_IN; ++i)
            s += int(h0[i]) * int(w[i]);
        h1[o] = static_cast<uint8_t>(clamp127(s >> WEIGHT_SCALE_BITS));
    }

    // Output (32 -> 1).
    int32_t s = out_bias[0];
    for (int i = 0; i < L2_IN; ++i)
        s += int(h1[i]) * int(out_weight[i]);

    // stm-relative centipawns.
    return Value(s / FV_SCALE);
}

}  // namespace

// ---------------------------------------------------------------------------
// evaluate() — full refresh, scalar inference.
// ---------------------------------------------------------------------------
NNUE_STACK_ALIGN Value evaluate(const Position& pos) {
    // 1) Build both accumulators from scratch.
    //    acc[perspective][i] = ft_bias[i] + Σ_active_f ft_weight[f*256 + i]
    alignas(64) int32_t acc[COLOR_NB][kHalfDimensions];
    for (int p = 0; p < COLOR_NB; ++p)
        for (int i = 0; i < kHalfDimensions; ++i)
            acc[p][i] = ft_bias[i];

    for (int pc = 0; pc < COLOR_NB; ++pc) {
        const Color perspective = Color(pc);
        const Square ksq = pos.king_square(perspective);

        // Iterate every non-king piece on the board.
        Bitboard occ = pos.pieces() ^ pos.pieces(KING);
        while (occ) {
            Square s   = pop_lsb(occ);
            Piece  pic = pos.piece_on(s);
            int    findex = make_index(perspective, ksq, pic, s);

            const int16_t* w = &ft_weight[static_cast<size_t>(findex) * kHalfDimensions];
            for (int i = 0; i < kHalfDimensions; ++i)
                acc[perspective][i] += w[i];
        }
    }

    const Color stm = pos.side_to_move();
    return network_forward<int32_t>(acc[stm], acc[~stm]);
}

// ---------------------------------------------------------------------------
// refresh_perspective() — recompute one perspective's accumulator from scratch.
// ---------------------------------------------------------------------------
NNUE_STACK_ALIGN void refresh_perspective(Accumulator& a, const Position& pos, Color c) {
    // int32 working sum, then narrowed back to int16 (values fit comfortably —
    // this matches the full-refresh path, which also sums into int32).
    alignas(64) int32_t sum[kHalfDimensions];
    for (int i = 0; i < kHalfDimensions; ++i)
        sum[i] = ft_bias[i];

    const Square ksq = pos.king_square(c);
    Bitboard occ = pos.pieces() ^ pos.pieces(KING);
    while (occ) {
        Square s   = pop_lsb(occ);
        Piece  pic = pos.piece_on(s);
        int    findex = make_index(c, ksq, pic, s);
        const int16_t* w = &ft_weight[static_cast<size_t>(findex) * kHalfDimensions];
        for (int i = 0; i < kHalfDimensions; ++i)
            sum[i] += w[i];
    }

    for (int i = 0; i < kHalfDimensions; ++i)
        a.acc[c][i] = static_cast<int16_t>(sum[i]);
    a.computed[c] = true;
}

void refresh(Accumulator& a, const Position& pos) {
    refresh_perspective(a, pos, WHITE);
    refresh_perspective(a, pos, BLACK);
}

// ---------------------------------------------------------------------------
// evaluate(Accumulator, stm) — inference over an already-built accumulator.
// ---------------------------------------------------------------------------
NNUE_STACK_ALIGN Value evaluate(const Accumulator& a, Color stm) {
    return network_forward<int16_t>(a.acc[stm], a.acc[~stm]);
}

// ---------------------------------------------------------------------------
// update() — derive `child` from `parent` by applying one move.
//
// MUST be called on the PRE-MOVE position `before` (its board still reflects the
// pre-move placement), typically immediately before pos.do_move(m). All feature
// squares/pieces are read from `before`.
//
// For each perspective we add the weight rows of newly-appearing features and
// subtract the rows of departing features. When a perspective's OWN king moves
// (normal king move OR castling) that perspective is instead refreshed from
// scratch, because every one of its features is keyed on its own king square;
// the king's destination square is derived here (no post-move position needed).
// ---------------------------------------------------------------------------
NNUE_STACK_ALIGN void update(Accumulator& child, const Accumulator& parent,
                             const Position& before, Move m) {
    const Color us    = before.side_to_move();
    const Color them  = ~us;
    const Square from = m.from_sq();
    const Square to   = m.to_sq();
    const MoveType mt = m.type_of();
    const Piece moved = before.piece_on(from);
    const bool  kingMove = (type_of(moved) == KING);  // includes castling (king from/to)

    // King's destination (for the moving side's refresh). For castling the king
    // lands on the relative g1/c1 square (this engine encodes `to` == king dest,
    // and for standard chess that already equals g1/c1, but derive it explicitly
    // to be safe against the raw encoding).
    const Square kto =
        (mt == CASTLING) ? relative_square(us, (to > from) ? SQ_G1 : SQ_C1) : to;

    // Captured piece + square (mirrors position.cpp exactly).
    Piece capturedPc = NO_PIECE;
    Square capsq = to;
    if (mt == EN_PASSANT) {
        capturedPc = make_piece(them, PAWN);
        capsq = to - (us == WHITE ? NORTH : SOUTH);
    } else if (mt != CASTLING) {
        Piece onTo = before.piece_on(to);
        if (onTo != NO_PIECE) {          // normal capture (own-piece capture never generated)
            capturedPc = onTo;
            capsq = to;
        }
    }

    // The piece that ends up on `to` (promotion changes the piece type there).
    const Piece placed =
        (mt == PROMOTION) ? make_piece(us, m.promotion_type()) : moved;

    // Process each perspective independently.
    for (int pc = 0; pc < COLOR_NB; ++pc) {
        const Color P = Color(pc);

        // If side P's own king moved (normal king move OR castling), P's whole
        // accumulator must be rebuilt against the new king square `kto`.
        if (kingMove && P == us) {
            // Rebuild from `before`'s board but with P's king relocated to `kto`
            // (and, for castling, the rook relocated too — but that is P's own
            // rook, whose feature is included in the loop below with the new
            // king). Simplest correct approach: enumerate `before`'s pieces,
            // apply the move's piece relocations on the fly.
            alignas(64) int32_t sum[kHalfDimensions];
            for (int i = 0; i < kHalfDimensions; ++i)
                sum[i] = ft_bias[i];

            const Square ksqNew = kto;  // P == us, king moved to kto

            // Castling rook relocation (own rook of the moving side).
            Square rfrom = SQ_NONE, rto = SQ_NONE;
            if (mt == CASTLING) {
                const bool kingSide = to > from;
                rfrom = kingSide ? relative_square(us, SQ_H1)
                                 : relative_square(us, SQ_A1);
                rto   = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
            }

            Bitboard occ = before.pieces() ^ before.pieces(KING);
            while (occ) {
                Square s   = pop_lsb(occ);
                Piece  pic = before.piece_on(s);

                // Skip a captured piece that will vanish (king can capture).
                if (capturedPc != NO_PIECE && s == capsq)
                    continue;

                // Relocate the castling rook from rfrom to rto.
                if (mt == CASTLING && s == rfrom)
                    s = rto;

                int findex = make_index(P, ksqNew, pic, s);
                const int16_t* w =
                    &ft_weight[static_cast<size_t>(findex) * kHalfDimensions];
                for (int i = 0; i < kHalfDimensions; ++i)
                    sum[i] += w[i];
            }

            for (int i = 0; i < kHalfDimensions; ++i)
                child.acc[P][i] = static_cast<int16_t>(sum[i]);
            child.computed[P] = true;
            continue;
        }

        // Otherwise start from the parent and apply feature deltas. King squares
        // for perspective P are unchanged here, so read P's king from `before`.
        const Square ksq = before.king_square(P);

        // Collect up to 3 subtractions and up to 2 additions.
        int subIdx[3];
        int addIdx[2];
        int nSub = 0, nAdd = 0;

        if (mt == CASTLING) {
            // The castling side is `us`; since P != us here (us was handled by
            // the king-move refresh above), P is the OTHER side and sees only the
            // rook relocation as a feature change (the enemy king is not a
            // feature). Derive rook from/to from the king from/to.
            const bool kingSide = to > from;
            const Square rfrom = kingSide ? relative_square(us, SQ_H1)
                                          : relative_square(us, SQ_A1);
            const Square rto   = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
            const Piece rook   = make_piece(us, ROOK);
            subIdx[nSub++] = make_index(P, ksq, rook, rfrom);
            addIdx[nAdd++] = make_index(P, ksq, rook, rto);
        } else {
            // Moved piece relocation. KINGS ARE NOT FEATURES, so if the moving
            // piece is a king (a king move by `us`, seen here from the OTHER
            // perspective P), the relocation contributes no feature delta — only
            // a capture (handled below) matters to P. For non-king movers, add
            // the moved piece's from/to delta (promotion swaps the piece kind on
            // arrival).
            if (!kingMove) {
                subIdx[nSub++] = make_index(P, ksq, moved, from);
                addIdx[nAdd++] = make_index(P, ksq, placed, to);
            }

            // Captured piece disappears from capsq (applies to king captures too).
            if (capturedPc != NO_PIECE)
                subIdx[nSub++] = make_index(P, ksq, capturedPc, capsq);
        }

        // Apply: child = parent - Σ sub + Σ add, per neuron.
        const int16_t* wsub[3];
        const int16_t* wadd[2];
        for (int k = 0; k < nSub; ++k)
            wsub[k] = &ft_weight[static_cast<size_t>(subIdx[k]) * kHalfDimensions];
        for (int k = 0; k < nAdd; ++k)
            wadd[k] = &ft_weight[static_cast<size_t>(addIdx[k]) * kHalfDimensions];

        const int16_t* src = parent.acc[P];
        int16_t*       dst = child.acc[P];
        for (int i = 0; i < kHalfDimensions; ++i) {
            int v = int(src[i]);
            for (int k = 0; k < nSub; ++k) v -= int(wsub[k][i]);
            for (int k = 0; k < nAdd; ++k) v += int(wadd[k][i]);
            dst[i] = static_cast<int16_t>(v);
        }
        child.computed[P] = true;
    }
}

}  // namespace NNUE
}  // namespace engine

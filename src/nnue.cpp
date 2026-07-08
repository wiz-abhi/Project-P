// nnue.cpp — NNUE evaluation, two architectures behind one API:
//
//   * HalfKP / SFNNv1  (256x2-32-32-1),  version 0x7AF32F16 — fallback.
//   * HalfKAv2_hm / SFNNv4 (SF15-era),   version 0x7AF32F20 — modern.
//
// SFNNv4 layout (verified against the Stockfish sf_15 sources and byte-exact
// against nn-ad9b42354671.nnue, 47,001,499 bytes):
//   header:  u32 version, u32 hash, u32 desc_size, desc bytes
//   FT:      u32 hash, i16 bias[1024], i16 weight[22528][1024],
//            i32 psqt[22528][8]
//   8 layer stacks, each: u32 hash,
//            fc_0: i32 bias[16],  i8 weight[16][1024]
//            fc_1: i32 bias[32],  i8 weight[32][32]   (input 15 padded to 32)
//            fc_2: i32 bias[1],   i8 weight[1][32]
//
// Inference (SFNNv4):
//   input[p*512+j]   = clip(acc[p][j])*clip(acc[p][j+512])/128   (u8, stm first)
//   fc_0 (1024->16), ac_0 = clip(fc_0 >> 6)
//   fc_1 (15->32),   ac_1 = clip(fc_1 >> 6)
//   fc_2 (32->1)
//   fwdOut     = fc_0_raw[15] * 600*OutputScale / (127*2^6)      (skip)
//   positional = fc_2 + fwdOut
//   psqt       = (psqtAcc[stm][bucket] - psqtAcc[~stm][bucket]) / 2
//   bucket     = (popcount(all pieces) - 1) / 4
//   value      = (psqt + positional) / FV_SCALE
//
// evaluate(pos) rebuilds both accumulators from scratch (full refresh) each
// call; the incremental Accumulator API mirrors it exactly (shared indexing).

#include "nnue.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <new>

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
// Common
// ---------------------------------------------------------------------------
enum Arch { ARCH_NONE = 0, ARCH_HALFKP, ARCH_HALFKAV2_HM };
Arch g_arch = ARCH_NONE;

constexpr uint32_t kVersionHalfKP = 0x7AF32F16u;  // SFNNv1
constexpr uint32_t kVersionSFNNv4 = 0x7AF32F20u;  // SFNNv2..v4 share this
constexpr int WEIGHT_SCALE_BITS = 6;

// Sequential little-endian reader over an in-memory buffer.
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
    // Bulk copy of little-endian scalars (host is LE, so a straight memcpy).
    bool bulk(void* dst, size_t bytes) {
        if (!need(bytes)) return false;
        std::memcpy(dst, data + pos, bytes);
        pos += bytes;
        return true;
    }
    void skip(size_t n) {
        if (need(n)) pos += n;
    }
};

inline int clamp127(int x) {
    return x < 0 ? 0 : (x > 127 ? 127 : x);
}

// 64-byte-aligned once-only heap allocation (never freed; lives for the whole
// process). Raw operator new so this TU can carry a `no-avx` target attribute
// without tripping over always_inline AVX code in std::allocator.
template <typename T>
T* aligned_alloc_once(T*& slot, size_t count) {
    if (slot == nullptr)
        slot = static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t(64)));
    return slot;
}

// ===========================================================================
// HalfKP / SFNNv1 (fallback architecture)
// ===========================================================================
constexpr int      kHalfDimensions  = 256;
constexpr int      kInputDimensions = 41024;

constexpr int L0_IN  = kHalfDimensions * 2;  // 512
constexpr int L0_OUT = 32;
constexpr int L1_IN  = 32;
constexpr int L1_OUT = 32;
constexpr int L2_IN  = 32;
constexpr int L2_OUT = 1;

// Final output divisor. The SFNNv1 reference constant is 16, but this particular
// net's raw output range is ~2x hotter than the reference nets the search was
// tuned against; dividing by 16 left NNUE evals ~2-3x larger than the
// hand-crafted eval, which wrecked search interaction (aspiration windows,
// futility/delta pruning and mate thresholds all assume ~HCE-scale scores).
constexpr int FV_SCALE = 32;

// HalfKP piece-square index bases.  PS_END == 641 == 41024 / 64.
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
// PS_* base for the given perspective. Kings are 0 (never features in HalfKP).
int PieceSquareIndex[COLOR_NB][PIECE_NB];

void init_piece_square_index() {
    for (int c = 0; c < COLOR_NB; ++c)
        for (int p = 0; p < PIECE_NB; ++p)
            PieceSquareIndex[c][p] = 0;

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

// Feature transformer. All hot buffers are 64-byte aligned so the auto-
// vectorizer (AVX2 under -march=native on Linux) may use aligned loads safely.
alignas(64) int16_t ft_bias[kHalfDimensions];
// Feature-major: ft_weight[f * kHalfDimensions + i].
int16_t* ft_weight = nullptr;   // size = kInputDimensions * kHalfDimensions

alignas(64) int32_t l0_bias[L0_OUT];
alignas(64) int8_t  l0_weight[L0_OUT * L0_IN];
alignas(64) int32_t l1_bias[L1_OUT];
alignas(64) int8_t  l1_weight[L1_OUT * L1_IN];
alignas(64) int32_t out_bias[L2_OUT];
alignas(64) int8_t  out_weight[L2_OUT * L2_IN];

bool g_loaded = false;

// orient(perspective, sq): 180-degree rotation for Black (classic HalfKP).
inline int orient_hk(Color perspective, Square s) {
    return perspective == WHITE ? int(s) : int(s) ^ 63;
}

// HalfKP feature index (single source of truth for full-refresh + incremental).
inline int make_index_hk(Color perspective, Square ksq, Piece pc, Square s) {
    const int koffset = PS_END * orient_hk(perspective, ksq);
    return koffset + PieceSquareIndex[perspective][pc] + orient_hk(perspective, s);
}

// ===========================================================================
// HalfKAv2_hm / SFNNv4 (modern architecture)
// ===========================================================================
constexpr int V4_HALF   = 1024;    // FT accumulator width per perspective
constexpr int V4_INPUTS = 22528;   // 64 sq * 704 planes / 2 (horizontal mirror)
constexpr int V4_PSQT   = 8;       // PSQT buckets
constexpr int V4_STACKS = 8;       // bucketed layer stacks
constexpr int V4_PS_NB  = 11 * 64; // 704

constexpr int FC0_IN      = 1024;  // = V4_HALF: 512 pairwise products per side
constexpr int FC0_OUT     = 16;    // FC_0_OUTPUTS + 1 skip neuron
constexpr int FC_0_OUTPUTS = 15;   // fc_0 neurons that feed fc_1 (index 15 = skip)
constexpr int FC1_IN  = 30;        // = FC_0_OUTPUTS * 2: [sqr(0..14), clip(0..14)]
constexpr int FC1_PAD = 32;        // padded input dims as stored in the file
constexpr int FC1_OUT = 32;
constexpr int FC2_IN  = 32;

// Final scaling. SF's raw layer output (materialist + positional) is first
// divided by OutputScale=16 to get an internal Value, then REPORTED in
// centipawns via UCI::to_cp: cp = value * 100 / NormalizeToPawnValue (=361).
// So the centipawn number SF prints for "NNUE evaluation" (what nnue_cmp.py
// parses) equals raw * 100 / (16 * 361). We reproduce that exactly here, which
// also lands the eval on a ~100cp/pawn HCE-comparable scale for search.
constexpr int V4_OUTPUT_SCALE = 16;   // SF OutputScale (also used in skip scaling)
constexpr int V4_NORMALIZE    = 361;  // SF UCI::NormalizeToPawnValue (sf 15.1)

// Expected FT hash: HalfKAv2_hm HashValue 0x7f234cb8 ^ (OutputDimensions(1024)*2)
constexpr uint32_t kFtHashV4 = 0x7f234cb8u ^ (V4_HALF * 2);

// King buckets (SF half_ka_v2_hm.h): indexed by the ORIENTED own-king square
// (after vertical flip for black + horizontal mirror onto files e-h). -1 for
// files a-d, which can never occur after mirroring.
constexpr int V4_KingBuckets[64] = {
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10,  9,  8,
    -1, -1, -1, -1,  7,  6,  5,  4,
    -1, -1, -1, -1,  3,  2,  1,  0
};

// PieceSquareIndex (SF half_ka_v2_hm.h, 0-based; BOTH kings use PS_KING).
// This engine's Piece enum matches Stockfish's layout exactly.
constexpr int V4_PS_W_PAWN = 0,        V4_PS_B_PAWN = 1 * 64,
              V4_PS_W_KNIGHT = 2 * 64, V4_PS_B_KNIGHT = 3 * 64,
              V4_PS_W_BISHOP = 4 * 64, V4_PS_B_BISHOP = 5 * 64,
              V4_PS_W_ROOK = 6 * 64,   V4_PS_B_ROOK = 7 * 64,
              V4_PS_W_QUEEN = 8 * 64,  V4_PS_B_QUEEN = 9 * 64,
              V4_PS_KING = 10 * 64;

constexpr int V4_PieceSquareIndex[COLOR_NB][PIECE_NB] = {
    // WHITE perspective: own (white) pieces -> W planes, black -> B planes.
    { 0, V4_PS_W_PAWN, V4_PS_W_KNIGHT, V4_PS_W_BISHOP, V4_PS_W_ROOK, V4_PS_W_QUEEN, V4_PS_KING, 0,
      0, V4_PS_B_PAWN, V4_PS_B_KNIGHT, V4_PS_B_BISHOP, V4_PS_B_ROOK, V4_PS_B_QUEEN, V4_PS_KING, 0 },
    // BLACK perspective: own (black) pieces -> W planes, white -> B planes.
    { 0, V4_PS_B_PAWN, V4_PS_B_KNIGHT, V4_PS_B_BISHOP, V4_PS_B_ROOK, V4_PS_B_QUEEN, V4_PS_KING, 0,
      0, V4_PS_W_PAWN, V4_PS_W_KNIGHT, V4_PS_W_BISHOP, V4_PS_W_ROOK, V4_PS_W_QUEEN, V4_PS_KING, 0 }
};

// One bucketed layer stack; weights row-major [out][paddedIn] as in the file.
struct V4Stack {
    alignas(64) int32_t b0[FC0_OUT];
    alignas(64) int8_t  w0[FC0_OUT * FC0_IN];
    alignas(64) int32_t b1[FC1_OUT];
    alignas(64) int8_t  w1[FC1_OUT * FC1_PAD];
    alignas(64) int32_t b2[1];
    alignas(64) int8_t  w2[FC2_IN];
};

alignas(64) int16_t v4_ft_bias[V4_HALF];
int16_t* v4_ft_weight   = nullptr;  // [V4_INPUTS * V4_HALF], feature-major
int32_t* v4_psqt_weight = nullptr;  // [V4_INPUTS * V4_PSQT], feature-major
V4Stack  v4_stack[V4_STACKS];

// Orientation (SF): vertical flip (^56) for Black + horizontal mirror (^7)
// whenever THAT PERSPECTIVE's king sits on files a-d. `ksq` is the raw
// (unoriented) king square of the perspective.
inline int orient_v4(Color perspective, Square s, Square ksq) {
    return int(s) ^ (perspective == BLACK ? 56 : 0)
                  ^ (file_of(ksq) < FILE_E ? 7 : 0);
}

// HalfKAv2_hm feature index — the single source of truth shared by the
// full-refresh and incremental paths.
inline int make_index_v4(Color perspective, Square s, Piece pc, Square ksq) {
    const int o_ksq = orient_v4(perspective, ksq, ksq);
    return orient_v4(perspective, s, ksq)
         + V4_PieceSquareIndex[perspective][pc]
         + V4_PS_NB * V4_KingBuckets[o_ksq];
}

inline int v4_bucket(const Position& pos) {
    return (popcount(pos.pieces()) - 1) / 4;
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

namespace {

// ---------------------------------------------------------------------------
// Parsers. Each takes the Reader positioned right AFTER the header (i.e. at
// the feature-transformer hash) and must consume the file exactly.
// ---------------------------------------------------------------------------
bool parse_halfkp(Reader& r, const char* path, size_t arch_off,
                  uint32_t arch_size) {
    size_t   ft_off  = r.pos;
    uint32_t ft_hash = r.u32();
    for (int i = 0; i < kHalfDimensions; ++i)
        ft_bias[i] = r.i16();

    const size_t ftCount = static_cast<size_t>(kInputDimensions) * kHalfDimensions;
    aligned_alloc_once(ft_weight, ftCount);
    r.bulk(ft_weight, ftCount * sizeof(int16_t));

    size_t   net_off  = r.pos;
    uint32_t net_hash = r.u32();

    for (int o = 0; o < L0_OUT; ++o) l0_bias[o] = r.i32();
    for (int i = 0; i < L0_OUT * L0_IN; ++i) l0_weight[i] = r.i8();
    for (int o = 0; o < L1_OUT; ++o) l1_bias[o] = r.i32();
    for (int i = 0; i < L1_OUT * L1_IN; ++i) l1_weight[i] = r.i8();
    for (int o = 0; o < L2_OUT; ++o) out_bias[o] = r.i32();
    for (int i = 0; i < L2_OUT * L2_IN; ++i) out_weight[i] = r.i8();

    if (!r.ok) {
        std::cerr << "NNUE: file too small / truncated while parsing " << path << "\n";
        return false;
    }
    if (r.pos != r.size) {
        std::cerr << "NNUE: LAYOUT ERROR (HalfKP) bytes consumed = " << r.pos
                  << " / " << r.size << " (mismatch)\n"
                  << "  arch@" << arch_off << " size " << arch_size
                  << "  ft@" << ft_off << "  net@" << net_off << "\n";
        return false;
    }

    init_piece_square_index();
    std::printf("NNUE (HalfKP) ft_hash=0x%08X net_hash=0x%08X\n", ft_hash, net_hash);
    return true;
}

bool parse_v4(Reader& r, const char* path, size_t arch_off,
              uint32_t arch_size) {
    size_t   ft_off  = r.pos;
    uint32_t ft_hash = r.u32();
    if (ft_hash != kFtHashV4) {
        std::cerr << "NNUE: unexpected SFNNv4 FT hash 0x" << std::hex << ft_hash
                  << " (expected 0x" << kFtHashV4 << ")" << std::dec << "\n";
        return false;
    }

    r.bulk(v4_ft_bias, sizeof(v4_ft_bias));

    const size_t ftCount   = static_cast<size_t>(V4_INPUTS) * V4_HALF;
    const size_t psqtCount = static_cast<size_t>(V4_INPUTS) * V4_PSQT;
    aligned_alloc_once(v4_ft_weight, ftCount);
    aligned_alloc_once(v4_psqt_weight, psqtCount);
    r.bulk(v4_ft_weight, ftCount * sizeof(int16_t));
    r.bulk(v4_psqt_weight, psqtCount * sizeof(int32_t));

    size_t   net_off  = r.pos;
    uint32_t net_hash = 0;
    for (int s = 0; s < V4_STACKS; ++s) {
        uint32_t h = r.u32();
        if (s == 0)
            net_hash = h;
        else if (h != net_hash) {
            std::cerr << "NNUE: SFNNv4 stack " << s << " hash mismatch\n";
            return false;
        }
        V4Stack& st = v4_stack[s];
        r.bulk(st.b0, sizeof(st.b0));
        r.bulk(st.w0, sizeof(st.w0));
        r.bulk(st.b1, sizeof(st.b1));
        r.bulk(st.w1, sizeof(st.w1));
        r.bulk(st.b2, sizeof(st.b2));
        r.bulk(st.w2, sizeof(st.w2));
    }

    if (!r.ok) {
        std::cerr << "NNUE: file too small / truncated while parsing " << path << "\n";
        return false;
    }
    if (r.pos != r.size) {
        std::cerr << "NNUE: LAYOUT ERROR (SFNNv4) bytes consumed = " << r.pos
                  << " / " << r.size << " (mismatch)\n"
                  << "  arch@" << arch_off << " size " << arch_size
                  << "  ft@" << ft_off << "  net@" << net_off << "\n";
        return false;
    }

    std::printf("NNUE (HalfKAv2_hm/SFNNv4) ft_hash=0x%08X net_hash=0x%08X\n",
                ft_hash, net_hash);
    return true;
}

// Load a single file without any fallback logic. On success sets g_arch and
// g_loaded; on failure leaves the previous network state untouched EXCEPT when
// the failing parse already overwrote the weight arrays of the same arch (the
// caller then falls back to reloading a known-good net).
NNUE_STACK_ALIGN bool load_file(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;

    std::streamsize fsize = f.tellg();
    if (fsize <= 0) return false;
    f.seekg(0, std::ios::beg);

    const size_t bufSize = static_cast<size_t>(fsize);
    uint8_t* raw = static_cast<uint8_t*>(std::malloc(bufSize));
    if (!raw) return false;
    if (!f.read(reinterpret_cast<char*>(raw), fsize)) { std::free(raw); return false; }

    Reader r{raw, bufSize};

    // ---- Header ----
    uint32_t version   = r.u32();
    uint32_t file_hash = r.u32();
    uint32_t arch_size = r.u32();
    size_t   arch_off  = r.pos;
    const char* desc    = nullptr;   // description bytes inside `raw` (not 0-terminated)
    size_t      descLen = 0;
    if (r.need(arch_size)) {
        desc    = reinterpret_cast<const char*>(raw + r.pos);
        descLen = arch_size;
        r.skip(arch_size);
    }

    bool ok = false;
    Arch arch = ARCH_NONE;
    if (version == kVersionHalfKP) {
        arch = ARCH_HALFKP;
        ok = parse_halfkp(r, path, arch_off, arch_size);
    } else if (version == kVersionSFNNv4) {
        arch = ARCH_HALFKAV2_HM;
        ok = parse_v4(r, path, arch_off, arch_size);
    } else {
        std::cerr << "NNUE: unsupported version 0x" << std::hex << version
                  << std::dec << " in " << path << "\n";
    }

    if (ok) {
        g_arch   = arch;
        g_loaded = true;
        std::cout << "NNUE loaded: " << path
                  << ", bytes consumed = " << r.pos << " / " << bufSize << "\n";
        std::printf("NNUE version=0x%08X file_hash=0x%08X arch=%s\n",
                    version, file_hash,
                    arch == ARCH_HALFKP ? "HalfKP/SFNNv1" : "HalfKAv2_hm/SFNNv4");
        if (desc && descLen)
            std::printf("NNUE description: %.*s\n", int(descLen), desc);
        std::fflush(stdout);
    }

    std::free(raw);
    return ok;
}

}  // namespace

// ---------------------------------------------------------------------------
// load() — public entry: parse `path`; if that fails and no (working) net is
// available, fall back to the bundled HalfKP net so the engine is never
// netless.
// ---------------------------------------------------------------------------
bool load(const std::string& path) {
    const bool ok = load_file(path.c_str());
    if (!ok && !g_loaded) {
        const char* fallbacks[] = {
            "nets/nn-halfkp.nnue",
            "C:/Users/abhis/Desktop/OSS/Client/nets/nn-halfkp.nnue",
        };
        for (const char* fb : fallbacks) {
            if (std::strcmp(path.c_str(), fb) != 0 && load_file(fb)) {
                std::cout << "info string NNUE fell back to " << fb << "\n";
                break;
            }
        }
    }
    return ok;
}

bool loaded() { return g_loaded; }

// ===========================================================================
// HalfKP inference
// ===========================================================================
namespace {

// Shared 512->32->32->1 forward pass over two clipped per-perspective
// accumulators (side-to-move first). Full-refresh and incremental paths both
// funnel through here, so their arithmetic is byte-for-byte identical.
template <typename Acc>
inline Value network_forward_hk(const Acc* acc_stm, const Acc* acc_nstm) {
    alignas(64) uint8_t input[L0_IN];
    for (int i = 0; i < kHalfDimensions; ++i)
        input[i] = static_cast<uint8_t>(clamp127(int(acc_stm[i])));
    for (int i = 0; i < kHalfDimensions; ++i)
        input[kHalfDimensions + i] = static_cast<uint8_t>(clamp127(int(acc_nstm[i])));

    alignas(64) uint8_t h0[L0_OUT];
    for (int o = 0; o < L0_OUT; ++o) {
        int32_t s = l0_bias[o];
        const int8_t* w = &l0_weight[static_cast<size_t>(o) * L0_IN];
        for (int i = 0; i < L0_IN; ++i)
            s += int(input[i]) * int(w[i]);
        h0[o] = static_cast<uint8_t>(clamp127(s >> WEIGHT_SCALE_BITS));
    }

    alignas(64) uint8_t h1[L1_OUT];
    for (int o = 0; o < L1_OUT; ++o) {
        int32_t s = l1_bias[o];
        const int8_t* w = &l1_weight[static_cast<size_t>(o) * L1_IN];
        for (int i = 0; i < L1_IN; ++i)
            s += int(h0[i]) * int(w[i]);
        h1[o] = static_cast<uint8_t>(clamp127(s >> WEIGHT_SCALE_BITS));
    }

    int32_t s = out_bias[0];
    for (int i = 0; i < L2_IN; ++i)
        s += int(h1[i]) * int(out_weight[i]);

    return Value(s / FV_SCALE);
}

NNUE_STACK_ALIGN Value evaluate_hk(const Position& pos) {
    alignas(64) int32_t acc[COLOR_NB][kHalfDimensions];
    for (int p = 0; p < COLOR_NB; ++p)
        for (int i = 0; i < kHalfDimensions; ++i)
            acc[p][i] = ft_bias[i];

    for (int pc = 0; pc < COLOR_NB; ++pc) {
        const Color perspective = Color(pc);
        const Square ksq = pos.king_square(perspective);

        Bitboard occ = pos.pieces() ^ pos.pieces(KING);
        while (occ) {
            Square s   = pop_lsb(occ);
            Piece  pic = pos.piece_on(s);
            int    findex = make_index_hk(perspective, ksq, pic, s);

            const int16_t* w = &ft_weight[static_cast<size_t>(findex) * kHalfDimensions];
            for (int i = 0; i < kHalfDimensions; ++i)
                acc[perspective][i] += w[i];
        }
    }

    const Color stm = pos.side_to_move();
    return network_forward_hk<int32_t>(acc[stm], acc[~stm]);
}

NNUE_STACK_ALIGN void refresh_perspective_hk(Accumulator& a, const Position& pos, Color c) {
    alignas(64) int32_t sum[kHalfDimensions];
    for (int i = 0; i < kHalfDimensions; ++i)
        sum[i] = ft_bias[i];

    const Square ksq = pos.king_square(c);
    Bitboard occ = pos.pieces() ^ pos.pieces(KING);
    while (occ) {
        Square s   = pop_lsb(occ);
        Piece  pic = pos.piece_on(s);
        int    findex = make_index_hk(c, ksq, pic, s);
        const int16_t* w = &ft_weight[static_cast<size_t>(findex) * kHalfDimensions];
        for (int i = 0; i < kHalfDimensions; ++i)
            sum[i] += w[i];
    }

    for (int i = 0; i < kHalfDimensions; ++i)
        a.acc[c][i] = static_cast<int16_t>(sum[i]);
    a.computed[c] = true;
}

// Derive `child` from `parent` for HalfKP (kings are NOT features; the moving
// side's accumulator is rebuilt when its king moves).
NNUE_STACK_ALIGN void update_hk(Accumulator& child, const Accumulator& parent,
                                const Position& before, Move m) {
    const Color us    = before.side_to_move();
    const Color them  = ~us;
    const Square from = m.from_sq();
    const Square to   = m.to_sq();
    const MoveType mt = m.type_of();
    const Piece moved = before.piece_on(from);
    const bool  kingMove = (type_of(moved) == KING);

    const Square kto =
        (mt == CASTLING) ? relative_square(us, (to > from) ? SQ_G1 : SQ_C1) : to;

    Piece capturedPc = NO_PIECE;
    Square capsq = to;
    if (mt == EN_PASSANT) {
        capturedPc = make_piece(them, PAWN);
        capsq = to - (us == WHITE ? NORTH : SOUTH);
    } else if (mt != CASTLING) {
        Piece onTo = before.piece_on(to);
        if (onTo != NO_PIECE) {
            capturedPc = onTo;
            capsq = to;
        }
    }

    const Piece placed =
        (mt == PROMOTION) ? make_piece(us, m.promotion_type()) : moved;

    for (int pc = 0; pc < COLOR_NB; ++pc) {
        const Color P = Color(pc);

        if (kingMove && P == us) {
            alignas(64) int32_t sum[kHalfDimensions];
            for (int i = 0; i < kHalfDimensions; ++i)
                sum[i] = ft_bias[i];

            const Square ksqNew = kto;

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

                if (capturedPc != NO_PIECE && s == capsq)
                    continue;
                if (mt == CASTLING && s == rfrom)
                    s = rto;

                int findex = make_index_hk(P, ksqNew, pic, s);
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

        const Square ksq = before.king_square(P);

        int subIdx[3];
        int addIdx[2];
        int nSub = 0, nAdd = 0;

        if (mt == CASTLING) {
            const bool kingSide = to > from;
            const Square rfrom = kingSide ? relative_square(us, SQ_H1)
                                          : relative_square(us, SQ_A1);
            const Square rto   = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
            const Piece rook   = make_piece(us, ROOK);
            subIdx[nSub++] = make_index_hk(P, ksq, rook, rfrom);
            addIdx[nAdd++] = make_index_hk(P, ksq, rook, rto);
        } else {
            if (!kingMove) {
                subIdx[nSub++] = make_index_hk(P, ksq, moved, from);
                addIdx[nAdd++] = make_index_hk(P, ksq, placed, to);
            }
            if (capturedPc != NO_PIECE)
                subIdx[nSub++] = make_index_hk(P, ksq, capturedPc, capsq);
        }

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

// ===========================================================================
// SFNNv4 inference
// ===========================================================================

// Layer-stack forward pass over the two int16 accumulators + PSQT sums.
// `acc_stm`/`acc_nstm` are the raw 1024-wide rows; psqt_* the 8 bucket sums.
NNUE_STACK_ALIGN Value network_forward_v4(const int16_t* acc_stm,
                                          const int16_t* acc_nstm,
                                          const int32_t* psqt_stm,
                                          const int32_t* psqt_nstm,
                                          int bucket) {
    const int32_t psqt = (psqt_stm[bucket] - psqt_nstm[bucket]) / 2;

    // Feature transform: pairwise clipped multiplication, stm half first.
    // input[p*512 + j] = clip(acc[j]) * clip(acc[j+512]) / 128    (0..126)
    alignas(64) uint8_t input[FC0_IN];
    const int16_t* accs[2] = {acc_stm, acc_nstm};
    for (int p = 0; p < 2; ++p) {
        const int16_t* a = accs[p];
        uint8_t* out = input + p * (V4_HALF / 2);
        for (int j = 0; j < V4_HALF / 2; ++j) {
            int s0 = clamp127(int(a[j]));
            int s1 = clamp127(int(a[j + V4_HALF / 2]));
            out[j] = static_cast<uint8_t>((s0 * s1) / 128);
        }
    }

    const V4Stack& st = v4_stack[bucket];

    // fc_0: 1024 -> 16 (raw int32 kept: neuron 15 is the skip connection).
    int32_t f0[FC0_OUT];
    for (int o = 0; o < FC0_OUT; ++o) {
        int32_t s = st.b0[o];
        const int8_t* w = &st.w0[static_cast<size_t>(o) * FC0_IN];
        for (int i = 0; i < FC0_IN; ++i)
            s += int(input[i]) * int(w[i]);
        f0[o] = s;
    }

    // ac_sqr_0 + ac_0 -> fc_1 input (30 wide). Stockfish (nnue_architecture.h
    // propagate + sqr_clipped_relu.h) builds fc_1's input as the CONCATENATION of
    //   [0 ..14]  SqrClippedReLU(f0[0..14]) = clamp(0,127, (f0^2 >> 12) / 128)
    //   [15..29]  ClippedReLU   (f0[0..14]) = clamp(0,127,  f0 >> 6)
    // (index 15 of fc_0 is the skip neuron only; it does NOT feed fc_1.)
    uint8_t a0[FC1_IN];  // FC1_IN == 30
    for (int i = 0; i < FC_0_OUTPUTS; ++i) {
        int64_t sq = ((int64_t(f0[i]) * f0[i]) >> (2 * WEIGHT_SCALE_BITS)) / 128;
        a0[i]               = static_cast<uint8_t>(clamp127(int(sq)));
        a0[FC_0_OUTPUTS + i] = static_cast<uint8_t>(clamp127(f0[i] >> WEIGHT_SCALE_BITS));
    }

    // fc_1: 30 -> 32 (weights stored row-major with padded stride FC1_PAD).
    uint8_t a1[FC1_OUT];
    for (int o = 0; o < FC1_OUT; ++o) {
        int32_t s = st.b1[o];
        const int8_t* w = &st.w1[static_cast<size_t>(o) * FC1_PAD];
        for (int i = 0; i < FC1_IN; ++i)
            s += int(a0[i]) * int(w[i]);
        a1[o] = static_cast<uint8_t>(clamp127(s >> WEIGHT_SCALE_BITS));
    }

    // fc_2: 32 -> 1.
    int32_t s2 = st.b2[0];
    for (int i = 0; i < FC2_IN; ++i)
        s2 += int(a1[i]) * int(st.w2[i]);

    // Skip connection: fc_0 raw neuron 15, rescaled exactly as Stockfish does.
    const int32_t fwdOut =
        f0[FC0_OUT - 1] * (600 * V4_OUTPUT_SCALE) / (127 * (1 << WEIGHT_SCALE_BITS));
    const int32_t positional = s2 + fwdOut;

    // (materialist + positional) -> SF-reported centipawns.
    return Value(int64_t(psqt + positional) * 100 / (V4_OUTPUT_SCALE * V4_NORMALIZE));
}

// Rebuild one perspective (accumulator + PSQT sums) from scratch. ALL pieces,
// including both kings, are features in HalfKAv2_hm.
NNUE_STACK_ALIGN void refresh_perspective_v4(Accumulator& a, const Position& pos, Color c) {
    alignas(64) int32_t sum[V4_HALF];
    for (int i = 0; i < V4_HALF; ++i)
        sum[i] = v4_ft_bias[i];
    int32_t psqt[V4_PSQT] = {};

    const Square ksq = pos.king_square(c);
    Bitboard occ = pos.pieces();
    while (occ) {
        Square s   = pop_lsb(occ);
        Piece  pic = pos.piece_on(s);
        int    findex = make_index_v4(c, s, pic, ksq);
        const int16_t* w = &v4_ft_weight[static_cast<size_t>(findex) * V4_HALF];
        for (int i = 0; i < V4_HALF; ++i)
            sum[i] += w[i];
        const int32_t* pw = &v4_psqt_weight[static_cast<size_t>(findex) * V4_PSQT];
        for (int k = 0; k < V4_PSQT; ++k)
            psqt[k] += pw[k];
    }

    for (int i = 0; i < V4_HALF; ++i)
        a.acc[c][i] = static_cast<int16_t>(sum[i]);
    for (int k = 0; k < V4_PSQT; ++k)
        a.psqt[c][k] = psqt[k];
    a.computed[c] = true;
}

NNUE_STACK_ALIGN Value evaluate_v4(const Position& pos) {
    // Full refresh into a local accumulator, then the shared forward pass —
    // guaranteed identical arithmetic to the incremental path.
    Accumulator a;
    refresh_perspective_v4(a, pos, WHITE);
    refresh_perspective_v4(a, pos, BLACK);
    const Color stm = pos.side_to_move();
    return network_forward_v4(a.acc[stm], a.acc[~stm], a.psqt[stm], a.psqt[~stm],
                              v4_bucket(pos));
}

// Derive `child` from `parent` for HalfKAv2_hm.
//
// Kings ARE features here, for BOTH perspectives:
//   * When side P's OWN king moves (incl. castling), every feature index of P
//     is keyed on the new king square / bucket / mirror — full rebuild of P.
//   * For the OTHER perspective, the enemy king is an ordinary feature: its
//     relocation is a plain sub/add delta (no refresh).
NNUE_STACK_ALIGN void update_v4(Accumulator& child, const Accumulator& parent,
                                const Position& before, Move m) {
    const Color us    = before.side_to_move();
    const Color them  = ~us;
    const Square from = m.from_sq();
    const Square to   = m.to_sq();
    const MoveType mt = m.type_of();
    const Piece moved = before.piece_on(from);
    const bool  kingMove = (type_of(moved) == KING);  // includes castling

    // King's destination (castling lands on the relative g1/c1 square).
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
        if (onTo != NO_PIECE) {
            capturedPc = onTo;
            capsq = to;
        }
    }

    const Piece placed =
        (mt == PROMOTION) ? make_piece(us, m.promotion_type()) : moved;

    // Castling rook relocation (both perspectives see it).
    Square rfrom = SQ_NONE, rto = SQ_NONE;
    if (mt == CASTLING) {
        const bool kingSide = to > from;
        rfrom = kingSide ? relative_square(us, SQ_H1) : relative_square(us, SQ_A1);
        rto   = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
    }

    for (int pc = 0; pc < COLOR_NB; ++pc) {
        const Color P = Color(pc);

        // Own king moved -> full rebuild of P against the new king square,
        // enumerating `before`'s pieces with the move's relocations applied.
        if (kingMove && P == us) {
            alignas(64) int32_t sum[V4_HALF];
            for (int i = 0; i < V4_HALF; ++i)
                sum[i] = v4_ft_bias[i];
            int32_t psqt[V4_PSQT] = {};

            const Square ksqNew = kto;

            Bitboard occ = before.pieces();
            while (occ) {
                Square s   = pop_lsb(occ);
                Piece  pic = before.piece_on(s);

                // Captured piece vanishes (kings can capture).
                if (capturedPc != NO_PIECE && s == capsq)
                    continue;
                // The moving king relocates from `from` to `kto`.
                if (s == from)
                    s = ksqNew;
                // Castling rook relocation.
                else if (mt == CASTLING && s == rfrom)
                    s = rto;

                int findex = make_index_v4(P, s, pic, ksqNew);
                const int16_t* w =
                    &v4_ft_weight[static_cast<size_t>(findex) * V4_HALF];
                for (int i = 0; i < V4_HALF; ++i)
                    sum[i] += w[i];
                const int32_t* pw =
                    &v4_psqt_weight[static_cast<size_t>(findex) * V4_PSQT];
                for (int k = 0; k < V4_PSQT; ++k)
                    psqt[k] += pw[k];
            }

            for (int i = 0; i < V4_HALF; ++i)
                child.acc[P][i] = static_cast<int16_t>(sum[i]);
            for (int k = 0; k < V4_PSQT; ++k)
                child.psqt[P][k] = psqt[k];
            child.computed[P] = true;
            continue;
        }

        // Delta path: P's own king did not move, so P's king square (and hence
        // bucket/mirror) is unchanged. The mover — even an enemy KING — is an
        // ordinary feature relocation.
        const Square ksq = before.king_square(P);

        int subIdx[3];
        int addIdx[2];
        int nSub = 0, nAdd = 0;

        if (mt == CASTLING) {
            // P != us here (us was handled above). Enemy king + rook relocate.
            subIdx[nSub++] = make_index_v4(P, from, moved, ksq);
            addIdx[nAdd++] = make_index_v4(P, kto, moved, ksq);
            const Piece rook = make_piece(us, ROOK);
            subIdx[nSub++] = make_index_v4(P, rfrom, rook, ksq);
            addIdx[nAdd++] = make_index_v4(P, rto, rook, ksq);
        } else {
            subIdx[nSub++] = make_index_v4(P, from, moved, ksq);
            addIdx[nAdd++] = make_index_v4(P, to, placed, ksq);
            if (capturedPc != NO_PIECE)
                subIdx[nSub++] = make_index_v4(P, capsq, capturedPc, ksq);
        }

        const int16_t* wsub[3];
        const int16_t* wadd[2];
        for (int k = 0; k < nSub; ++k)
            wsub[k] = &v4_ft_weight[static_cast<size_t>(subIdx[k]) * V4_HALF];
        for (int k = 0; k < nAdd; ++k)
            wadd[k] = &v4_ft_weight[static_cast<size_t>(addIdx[k]) * V4_HALF];

        const int16_t* src = parent.acc[P];
        int16_t*       dst = child.acc[P];
        for (int i = 0; i < V4_HALF; ++i) {
            int v = int(src[i]);
            for (int k = 0; k < nSub; ++k) v -= int(wsub[k][i]);
            for (int k = 0; k < nAdd; ++k) v += int(wadd[k][i]);
            dst[i] = static_cast<int16_t>(v);
        }

        for (int k = 0; k < V4_PSQT; ++k) {
            int32_t v = parent.psqt[P][k];
            for (int j = 0; j < nSub; ++j)
                v -= v4_psqt_weight[static_cast<size_t>(subIdx[j]) * V4_PSQT + k];
            for (int j = 0; j < nAdd; ++j)
                v += v4_psqt_weight[static_cast<size_t>(addIdx[j]) * V4_PSQT + k];
            child.psqt[P][k] = v;
        }
        child.computed[P] = true;
    }
}

}  // namespace

// ===========================================================================
// Public dispatch
// ===========================================================================
NNUE_STACK_ALIGN Value evaluate(const Position& pos) {
    return g_arch == ARCH_HALFKAV2_HM ? evaluate_v4(pos) : evaluate_hk(pos);
}

NNUE_STACK_ALIGN void refresh_perspective(Accumulator& a, const Position& pos, Color c) {
    if (g_arch == ARCH_HALFKAV2_HM)
        refresh_perspective_v4(a, pos, c);
    else
        refresh_perspective_hk(a, pos, c);
}

void refresh(Accumulator& a, const Position& pos) {
    refresh_perspective(a, pos, WHITE);
    refresh_perspective(a, pos, BLACK);
}

NNUE_STACK_ALIGN Value evaluate(const Accumulator& a, const Position& pos) {
    const Color stm = pos.side_to_move();
    Value inc = (g_arch == ARCH_HALFKAV2_HM)
        ? network_forward_v4(a.acc[stm], a.acc[~stm], a.psqt[stm], a.psqt[~stm], v4_bucket(pos))
        : network_forward_hk<int16_t>(a.acc[stm], a.acc[~stm]);
#ifdef NNUE_EXACT_CHECK
    static long long checked = 0, mism = 0;
    Accumulator fresh;
    refresh(fresh, pos);
    Value ref = (g_arch == ARCH_HALFKAV2_HM)
        ? network_forward_v4(fresh.acc[stm], fresh.acc[~stm], fresh.psqt[stm], fresh.psqt[~stm], v4_bucket(pos))
        : network_forward_hk<int16_t>(fresh.acc[stm], fresh.acc[~stm]);
    ++checked;
    if (ref != inc) { ++mism; if (mism <= 8) std::fprintf(stderr, "INCMISMATCH inc=%d ref=%d\n", int(inc), int(ref)); }
    if ((checked % 50000) == 0) std::fprintf(stderr, "EXACTCHK checked=%lld mismatches=%lld\n", checked, mism);
#endif
    return inc;
}

NNUE_STACK_ALIGN void update(Accumulator& child, const Accumulator& parent,
                             const Position& before, Move m) {
    if (g_arch == ARCH_HALFKAV2_HM)
        update_v4(child, parent, before, m);
    else
        update_hk(child, parent, before, m);
}

}  // namespace NNUE
}  // namespace engine

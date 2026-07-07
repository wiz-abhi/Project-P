// tt.cpp — Transposition table implementation.
//
// Implements the contract declared in tt.hpp: TTEntry::save, and the
// TranspositionTable lifecycle (resize/clear/dtor), probe, and hashfull.
#include "tt.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
    #include <malloc.h>
#endif

namespace engine {

// Portable 64-byte-aligned allocation (Windows lacks posix_memalign; POSIX lacks
// _aligned_malloc). Keeps the engine buildable on the Linux deploy target.
static void* tt_aligned_alloc(size_t bytes) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, 64);
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0)
        p = nullptr;
    return p;
#endif
}

static void tt_aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// The global transposition table.
TranspositionTable TT;

// ---------------------------------------------------------------------------
// TTEntry::save — store a search result into this entry.
//
// Bit layout of genBound8: [7..3]=generation, [2]=pv, [1..0]=bound.
// ---------------------------------------------------------------------------
void TTEntry::save(Key k, Value v, bool pv, Bound b, int d, Move m, Value ev,
                   uint8_t gen8) {
    // Preserve an existing move on a same-position overwrite: only clobber the
    // stored move if the caller supplied one, or the position (key16) differs.
    if (m || uint16_t(k >> 48) != key16)
        move16 = m.raw();

    // Overwrite the rest only when the new entry is at least as valuable:
    //   - it is an exact bound, or
    //   - it refers to a different position, or
    //   - it is deep enough relative to what is already stored (Stockfish rule).
    if (b == BOUND_EXACT || uint16_t(k >> 48) != key16 ||
        d - DEPTH_OFFSET + 2 * pv > depth8 - 4) {

        assert(d - DEPTH_OFFSET >= 0 && "depth would underflow depth8");

        key16     = uint16_t(k >> 48);
        value16   = int16_t(v);
        eval16    = int16_t(ev);
        genBound8 = uint8_t(gen8 | (uint8_t(pv) << 2) | b);
        depth8    = uint8_t(d - DEPTH_OFFSET);
    }
}

// ---------------------------------------------------------------------------
// TranspositionTable lifecycle
// ---------------------------------------------------------------------------
TranspositionTable::~TranspositionTable() {
    if (table)
        tt_aligned_free(table);
}

void TranspositionTable::resize(size_t mbSize) {
    if (table) {
        tt_aligned_free(table);
        table = nullptr;
    }

    clusterCount = (mbSize * 1024 * 1024) / sizeof(TTCluster);
    if (clusterCount == 0)
        clusterCount = 1;   // always keep a usable table

    table = static_cast<TTCluster*>(
        tt_aligned_alloc(clusterCount * sizeof(TTCluster)));
    assert(table && "TT allocation failed");

    clear();
}

void TranspositionTable::clear() {
    if (table)
        std::memset(table, 0, clusterCount * sizeof(TTCluster));
}

// ---------------------------------------------------------------------------
// probe — locate the entry for `key`, or the slot to overwrite on a miss.
// ---------------------------------------------------------------------------
TTEntry* TranspositionTable::probe(Key key, bool& found) const {
    TTEntry* const first = table[key % clusterCount].entry;
    const uint16_t key16 = uint16_t(key >> 48);

    for (int i = 0; i < TTCluster::ClusterSize; ++i) {
        TTEntry* e = &first[i];

        if (e->key16 == key16 && e->occupied()) {
            // Hit: refresh the generation, preserving the pv + bound bits.
            e->genBound8 = uint8_t(generation8 |
                                   (e->genBound8 & (0x4 | 0x3)));
            found = true;
            return e;
        }

        if (!e->occupied()) {
            // Empty slot: available to fill.
            found = false;
            return e;
        }
    }

    // No hit and no empty slot — evict the least valuable entry using the
    // standard relative-age replacement metric (higher = more valuable).
    TTEntry* replace = first;
    for (int i = 1; i < TTCluster::ClusterSize; ++i) {
        TTEntry* e = &first[i];
        if (replace->depth8 -
                ((TTEntry::GENERATION_CYCLE + generation8 - replace->genBound8) &
                 TTEntry::GENERATION_MASK)
            > e->depth8 -
                  ((TTEntry::GENERATION_CYCLE + generation8 - e->genBound8) &
                   TTEntry::GENERATION_MASK))
            replace = e;
    }

    found = false;
    return replace;
}

// ---------------------------------------------------------------------------
// hashfull — estimate table occupancy (permille) for the current generation.
// ---------------------------------------------------------------------------
int TranspositionTable::hashfull() const {
    int cnt = 0;
    const size_t clusters = clusterCount < 1000 ? clusterCount : 1000;
    for (size_t i = 0; i < clusters; ++i)
        for (int j = 0; j < TTCluster::ClusterSize; ++j) {
            const TTEntry& e = table[i].entry[j];
            if (e.occupied() && e.generation() == generation8)
                ++cnt;
        }

    const int examined = int(clusters) * TTCluster::ClusterSize;
    return examined ? cnt * 1000 / examined : 0;
}

}  // namespace engine

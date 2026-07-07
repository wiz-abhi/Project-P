// tt.hpp — Transposition table (Zobrist-keyed search cache).
//
// A large hash table mapping position keys to previously-computed search results
// so identical positions reached by different move orders are not re-searched.
#pragma once

#include "types.hpp"

namespace engine {

enum Bound : uint8_t {
    BOUND_NONE  = 0,
    BOUND_UPPER = 1,   // fail-low: value is an upper bound
    BOUND_LOWER = 2,   // fail-high: value is a lower bound
    BOUND_EXACT = BOUND_UPPER | BOUND_LOWER
};

// One packed table entry (10 bytes). Access via the accessor methods; write via
// save(). key16 holds the upper 16 bits of the Zobrist key for collision checks.
struct TTEntry {
    Move   move() const  { return Move(move16); }
    Value  value() const { return Value(value16); }
    Value  eval() const  { return Value(eval16); }
    int    depth() const { return int(depth8) + DEPTH_OFFSET; }
    Bound  bound() const { return Bound(genBound8 & 0x3); }
    bool   is_pv() const { return bool(genBound8 & 0x4); }
    uint8_t generation() const { return uint8_t(genBound8 & GENERATION_MASK); }
    bool   occupied() const { return bool(depth8); }

    // Store search result. Preserves an existing move on a same-key overwrite.
    void save(Key k, Value v, bool pv, Bound b, int d, Move m, Value ev, uint8_t gen8);

    static constexpr int     DEPTH_OFFSET     = -7;      // lowest storable depth
    static constexpr unsigned GENERATION_BITS  = 3;
    static constexpr unsigned GENERATION_DELTA = 1 << GENERATION_BITS;   // 8
    static constexpr unsigned GENERATION_CYCLE = 255 + GENERATION_DELTA;
    static constexpr unsigned GENERATION_MASK  = (0xFF << GENERATION_BITS) & 0xFF;

   private:
    friend class TranspositionTable;
    uint16_t key16;
    uint8_t  depth8;
    uint8_t  genBound8;   // bits: [7..3]=generation, [2]=pv, [1..0]=bound
    uint16_t move16;
    int16_t  value16;
    int16_t  eval16;
};

// A cluster is a cache-line-sized group of entries sharing a hash slot; save()
// picks the least valuable one to replace.
struct TTCluster {
    static constexpr int ClusterSize = 3;
    TTEntry entry[ClusterSize];
    char    padding[2];   // pad TTCluster to 32 bytes
};

class TranspositionTable {
   public:
    ~TranspositionTable();

    void      resize(size_t mbSize);           // (re)allocate; clears the table
    void      clear();                          // zero all entries
    void      new_search() { generation8 += TTEntry::GENERATION_DELTA; }
    uint8_t   generation() const { return generation8; }

    // Probe: sets found; returns a writable entry pointer (a hit, or the slot to
    // overwrite on a miss).
    TTEntry*  probe(Key key, bool& found) const;
    int       hashfull() const;                 // permille of entries used this gen

   private:
    friend struct TTEntry;
    TTCluster* table       = nullptr;
    size_t     clusterCount = 0;
    uint8_t    generation8  = 0;
};

extern TranspositionTable TT;   // global, defined in tt.cpp

}  // namespace engine

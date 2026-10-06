// A program is reusable only when its entire byte image, size and VU unit
// match. Hashes identify native programs; they never authorize a cache hit.
namespace wotmVuImages {
template<class Pair> struct Cache {
    struct Entry {
        std::vector<uint8_t> bytes;
        std::vector<Pair> pairs;
        uint64_t hash=0;
        uint8_t unit=0;
        bool valid=false;
    };
    std::array<Entry,8> entries;
    unsigned next=0;
    Entry *find(const uint8_t *code,uint32_t size,uint8_t unit) {
        for(auto &entry:entries)
            if(entry.valid && entry.unit==unit && entry.bytes.size()==size &&
               std::memcmp(entry.bytes.data(),code,size)==0) return &entry;
        return nullptr;
    }
    void save(const uint8_t *code,uint32_t size,uint8_t unit,
              const Pair *pairs,uint32_t count,uint64_t hash) {
        auto &entry=entries[next];next=(next+1u)%entries.size();
        entry.bytes.assign(code,code+size);
        entry.pairs.assign(pairs,pairs+count);
        entry.unit=unit;entry.hash=hash;entry.valid=true;
    }
};
template<class Usage> bool sameUsage(const Usage &a,const Usage &b) {
    for(unsigned i=0;i<2;++i)
        if(a.vfRead[i].reg!=b.vfRead[i].reg || a.vfRead[i].lanes!=b.vfRead[i].lanes) return false;
    return a.vfWrite.reg==b.vfWrite.reg && a.vfWrite.lanes==b.vfWrite.lanes &&
        a.vfReadCount==b.vfReadCount && a.viRead==b.viRead && a.viWrite==b.viWrite &&
        a.accRead==b.accRead && a.accWrite==b.accWrite && a.latency==b.latency &&
        a.vfLatency==b.vfLatency && a.viLatency==b.viLatency && a.pipeline==b.pipeline &&
        a.waitQ==b.waitQ && a.waitP==b.waitP && a.readsClip==b.readsClip &&
        a.writesClip==b.writesClip && a.delaysNextBranchRead==b.delaysNextBranchRead &&
        a.reserved==b.reserved;
}
template<class Pair> bool samePair(const Pair &a,const Pair &b) {
    return a.lower==b.lower && a.upper==b.upper && sameUsage(a.lowerUsage,b.lowerUsage) &&
        sameUsage(a.upperUsage,b.upperUsage) && a.iBit==b.iBit && a.eBit==b.eBit &&
        a.mBit==b.mBit && a.dBit==b.dBit && a.tBit==b.tBit &&
        a.upperVfShadowReg==b.upperVfShadowReg && a.suppressedLowerVf==b.suppressedLowerVf &&
        a.vfReadMask==b.vfReadMask && a.viReadMask==b.viReadMask && a.readsAcc==b.readsAcc;
}
}

#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// Exact FNV continuation cache for owned DMA snapshots. An address/seed match
// only selects a candidate: every payload byte is compared before reusing it.
// No references to guest RAM or a mutable DMA buffer survive this call.
struct WotmMotionSignatureCache {
    struct Entry {uint32_t tag=0;uint64_t seed=0,result=0;bool valid=false;std::vector<uint8_t> bytes;};
    static constexpr size_t slotCount=512,minBytes=64,maxBytes=16384;
    std::array<Entry,slotCount> entries;
    std::array<uint8_t,slotCount/2> victim{};
    uint64_t queries=0,hits=0,misses=0,comparedBytes=0,hashedBytes=0;

    static uint64_t reference(uint64_t seed,const uint8_t* data,size_t size) {
        for(size_t i=0;i<size;++i){seed^=data[i];seed*=1099511628211ull;}
        return seed;
    }
    uint64_t hash(uint32_t tag,uint64_t seed,const uint8_t* data,size_t size) {
        ++queries;
        if(!size)return seed;
        if(size<minBytes || size>maxBytes){++misses;hashedBytes+=size;return reference(seed,data,size);}
        // Mix components separately; XOR alone cancels correlated tag strides
        // and continuation seeds, making unrelated keys fight for one slot.
        uint64_t mixed=seed+uint64_t(tag)*0x9e3779b97f4a7c15ull;
        mixed=(mixed^(mixed>>30))*0xbf58476d1ce4e5b9ull;
        mixed=(mixed^(mixed>>27))*0x94d049bb133111ebull;
        const size_t group=(mixed^(mixed>>31))&(slotCount/2-1),base=group*2;
        size_t selected=base+victim[group];
        for(size_t way=0;way<2;++way) {
            auto& candidate=entries[base+way];
            if(!candidate.valid)selected=base+way;
            if(candidate.valid && candidate.tag==tag && candidate.seed==seed) {
                selected=base+way;
                if(candidate.bytes.size()==size) {
                    comparedBytes+=size;
                    if(std::memcmp(candidate.bytes.data(),data,size)==0){
                        ++hits;victim[group]=uint8_t(way^1u);return candidate.result;
                    }
                }
                break;
            }
        }
        auto& entry=entries[selected];victim[group]=uint8_t((selected-base)^1u);
        ++misses;hashedBytes+=size;
        const auto result=reference(seed,data,size);
        entry.bytes.assign(data,data+size);entry.tag=tag;entry.seed=seed;entry.result=result;entry.valid=true;
        return result;
    }
};

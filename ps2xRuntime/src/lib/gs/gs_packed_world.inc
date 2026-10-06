// Private prototype: complete 0x412 PACKED world vertices only.
// Input contains count disjoint, readable 48-byte vertices; output holds count
// host vertices. The caller retains ADC, GS continuation, material and ownership.
// Preserve all host/GS vertex bytes and MXCSR controls. Admit signed zero or
// finite normal Q with all exceptions masked; retain legacy for other Q/modes.
// Unread masked status flags from unused auto-vectorized lanes are not a
// renderer contract; their diagnosed differences remain recorded in the fixture.
#include <cstdint>
#include <cstddef>
#include <cstring>
namespace wotm_vertex {
inline float normalizePackedWorldQ(float q) {
    if(q==0.0f) q=1.0f;
    return q;
}
inline uint64_t packedWorldLoad64(const uint8_t* p) {
    uint64_t value;
    std::memcpy(&value,p,sizeof(value));
    return value;
}
inline bool packedWorldQAdmitted(const uint8_t* raw,size_t count,unsigned mxcsr) {
    if((mxcsr&0x1f80u)!=0x1f80u) return false;
    for(size_t i=0;i<count;++i) {
        uint32_t q;std::memcpy(&q,raw+i*48u+8u,4);
        const uint32_t exponent=q&0x7f800000u;
        if(exponent==0x7f800000u || (exponent==0u && (q&0x007fffffu)!=0u)) return false;
    }
    return true;
}
template<class Out>
inline Out decodePackedGSVertex(const uint8_t* p,uint16_t u,uint16_t v) {
    Out out{};
    std::memcpy(&out.s,p,4);std::memcpy(&out.t,p+4,4);std::memcpy(&out.q,p+8,4);
    out.q=normalizePackedWorldQ(out.q);
    out.r=p[16];out.g=p[20];out.b=p[24];out.a=p[28];
    out.u=u;out.v=v;
    const uint64_t xy=packedWorldLoad64(p+32),zf=packedWorldLoad64(p+40);
    out.x=float(xy&0xffffu)/16.0f;out.y=float((xy>>32)&0xffffu)/16.0f;
    out.z=float((zf>>4)&0xffffffu);out.fog=uint8_t(zf>>36);
    return out;
}
}
// The scalar GS decoder above has no intrinsic dependencies. This x64 world
// converter below preserves the retained recorder's paired SSE2 division.
#if !defined(WOTM_PACKED_DECODE_ONLY)
#include <immintrin.h>
namespace wotm_vertex {
inline double packedWorldDepth(const uint8_t* p) {
    // Preserve the legacy bounded integer -> float -> double sequence.
    return float((packedWorldLoad64(p+40)>>4)&0xffffffu);
}
template<class Out>
inline void storePackedWorldVertex(Out& out,const uint8_t* p,float depth,
                                  float ofx,float ofy) {
    const uint64_t xy=packedWorldLoad64(p+32);
    const uint64_t zf=packedWorldLoad64(p+40);
    out.x=float(xy&0xffffu)/16.0f-ofx;
    out.y=float((xy>>32)&0xffffu)/16.0f-ofy;
    out.z=depth;
    out.r=static_cast<float>(p[16])/255.0f;
    out.g=static_cast<float>(p[20])/255.0f;
    out.b=static_cast<float>(p[24])/255.0f;
    out.a=static_cast<float>(p[28])/128.0f;
    std::memcpy(&out.s,p,4);
    std::memcpy(&out.t,p+4,4);
    std::memcpy(&out.q,p+8,4);
    out.q=normalizePackedWorldQ(out.q);
    out.fog=static_cast<float>(uint8_t(zf>>36));
}
template<class Out>
inline void convertPackedWorldVertices(Out* __restrict output,
    const uint8_t* __restrict raw,size_t count,float ofx,float ofy) {
    size_t i=0;
    for(;i+1<count;i+=2) {
        const uint8_t* p=raw+i*48u;
        const __m128d z=_mm_set_pd(packedWorldDepth(p+48u),packedWorldDepth(p));
        const __m128 depth=_mm_cvtpd_ps(_mm_div_pd(z,_mm_set1_pd(16777215.0)));
        storePackedWorldVertex(output[i],p,_mm_cvtss_f32(depth),ofx,ofy);
        storePackedWorldVertex(output[i+1],p+48u,
            _mm_cvtss_f32(_mm_shuffle_ps(depth,depth,_MM_SHUFFLE(1,1,1,1))),ofx,ofy);
    }
    if(i<count) {
        const uint8_t* p=raw+i*48u;
        storePackedWorldVertex(output[i],p,
            static_cast<float>(packedWorldDepth(p)/16777215.0),ofx,ofy);
    }
}
}
#endif

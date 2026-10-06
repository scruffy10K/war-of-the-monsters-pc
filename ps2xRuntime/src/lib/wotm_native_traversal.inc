// Native memory specialization of the retail hierarchy walker. Reuse its
// control flow so every resume PC, delay slot and scheduler checkpoint stays
// authoritative. The generated source is included, never edited or regenerated.
#include "ps2_runtime_macros.h"
#include <ps2_recompiled_functions.h>
#include <ps2_recompiled_stubs.h>
#include "ps2_syscalls.h"
#include "ps2_stubs.h"

namespace {
uint64_t s_traversalReadChecks=0;
// Keep uncommon generic decoding out of the node-loop instruction footprint.
template<class T> __declspec(noinline)
T wotmTraversalReference(uint8_t *ram, R5900Context *ctx,
                         PS2Runtime *runtime, uint32_t address) {
    const bool special=PS2Runtime::isSpecialAddress(address);
    if constexpr(sizeof(T)==1) return special ? runtime->Load8(ram,ctx,address) : Ps2FastRead8(ram,address);
    if constexpr(sizeof(T)==2) return special ? runtime->Load16(ram,ctx,address) : Ps2FastRead16(ram,address);
    if constexpr(sizeof(T)==4) return special ? runtime->Load32(ram,ctx,address) : Ps2FastRead32(ram,address);
    if constexpr(sizeof(T)==8) return special ? runtime->Load64(ram,ctx,address) : Ps2FastRead64(ram,address);
    if constexpr(sizeof(T)==16) return special ? runtime->Load128(ram,ctx,address) : Ps2FastRead128(ram,address);
}
template<class T, bool Check>
inline T wotmTraversalRead(uint8_t *ram, R5900Context *ctx, PS2Runtime *runtime,
                          uint32_t address) {
    const uint8_t *source=nullptr;
    if (address<=PS2_RAM_SIZE-sizeof(T)) source=ram+address;
    else if (address-0x70000000u<=PS2_SCRATCHPAD_SIZE-sizeof(T) &&
             (address&(sizeof(T)-1u))==0u)
        source=runtime->memory().getScratchpad()+(address-0x70000000u);
    // Aliases, wrapping, misaligned scratchpad and all MMIO retain the original
    // read path, including address exceptions and device side effects.
    if (!source) return wotmTraversalReference<T>(ram,ctx,runtime,address);
    T value;std::memcpy(&value,source,sizeof(value));
    if constexpr(Check) {
        const T expected=wotmTraversalReference<T>(ram,ctx,runtime,address);
        if(std::memcmp(&value,&expected,sizeof(value))) {
            std::fprintf(stderr,"[ee:traversal] read mismatch addr=%x bytes=%zu\n",address,sizeof(T));std::abort();
        }
        if(++s_traversalReadChecks%10000000ull==0)
            std::fprintf(stderr,"[ee:traversal] reads=%llu mismatches=0\n",s_traversalReadChecks);
    }
    return value;
}
}
#pragma push_macro("READ8")
#undef READ8
#define READ8(expression) wotmTraversalRead<uint8_t, WOTM_TRAVERSAL_CHECK>(rdram,ctx,runtime,uint32_t(expression))
#pragma push_macro("READ16")
#undef READ16
#define READ16(expression) wotmTraversalRead<uint16_t, WOTM_TRAVERSAL_CHECK>(rdram,ctx,runtime,uint32_t(expression))
#pragma push_macro("READ32")
#undef READ32
#define READ32(expression) wotmTraversalRead<uint32_t, WOTM_TRAVERSAL_CHECK>(rdram,ctx,runtime,uint32_t(expression))
#pragma push_macro("READ64")
#undef READ64
#define READ64(expression) wotmTraversalRead<uint64_t, WOTM_TRAVERSAL_CHECK>(rdram,ctx,runtime,uint32_t(expression))
#pragma push_macro("READ128")
#undef READ128
#define READ128(expression) wotmTraversalRead<__m128i, WOTM_TRAVERSAL_CHECK>(rdram,ctx,runtime,uint32_t(expression))
#define WOTM_TRAVERSAL_CHECK false
#define hierTraverseAsm_0x207198 wotmNativeTraversal
#include <hierTraverseAsm_0x207198.cpp>
#undef hierTraverseAsm_0x207198
#undef WOTM_TRAVERSAL_CHECK
#define WOTM_TRAVERSAL_CHECK true
#define hierTraverseAsm_0x207198 wotmCheckedTraversal
#include <hierTraverseAsm_0x207198.cpp>
#undef hierTraverseAsm_0x207198
#undef WOTM_TRAVERSAL_CHECK
#pragma pop_macro("READ8")
#pragma pop_macro("READ16")
#pragma pop_macro("READ32")
#pragma pop_macro("READ64")
#pragma pop_macro("READ128")


// Native memory specialization of the object packet builder (208410..20895c).
// Keep the original control flow, all resume slots, register effects, calls,
// exception paths and scheduler checkpoints. Generated code is included read-only.
namespace {
uint64_t s_packetReadChecks=0, s_packetWriteChecks=0;

template<class T>
inline uint8_t *wotmPacketPointer(uint8_t *ram, PS2Runtime *runtime, uint32_t address) {
    if(address<=PS2_RAM_SIZE-sizeof(T)) return ram+address;
    if(address-0x70000000u<=PS2_SCRATCHPAD_SIZE-sizeof(T) &&
       (address&(sizeof(T)-1u))==0u)
        return runtime->memory().getScratchpad()+(address-0x70000000u);
    // These are the same masked RAM aliases used by the generated fast path.
    // Admit only contiguous ranges; wrapped accesses and mapped devices keep
    // the reference decoder. KSEG2/3 translation is never bypassed.
    const uint32_t offset=address&PS2_RAM_MASK;
    if((Ps2IsUncachedRamMirrorAddress(address) ||
        (Ps2IsKseg01Address(address) && Ps2DirectMappedPhysicalAddress(address)<PS2_RAM_SIZE)) &&
       offset<=PS2_RAM_SIZE-sizeof(T)) return ram+offset;
    return nullptr;
}

template<class T, bool Check>
inline T wotmPacketRead(uint8_t *ram, R5900Context *ctx, PS2Runtime *runtime, uint32_t address) {
    uint8_t *source=wotmPacketPointer<T>(ram,runtime,address);
    if(!source) return wotmTraversalReference<T>(ram,ctx,runtime,address);
    T value;std::memcpy(&value,source,sizeof(value));
    if constexpr(Check) {
        const T expected=wotmTraversalReference<T>(ram,ctx,runtime,address);
        if(std::memcmp(&value,&expected,sizeof(value))) {
            std::fprintf(stderr,"[ee:packets] read mismatch pc=%x addr=%x bytes=%zu\n",ctx->pc,address,sizeof(T));
            std::abort();
        }
        if(++s_packetReadChecks==1 || s_packetReadChecks%1000000ull==0)
            std::fprintf(stderr,"[ee:packets] reads=%llu mismatches=0\n",s_packetReadChecks);
    }
    return value;
}

template<class T> __declspec(noinline)
void wotmPacketWriteReference(uint8_t *ram, R5900Context *ctx, PS2Runtime *runtime,
                              uint32_t address, T value) {
    const bool special=PS2Runtime::isSpecialAddress(address);
    if constexpr(sizeof(T)==1) {if(special)runtime->Store8(ram,ctx,address,value);else Ps2FastWrite8(ram,address,value);}
    if constexpr(sizeof(T)==2) {if(special)runtime->Store16(ram,ctx,address,value);else Ps2FastWrite16(ram,address,value);}
    if constexpr(sizeof(T)==4) {if(special)runtime->Store32(ram,ctx,address,value);else Ps2FastWrite32(ram,address,value);}
    if constexpr(sizeof(T)==8) {if(special)runtime->Store64(ram,ctx,address,value);else Ps2FastWrite64(ram,address,value);}
    if constexpr(sizeof(T)==16) {if(special)runtime->Store128(ram,ctx,address,value);else Ps2FastWrite128(ram,address,value);}
}

template<class T, bool Check>
inline void wotmPacketWrite(uint8_t *ram, R5900Context *ctx, PS2Runtime *runtime,
                           uint32_t address, T value) {
    uint64_t words[2]{};std::memcpy(words,&value,sizeof(value));
    ps2TraceGuestWrite(ram,address,sizeof(value),words[0],words[1],"native-packet",ctx);
    uint8_t *dest=wotmPacketPointer<T>(ram,runtime,address);
    // Store32 drains completed DMA handlers even for scratchpad destinations.
    // Keep that behavior; only stores without callbacks use the direct path.
    if constexpr(sizeof(T)==4) if(address-0x70000000u<PS2_SCRATCHPAD_SIZE) dest=nullptr;
    if(!dest) {wotmPacketWriteReference(ram,ctx,runtime,address,value);return;}
    T before;
    if constexpr(Check) std::memcpy(&before,dest,sizeof(before));
    std::memcpy(dest,&value,sizeof(value));
    if constexpr(Check) {
        T actual;std::memcpy(&actual,dest,sizeof(actual));
        std::memcpy(dest,&before,sizeof(before));
        wotmPacketWriteReference(ram,ctx,runtime,address,value);
        if(std::memcmp(&actual,dest,sizeof(actual))) {
            std::fprintf(stderr,"[ee:packets] write mismatch pc=%x addr=%x bytes=%zu\n",ctx->pc,address,sizeof(T));
            std::abort();
        }
        if(++s_packetWriteChecks==1 || s_packetWriteChecks%1000000ull==0)
            std::fprintf(stderr,"[ee:packets] writes=%llu mismatches=0\n",s_packetWriteChecks);
    }
}
}

#pragma push_macro("READ8")
#pragma push_macro("READ16")
#pragma push_macro("READ32")
#pragma push_macro("READ64")
#pragma push_macro("READ128")
#pragma push_macro("WRITE8")
#pragma push_macro("WRITE16")
#pragma push_macro("WRITE32")
#pragma push_macro("WRITE64")
#pragma push_macro("WRITE128")
#undef READ8
#undef READ16
#undef READ32
#undef READ64
#undef READ128
#undef WRITE8
#undef WRITE16
#undef WRITE32
#undef WRITE64
#undef WRITE128
#define READ8(a) wotmPacketRead<uint8_t,WOTM_PACKET_CHECK>(rdram,ctx,runtime,uint32_t(a))
#define READ16(a) wotmPacketRead<uint16_t,WOTM_PACKET_CHECK>(rdram,ctx,runtime,uint32_t(a))
#define READ32(a) wotmPacketRead<uint32_t,WOTM_PACKET_CHECK>(rdram,ctx,runtime,uint32_t(a))
#define READ64(a) wotmPacketRead<uint64_t,WOTM_PACKET_CHECK>(rdram,ctx,runtime,uint32_t(a))
#define READ128(a) wotmPacketRead<__m128i,WOTM_PACKET_CHECK>(rdram,ctx,runtime,uint32_t(a))
#define WRITE8(a,v) wotmPacketWrite<uint8_t,WOTM_PACKET_CHECK>(rdram,ctx,runtime,uint32_t(a),uint8_t(v))
#define WRITE16(a,v) wotmPacketWrite<uint16_t,WOTM_PACKET_CHECK>(rdram,ctx,runtime,uint32_t(a),uint16_t(v))
#define WRITE32(a,v) wotmPacketWrite<uint32_t,WOTM_PACKET_CHECK>(rdram,ctx,runtime,uint32_t(a),uint32_t(v))
#define WRITE64(a,v) wotmPacketWrite<uint64_t,WOTM_PACKET_CHECK>(rdram,ctx,runtime,uint32_t(a),uint64_t(v))
#define WRITE128(a,v) wotmPacketWrite<__m128i,WOTM_PACKET_CHECK>(rdram,ctx,runtime,uint32_t(a),v)
#define WOTM_PACKET_CHECK false
#define pktAddVu1ObjAsm_0x208410 wotmNativeObjectPacket
#include <pktAddVu1ObjAsm_0x208410.cpp>
#undef pktAddVu1ObjAsm_0x208410
#undef WOTM_PACKET_CHECK
#define WOTM_PACKET_CHECK true
#define pktAddVu1ObjAsm_0x208410 wotmCheckedObjectPacket
#include <pktAddVu1ObjAsm_0x208410.cpp>
#undef pktAddVu1ObjAsm_0x208410
#undef WOTM_PACKET_CHECK
#pragma pop_macro("READ8")
#pragma pop_macro("READ16")
#pragma pop_macro("READ32")
#pragma pop_macro("READ64")
#pragma pop_macro("READ128")
#pragma pop_macro("WRITE8")
#pragma pop_macro("WRITE16")
#pragma pop_macro("WRITE32")
#pragma pop_macro("WRITE64")
#pragma pop_macro("WRITE128")

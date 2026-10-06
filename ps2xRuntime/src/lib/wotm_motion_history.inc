#include "motion_provenance.inc"
// Opt-in source provenance/history audit for future temporal rendering.
// These are raw engine transforms, not yet a per-pixel motion-vector buffer.
#include <array>
#include <map>
#include <cmath>
#include <tuple>
namespace {
struct WotmMotionSample {
    uint64_t frame = 0;
    std::array<float,16> transform{}, camera{};
    uint32_t cameraIdentity=0;
    bool ambiguous = false;
    std::shared_ptr<std::atomic<bool>> valid=std::make_shared<std::atomic<bool>>(true);
};
using WotmMotionKey = std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>;
std::map<WotmMotionKey,WotmMotionSample> s_motionPrevious, s_motionCurrent;
PS2Runtime::RecompiledFunction s_motionPacketOriginal=nullptr;
std::FILE *s_motionCsv=nullptr;
uint64_t s_motionFrame=0, s_motionRows=0, s_motionMatched=0, s_motionAmbiguous=0;
constexpr uint64_t kMotionRowLimit=200000;
struct WotmPendingPacket {uint32_t builder=0,begin=0,returnPc=0;MotionProvenance::Object object;};
std::map<R5900Context*,WotmPendingPacket> s_motionPackets;
uint32_t motionRead(uint8_t* ram,uint32_t a) {
    uint32_t v=0;
    if(a<=PS2_RAM_SIZE-4u)std::memcpy(&v,ram+a,4);
    return v;
}
void motionFlush() {
    if(!s_motionCsv && !MotionProvenance::vectorsEnabled())return;
    for(const auto& p:s_motionCurrent) {
        if(!s_motionCsv || s_motionRows>=kMotionRowLimit)break;
        const auto& sample=p.second;
        auto old=s_motionPrevious.find(p.first);
        const bool match=old!=s_motionPrevious.end() && old->second.frame+1==sample.frame &&
                         !sample.ambiguous && !old->second.ambiguous &&
                         old->second.cameraIdentity==sample.cameraIdentity;
        s_motionMatched+=match;s_motionAmbiguous+=sample.ambiguous;
        const auto [view,slot,node,source]=p.first;
        std::fprintf(s_motionCsv,"%llu,%u,%u,%08x,%08x,%u,%u",sample.frame,view,slot,node,source,unsigned(match),unsigned(sample.ambiguous));
        for(float v:sample.transform)std::fprintf(s_motionCsv,",%.9g",v);
        // Invalid history is explicitly tagged; zero motion is not implied.
        for(float v:match?old->second.transform:sample.transform)std::fprintf(s_motionCsv,",%.9g",v);
        for(float v:sample.camera)std::fprintf(s_motionCsv,",%.9g",v);
        for(float v:match?old->second.camera:sample.camera)std::fprintf(s_motionCsv,",%.9g",v);
        std::fputc('\n',s_motionCsv);++s_motionRows;
    }
    if(s_motionCsv)std::fflush(s_motionCsv);
    s_motionPrevious=std::move(s_motionCurrent);s_motionCurrent.clear();
    if(s_motionCsv && s_motionRows>=kMotionRowLimit) {
        std::fclose(s_motionCsv);s_motionCsv=nullptr;
        if(!MotionProvenance::vectorsEnabled())s_motionPrevious.clear();
        std::fprintf(stderr,"[motion:history] complete rows=%llu matched=%llu ambiguous=%llu\n",s_motionRows,s_motionMatched,s_motionAmbiguous);
    }
}
void wotmResetMotion(const char* reason) {
    for(auto& x:s_motionCurrent)x.second.valid->store(false);
    for(auto& x:s_motionPrevious)x.second.valid->store(false);
    s_motionCurrent.clear();s_motionPrevious.clear();s_motionPackets.clear();s_motionFrame=0;
    MotionProvenance::resetHistory();
    std::fprintf(stderr,"[motion:reset] epoch=%llu reason=%s\n",MotionProvenance::resetSerial(),reason);
}
struct MotionResetHook {uint32_t begin,end;const char* reason;PS2Runtime::RecompiledFunction original=nullptr;};
MotionResetHook motionResetHooks[]={
    {0x2287f8u,0x22882cu,"world"}, {0x229068u,0x229070u,"view-layout"},
    {0x137e48u,0x137e94u,"camera-init"}, {0x11bc50u,0x11bea4u,"camera-pov"}};
void wotmMotionResetHook(uint8_t* ram,R5900Context* ctx,PS2Runtime* runtime){
    for(auto& h:motionResetHooks)if(ctx->pc>=h.begin && ctx->pc<h.end){
        if(ctx->pc==h.begin){
            bool changed=true;
            const uint32_t arg=getRegU32(ctx,4);
            if(h.begin==0x2287f8u)changed=motionRead(ram,0x446490u)!=arg;
            if(h.begin==0x229068u){const uint32_t gp=getRegU32(ctx,28);changed=gp>=0x6e10u && gp<PS2_RAM_SIZE && motionRead(ram,gp-0x6e10u)!=arg;}
            // SetCameraPOV retail entry uses A0=index, A1=POV; existing mode is
            // Cameras[index*0xeb0+0xcf4], base0x7c01f0 (11bc50..11bc84).
            if(h.begin==0x11bc50u)changed=arg<4 && motionRead(ram,0x7c01f0u+arg*0xeb0u+0xcf4u)!=getRegU32(ctx,5);
            if(changed)wotmResetMotion(h.reason);
        }
        h.original(ram,ctx,runtime);return;
    }
}
void wotmMotionFrame(uint64_t frame) {
    if(!s_motionCsv && !MotionProvenance::vectorsEnabled())return;
    if(frame<s_motionFrame)wotmResetMotion("frame-rewind");
    motionFlush();
    if(frame%120==0)std::fprintf(stderr,"[motion:history] frame=%llu rows=%llu matched=%llu ambiguous=%llu\n",frame,s_motionRows,s_motionMatched,s_motionAmbiguous);
}
void wotmMotionPacket(uint8_t* ram,R5900Context* ctx,PS2Runtime* runtime) {
    // Resume PCs must call the original without sampling clobbered entry regs.
    if((s_motionCsv || MotionProvenance::vectorsEnabled()) && ctx->pc==0x208410u) {
        s_motionPackets.erase(ctx);
        const uint64_t frame=g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed)+1;
        static const bool traceEntries=std::getenv("PS2X_MOTION_OWNER_TRACE")!=nullptr;
        if(traceEntries && frame==3001)
            std::fprintf(stderr,"[motion:builder-entry] begin=%x node=%x ra=%x builder=%x\n",
                runtime->memory().read32(getRegU32(ctx,4)+8u)&0x0fffffffu,
                getRegU32(ctx,21),getRegU32(ctx,31),getRegU32(ctx,4));
        if(frame>1) {
            if(frame!=s_motionFrame){s_motionFrame=frame;s_motionCurrent.clear();}
            const uint32_t node=getRegU32(ctx,21);
            if(node && node<=PS2_RAM_SIZE-0x6cu) {
                const uint32_t source=motionRead(ram,node+0x68u);
                // viewGetCurView at 0x2287b8; viewGetViewCs at 0x2287c0.
                // hierTraverseAsm saves GP at 0x20749c, then uses GP as a
                // scratch pointer until 0x2077f8 on its deferred-alpha path.
                // Only these verified call sites use that scratchpad layout.
                uint32_t globals=getRegU32(ctx,28);
                const uint32_t caller=getRegU32(ctx,31);
                if(getRegU32(ctx,16)==0x70000000u && (caller==0x2076c0u || caller==0x2077b8u))
                    globals=runtime->memory().read32(0x70000268u);
                const uint32_t view=globals>=0x6e0cu && globals<PS2_RAM_SIZE ? motionRead(ram,globals-0x6e0cu) : ~0u;
                static const bool traceOwners=std::getenv("PS2X_MOTION_OWNER_TRACE")!=nullptr;
                if(traceOwners && frame==3001 && globals!=getRegU32(ctx,28))
                    std::fprintf(stderr,"[motion:saved-gp] node=%x gp=%x saved=%x view=%u caller=%x\n",node,getRegU32(ctx,28),globals,view,caller);
                const uint32_t camera=view<4 ? motionRead(ram,0x6e1900u+view*20u) : 0;
                // S3's high half selects the matrix/cache slot in the original
                // packet builder. Include it to avoid merging distinct passes.
                const uint32_t slot=getRegU32(ctx,19)>>16;
                WotmMotionSample sample;sample.frame=frame;
                sample.cameraIdentity=camera;
                if(camera && camera<=PS2_RAM_SIZE-0x50u)
                    std::memcpy(sample.camera.data(),ram+camera+0x10u,64);
                const int registers[]={1,2,3,6};
                for(int i=0;i<4;++i)_mm_storeu_ps(sample.transform.data()+4*i,ctx->vu0_vf[registers[i]]);
                bool finite=camera && camera<=PS2_RAM_SIZE-0x50u && view<4;for(float f:sample.transform)finite&=std::isfinite(f);
                for(float f:sample.camera)finite&=std::isfinite(f);
                if(finite && s_motionCurrent.size()<8192) {
                    const WotmMotionKey key{view,slot,node,source};
                    auto [it,added]=s_motionCurrent.emplace(key,sample);
                    if(!added && (it->second.transform!=sample.transform || it->second.camera!=sample.camera)){it->second.ambiguous=true;it->second.valid->store(false);}
                    if(MotionProvenance::enabled()) {
                        WotmPendingPacket packet;
                        static unsigned packetEntries=0;
                        if(packetEntries++<3)std::fprintf(stderr,"[motion:entry] builder=%x ra=%x pc=%x\n",getRegU32(ctx,4),getRegU32(ctx,31),ctx->pc);
                        packet.builder=getRegU32(ctx,4);
                        packet.begin=runtime->memory().read32(packet.builder+8u)&0x0fffffffu;
                        packet.returnPc=getRegU32(ctx,31);
                        auto& o=packet.object;o.valid=it->second.valid;o.frame=frame;o.node=node;o.view=view;o.slot=slot;o.source=source;
                        auto old=s_motionPrevious.find(key);
                        o.candidate=old!=s_motionPrevious.end() && old->second.frame+1==frame &&
                            !old->second.ambiguous && !it->second.ambiguous && old->second.cameraIdentity==camera;
                        std::copy(sample.transform.begin(),sample.transform.end(),o.transforms.begin());
                        const auto& prev=o.candidate?old->second:sample;
                        std::copy(prev.transform.begin(),prev.transform.end(),o.transforms.begin()+16);
                        std::copy(sample.camera.begin(),sample.camera.end(),o.transforms.begin()+32);
                        std::copy(prev.camera.begin(),prev.camera.end(),o.transforms.begin()+48);
                        if(packet.begin<PS2_RAM_SIZE)
                            s_motionPackets[ctx]=packet;
                    }
                }
            }
        }
    }
    s_motionPacketOriginal(ram,ctx,runtime);
    if(MotionProvenance::enabled()) {
        auto pending=s_motionPackets.find(ctx);
        if(pending!=s_motionPackets.end() && ctx->pc==pending->second.returnPc) {
            const auto packet=pending->second;s_motionPackets.erase(pending);
            const uint32_t end=runtime->memory().read32(packet.builder+8u)&0x0fffffffu;
            // The END tag at end is a shared append sentinel, not this object's payload.
            if(end>packet.begin && end-packet.begin<=0x10000u)
                MotionProvenance::registerPacket(packet.begin,end,packet.object);
        }
    }
}
void wotmInstallMotionHistory(PS2Runtime& runtime) {
    const char* path=std::getenv("PS2X_MOTION_HISTORY_CSV");
    if((!path||!*path) && !MotionProvenance::vectorsEnabled())return;
    wotmResetMotion("install");
    for(auto& h:motionResetHooks){
        h.original=runtime.lookupFunction(h.begin);
        if(h.original)for(uint32_t pc=h.begin;pc<h.end;pc+=4)
            if(runtime.hasFunction(pc) && runtime.lookupFunction(pc)==h.original)runtime.replaceFunction(pc,&wotmMotionResetHook);
    }
    s_motionPacketOriginal=runtime.lookupFunction(0x208410u);
    if(!s_motionPacketOriginal)return;
    s_motionCsv=path && *path?std::fopen(path,"w"):nullptr;
    if(!s_motionCsv && !MotionProvenance::vectorsEnabled())return;
    if(s_motionCsv) {
    std::fprintf(s_motionCsv,"frame,view,matrix_slot,node,source,history_candidate,ambiguous");
    for(const char* prefix:{"current","previous"})for(int i=0;i<16;++i)std::fprintf(s_motionCsv,",%s_%d",prefix,i);
    for(const char* prefix:{"camera_current","camera_previous"})for(int i=0;i<16;++i)std::fprintf(s_motionCsv,",%s_%d",prefix,i);
    std::fputc('\n',s_motionCsv);
    std::fflush(s_motionCsv);
    }
    for(uint32_t pc=0x208410u;pc<0x20895cu;pc+=4u)
        if(runtime.hasFunction(pc)&&runtime.lookupFunction(pc)==s_motionPacketOriginal)
            runtime.replaceFunction(pc,&wotmMotionPacket);
    std::fprintf(stderr,"[motion:history] raw object transform audit enabled; not DLSS motion vectors\n");
}
}


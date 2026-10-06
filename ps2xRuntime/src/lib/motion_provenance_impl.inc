// Included only by ps2_memory.cpp. Snapshots travel with the owned DMA buffer,
// never with a pointer back into mutable guest memory.
#include "motion_provenance.inc"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
extern std::atomic<uint64_t> g_ps2xWotmCompletedFrames;
namespace MotionProvenance {
namespace {
std::mutex mutex;
struct Range {uint32_t end;std::shared_ptr<const Object> object;};
std::map<uint32_t,Range> ranges;
std::map<const uint8_t*,std::vector<Span>> transfers;
uint64_t rangeFrame=0;
std::atomic<uint64_t> epoch{1};
struct Context {Span span;uint32_t opcode=0,pc=0,commandOffset=0;uint64_t run=0;};
thread_local Context current,submitted;
thread_local uint64_t submittedImage=0;
thread_local std::map<const void*,Context> pending;
std::atomic<uint64_t> runSerial{0};
std::FILE* output=nullptr;
uint64_t rows=0;
}
uint64_t resetSerial(){return epoch.load(std::memory_order_acquire);}
void resetHistory(){
    std::lock_guard<std::mutex> lock(mutex);
    epoch.fetch_add(1,std::memory_order_acq_rel);ranges.clear();rangeFrame=0;
}
bool vectorsEnabled() {static const bool on=[] {const char* p=std::getenv("PS2X_D3D12_MOTION");const char* i=std::getenv("PS2X_FRAME_INTERPOLATION");return (p && p[0]=='1') || (i && i[0]=='1');}();return on;}
bool enabled() {
    static const bool on=[] {
        const char* path=std::getenv("PS2X_MOTION_PROVENANCE_CSV");
        if(!path||!*path)return vectorsEnabled();
        output=std::fopen(path,"w");
        if(!output)return false;
        std::fprintf(output,"run,kick,frame,node,view,slot,source,history_candidate,dma_tag,vif_opcode,start_pc,kick_pc,bytes\n");
        std::fflush(output);return true;
    }();return on;
}
void registerPacket(uint32_t begin,uint32_t end,const Object& object) {
    if(!enabled()||begin>=end||end>0x02000000u)return;
    std::lock_guard<std::mutex> lock(mutex);
    if(object.epoch!=resetSerial())return;
    if(rangeFrame!=object.frame){ranges.clear();rangeFrame=object.frame;}
    static const bool traceOwners=std::getenv("PS2X_MOTION_OWNER_TRACE")!=nullptr;
    if(traceOwners && object.frame>=3001 && object.frame<=3002)
        std::fprintf(stderr,"[motion:register] frame=%llu begin=%x end=%x node=%x\n",object.frame,begin,end,object.node);
    // Reused allocator addresses replace old ownership. No nearest-address or
    // previous-draw heuristic is allowed across unowned tags.
    auto it=ranges.lower_bound(begin);
    if(it!=ranges.begin()){auto prev=std::prev(it);if(prev->second.end>begin)it=prev;}
    while(it!=ranges.end()&&it->first<end)it=ranges.erase(it);
    static uint64_t registered=0;
    if(++registered==1)std::fprintf(stderr,"[motion:packet] first range=%x..%x node=%x frame=%llu\n",begin,end,object.node,object.frame);
    if(ranges.size()<8192)ranges.emplace(begin,Range{end,std::make_shared<const Object>(object)});
}
std::shared_ptr<const Object> lookup(uint32_t tag) {
    std::lock_guard<std::mutex> lock(mutex);
    if(rangeFrame!=g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed)+1)return {};
    auto it=ranges.upper_bound(tag);
    if(it==ranges.begin())return {};
    --it;return tag<it->second.end?it->second.object:nullptr;
}
void publish(const uint8_t* data,std::vector<Span>&& spans) {
    std::lock_guard<std::mutex> lock(mutex);
    // Bound abandoned transfers (e.g. a DMA reset). Losing provenance means
    // unknown ownership, never using another transfer's metadata.
    if(transfers.size()>=256)transfers.clear();
    transfers[data]=std::move(spans);
}
std::vector<Span> take(const uint8_t* data) {
    if(!enabled())return {};
    std::lock_guard<std::mutex> lock(mutex);
    auto it=transfers.find(data);if(it==transfers.end())return {};
    auto result=std::move(it->second);transfers.erase(it);return result;
}
Run::Run(const std::vector<Span>& spans,uint32_t offset,uint32_t opcode,uint32_t pc) {
    current={};current.opcode=opcode;current.pc=pc;
    current.run=runSerial.fetch_add(1,std::memory_order_relaxed)+1;
    auto it=std::upper_bound(spans.begin(),spans.end(),offset,
        [](uint32_t p,const Span& s){return p<s.begin;});
    if(it!=spans.begin()){--it;if(offset<it->end){current.span=*it;current.commandOffset=offset-it->begin;}}
}
Run::~Run(){current={};}
void issue(const void* interpreter){if(enabled())pending[interpreter]=current;}
void kick(const void* interpreter,uint64_t serial,uint32_t bytes,uint32_t pc,uint64_t image) {
    if(!enabled())return;
    submitted={};submittedImage=image;
    auto it=pending.find(interpreter);if(it==pending.end())return;
    const Context ctx=it->second;pending.erase(it);submitted=ctx;
    if(!output)return;
    // Intro/movie/menu traffic precedes RtLoopView game frames and would
    // exhaust the bounded capture before any object-history samples exist.
    if(g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed)<1)return;
    const Object empty{};const Object& o=ctx.span.object?*ctx.span.object:empty;
    std::lock_guard<std::mutex> lock(mutex);
    if(rows>=200000)return;
    std::fprintf(output,"%llu,%llu,%llu,%08x,%u,%u,%08x,%u,%08x,%u,%x,%x,%u\n",
        ctx.run,serial,o.frame,o.node,o.view,o.slot,o.source,unsigned(o.candidate),
        ctx.span.tag,ctx.opcode,ctx.pc,pc,bytes);
    if(++rows%4096==0)std::fflush(output);
}
}

#include "motion_vertices.inc"

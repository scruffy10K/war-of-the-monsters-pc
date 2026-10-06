// Verified native terminal meshes only. Identity comes from object + DMA CALL
// payload + original vertex index; never from the order meshes are submitted.
#include <cmath>
#include <cstring>
#include <tuple>
#include <unordered_map>
namespace MotionProvenance {
namespace {
using MeshKey=std::tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t>;
struct MeshKeyHash {
    size_t operator()(const MeshKey& key) const noexcept {
        uint64_t h=0x9e3779b97f4a7c15ull;
        std::apply([&](auto... words) {
            ((h^=uint64_t(words)+0x9e3779b97f4a7c15ull+(h<<6)+(h>>2)),...);
        },key);
        h^=h>>33;h*=0xff51afd7ed558ccdull;
        h^=h>>33;h*=0xc4ceb9fe1a85ec53ull;
        return size_t(h^(h>>33));
    }
};
struct Record {
    uint64_t frame=0;
    std::vector<Point> points;
    std::vector<uint8_t> adc;
    std::shared_ptr<std::atomic<bool>> unique,sourceValid;
    std::shared_ptr<Mesh> mesh;
};
using HashHistory=std::unordered_map<MeshKey,Record,MeshKeyHash>;
thread_local HashHistory vertexHistory;
thread_local std::map<MeshKey,Record> treeHistory;
thread_local std::shared_ptr<Mesh> active;
std::array<uint32_t,4> pointKey(Point p){std::array<uint32_t,4> k;std::memcpy(k.data(),&p,sizeof(p));return k;}
}
std::shared_ptr<Mesh> activeMesh(){return active;}
MeshScope::MeshScope(std::shared_ptr<Mesh> mesh):saved(std::move(active)){active=std::move(mesh);}
MeshScope::~MeshScope(){active=std::move(saved);}
std::array<float,4> previousVertex(const Mesh& mesh,Point point){
    if(!mesh.usable())return {};
    const auto key=pointKey(point);
    auto it=std::lower_bound(mesh.previous.begin(),mesh.previous.end(),key,
        [](const auto& entry,const auto& value){return entry.first<value;});
    return it==mesh.previous.end() || it->first!=key ? std::array<float,4>{} : it->second;
}
template<bool Hashed,class History>
static std::shared_ptr<Mesh> prepareMeshInHistory(History& history,const std::vector<Point>& points,const uint8_t* adc,uint64_t tag) {
    if(!vectorsEnabled())return {};
    static thread_local uint64_t historyEpoch=0;
    if(historyEpoch!=resetSerial()){history.clear();active.reset();historyEpoch=resetSerial();}
    if(submitted.span.object && submitted.span.object->epoch!=historyEpoch)return {};
    // Opt-in aggregate diagnostics; count input vertices rather than claiming
    // that accepted meshes represent visible pixel coverage.
    struct Audit {
        size_t vertices;unsigned reason=0;
        ~Audit(){
            static thread_local std::array<uint64_t,13> counts{};
            static thread_local uint64_t last=0;
            static const bool on=[] {const char* p=std::getenv("PS2X_MOTION_AUDIT");return p&&p[0]=='1';}();
            if(!on)return;
            counts[reason]+=vertices;
            const auto frame=g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);
            if(frame<last || frame>=last+120){
                std::fprintf(stderr,"[motion:audit] frame=%llu accepted=%llu owner=%llu signature=%llu shape=%llu image=%llu finite=%llu duplicate=%llu new=%llu gap=%llu source=%llu topology=%llu discontinuity=%llu repeated=%llu\n",frame,counts[0],counts[1],counts[2],counts[3],counts[4],counts[5],counts[6],counts[7],counts[8],counts[9],counts[10],counts[11],counts[12]);
                counts={};last=frame;
            }
        }
    } audit{points.size()};
    if(!submitted.span.object){
        audit.reason=1;
        static std::FILE* unowned=[] {const char* p=std::getenv("PS2X_MOTION_UNOWNED_CSV");auto* f=p&&*p?std::fopen(p,"w"):nullptr;if(f)std::fputs("frame,tag,begin,end,command_offset,opcode,start_pc,image,vertices,root_tag\n",f);return f;}();
        static uint32_t rows=0;
        const uint64_t frame=g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);
        if(unowned && frame>=3000 && rows<8192){
            std::fprintf(unowned,"%llu,%x,%u,%u,%u,%u,%x,%llx,%zu,%x\n",frame,submitted.span.tag,submitted.span.begin,submitted.span.end,submitted.commandOffset,submitted.opcode,submitted.pc,submittedImage,points.size(),submitted.span.rootTag);
            if(++rows%512==0)std::fflush(unowned);
            if(rows==8192){std::fclose(unowned);unowned=nullptr;}
        }
        // Metadata only: blocker meshes have no ordinary object history.
        // The renderer must still verify the shadow composite's blend/depth state.
        if(submitted.span.rootTag>=0x544800u && submitted.span.rootTag<0x54c800u){
            auto shadow=std::make_shared<Mesh>();shadow->shadow=true;return shadow;
        }
        return {};
    }
    if(!submitted.span.signature){audit.reason=2;return {};}
    if(points.size()<3 || points.size()>341){audit.reason=3;return {};}
    const auto& o=*submitted.span.object;
    if(submittedImage!=0x7d7edbc78086ac63ull && submittedImage!=0xd166d271aef9b896ull){audit.reason=4;return {};}
    for(const auto& p:points)if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)||!std::isfinite(p.q)){audit.reason=5;return {};}
    const MeshKey key{o.view,o.slot,o.node,o.source,submitted.span.tag,submitted.span.signature,submittedImage,tag,submitted.commandOffset,submitted.opcode,submitted.pc};
    auto found=history.find(key);
    auto result=std::make_shared<Mesh>();result->frame=o.frame;result->unique=std::make_shared<std::atomic<bool>>(true);result->sourceValid=o.valid;
    if(found!=history.end() && found->second.frame==o.frame) {
        const auto& old=found->second;
        // Repeated render passes with byte-identical projected geometry have
        // the same correspondence. Share validity so any later conflicting
        // submission still invalidates every earlier copy of this mesh.
        const bool identical=old.sourceValid==o.valid && old.points.size()==points.size() &&
            std::equal(old.adc.begin(),old.adc.end(),adc) &&
            std::equal(points.begin(),points.end(),old.points.begin(),
                [](Point a,Point b){return pointKey(a)==pointKey(b);});
        if(identical){audit.reason=12;return old.mesh;}
        audit.reason=6;found->second.unique->store(false);result->unique->store(false);return result;
    }
    bool valid=found!=history.end() && found->second.frame+1==o.frame && o.candidate &&
        found->second.points.size()==points.size() && found->second.unique->load() &&
        found->second.sourceValid && found->second.sourceValid->load() && o.valid && o.valid->load() &&
        std::equal(found->second.adc.begin(),found->second.adc.end(),adc);
    if(found==history.end())audit.reason=7;
    else if(found->second.frame+1!=o.frame)audit.reason=8;
    else if(!o.candidate || !found->second.unique->load() || !found->second.sourceValid || !found->second.sourceValid->load() || !o.valid || !o.valid->load())audit.reason=9;
    else if(!valid)audit.reason=10;
    // Conservative cut/teleport guard. Final temporal integration will also
    // consume an explicit engine reset; do not smear large discontinuities.
    if(valid)for(size_t i=0;i<points.size();++i){
        const auto& a=points[i];const auto& b=found->second.points[i];
        // Q<=0 is behind/on the projection plane. It invalidates incident
        // triangles, not every unrelated triangle in this source mesh.
        if(a.q<=0 || b.q<=0)continue;
        const float ratio=a.q/b.q;
        if(std::abs(a.x-b.x)>128 || std::abs(a.y-b.y)>64 || ratio<0.25f || ratio>4.f){audit.reason=11;valid=false;break;}
    }
    if(valid) {
        result->previousValid=found->second.sourceValid;
        result->previous.reserve(points.size());
        for(size_t i=0;i<points.size();++i){
            const auto& a=points[i];const auto& b=found->second.points[i];
            std::array<float,4> prev{};
            if(a.q>0 && b.q>0){
                const float ratio=a.q/b.q;
                if(std::isfinite(b.x*ratio) && std::isfinite(b.y*ratio) && std::isfinite(ratio))
                    prev={b.x*ratio,b.y*ratio,ratio,1};
            }
            // Keep invalid entries too: an invalid previous vertex may share
            // its current position with a valid one; both must then be rejected.
            result->previous.emplace_back(pointKey(a),prev);
        }
        static const bool verifyCompact=[] {
            const char* p=std::getenv("PS2X_MOTION_COMPACT_VERIFY");return p && p[0]=='1';
        }();
        static thread_local uint64_t compactCalls=0,compactChecks=0;
        const bool checkCompact=verifyCompact && (++compactCalls&255u)==0;
        std::map<std::array<uint32_t,4>,std::array<float,4>> reference;
        if(checkCompact)for(const auto& entry:result->previous) {
            auto [it,added]=reference.emplace(entry);
            if(!added && it->second!=entry.second)it->second={};
        }
        // Stable ties retain the original first value, including signed zero.
        // Apply the same sticky rejection as the original per-vertex map.
        std::stable_sort(result->previous.begin(),result->previous.end(),
            [](const auto& a,const auto& b){return a.first<b.first;});
        size_t used=0;
        for(size_t i=0;i<result->previous.size();++i) {
            const auto entry=result->previous[i];
            if(used && result->previous[used-1].first==entry.first) {
                if(result->previous[used-1].second!=entry.second)result->previous[used-1].second={};
            } else result->previous[used++]=entry;
        }
        result->previous.resize(used);
        if(checkCompact) {
            auto it=reference.begin();bool same=reference.size()==result->previous.size();
            for(const auto& entry:result->previous) {
                if(it==reference.end()){same=false;break;}
                same &= it->first==entry.first && std::memcmp(it->second.data(),entry.second.data(),16)==0;
                ++it;
            }
            if(!same){std::fprintf(stderr,"[motion:compact] MISMATCH\n");std::abort();}
            if((++compactChecks&127u)==0)
                std::fprintf(stderr,"[motion:compact] checked-meshes=%llu mismatches=0\n",compactChecks);
        }
    }
    if(history.size()>=16384){
        // The found record can itself expire during pruning. No iterator is
        // retained across its erase; other erasures keep it valid.
        if constexpr(Hashed)
            if(found!=history.end() && found->second.frame+1<o.frame)found=history.end();
        for(auto it=history.begin();it!=history.end();) {
            if(it->second.frame+1<o.frame)it=history.erase(it);else ++it;
        }
        if(history.size()>=16384){result->unique->store(false);return result;}
    }
    if constexpr(Hashed) {
        Record next{o.frame,points,std::vector<uint8_t>(adc,adc+points.size()),result->unique,o.valid,result};
        if(found!=history.end())found->second=std::move(next);
        else history.emplace(key,std::move(next));
    } else {
        history[key]={o.frame,points,std::vector<uint8_t>(adc,adc+points.size()),result->unique,o.valid,result};
    }
    return result;
}
// Both paths share all admission, motion math and validity rules. The tree
// remains available for diagnostics; only the chosen worker-local cache fills.
std::shared_ptr<Mesh> prepareMesh(const std::vector<Point>& points,const uint8_t* adc,uint64_t tag) {
    static const bool hashed=[] {
        const char* p=std::getenv("PS2X_MOTION_HISTORY_HASH");return !p || p[0]!='0';
    }();
    static thread_local const bool logged=[&] {
        const char* p=std::getenv("PS2X_MOTION_HISTORY_AUDIT");
        if(p && p[0]=='1')std::fprintf(stderr,"[motion:history] exact-key-%s\n",hashed?"hash":"tree");
        return true;
    }();
    (void)logged;
    if(hashed)return prepareMeshInHistory<true>(vertexHistory,points,adc,tag);
    return prepareMeshInHistory<false>(treeHistory,points,adc,tag);
}
}

// Opt-in attribution only. Original functions run at entry and resume PCs.
// Inclusive scope times must not be added: e.g. game_update can contain debris.
namespace {
struct WotmDestructionScope {
    uint32_t address;
    const char *name;
    PS2Runtime::RecompiledFunction original=nullptr;
    uint64_t calls=0, nonReturns=0, instructions=0, runs=0, segments=0;
    double milliseconds=0;
};
WotmDestructionScope s_destructionScopes[] = {
    {0x12f270u,"debris_activate"}, {0x12f810u,"debris_cull"},
    {0x12f940u,"debris_update"}, {0x18a068u,"rigid_debris_update"},
    {0x1306e0u,"destructible_hit"}, {0x130b90u,"destructible_change"},
    {0x130d80u,"destructible_switch"}, {0x1372b8u,"game_update"},
    {0x1376a8u,"game_update2"}, {0x2029b0u,"scene_hier"},
    {0x209d88u,"scene_flush"}, {0x202f40u,"scene_world"},
    {0x202fd8u,"scene_objects"}, {0x2030c0u,"scene_hp_objects"},
    {0x212390u,"fragment_packets"}
};
std::FILE *s_destructionCsv=nullptr;
uint64_t s_destructionFirst=1800, s_destructionLast=3200;
// Persistent section boundaries include child continuations that resume outside
// their parent's native C++ stack. These are wall spans (including any waits),
// unlike the inclusive function-segment timers above. VU deltas identify work.
struct WotmSceneSection {
    double milliseconds=0;
    uint64_t instructions=0, runs=0;
};
WotmSceneSection s_sceneSections[4];
unsigned s_sceneSection=0;
uint32_t s_sceneReturn[3]{};
auto s_sceneSectionStart=std::chrono::steady_clock::now();
uint64_t s_sceneInstructions=0,s_sceneRuns=0;
void wotmSceneSection(unsigned next) {
    const auto now=std::chrono::steady_clock::now();
    const auto instructions=g_ps2xVuInstr.load(std::memory_order_relaxed);
    const auto runs=g_ps2xVuRuns.load(std::memory_order_relaxed);
    auto &s=s_sceneSections[s_sceneSection];
    s.milliseconds+=std::chrono::duration<double,std::milli>(now-s_sceneSectionStart).count();
    s.instructions+=instructions-s_sceneInstructions;s.runs+=runs-s_sceneRuns;
    s_sceneSection=next;s_sceneSectionStart=now;s_sceneInstructions=instructions;s_sceneRuns=runs;
}

template<unsigned Index>
void wotmDestructionScope(uint8_t *rdram,R5900Context *ctx,PS2Runtime *runtime) {
    auto &s=s_destructionScopes[Index];
    const auto frame=g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed)+1;
    if(frame<s_destructionFirst || frame>s_destructionLast) {
        s.original(rdram,ctx,runtime);
        return;
    }
    const uint32_t returnPc=getRegU32(ctx,31);
    const bool entry=ctx->pc==s.address;
    if constexpr(Index>=11 && Index<=13) {
        if(entry) {
            s_sceneReturn[Index-11]=returnPc;
            wotmSceneSection(Index-10);
        }
    }
    const auto instructions=g_ps2xVuInstr.load(std::memory_order_relaxed);
    const auto runs=g_ps2xVuRuns.load(std::memory_order_relaxed);
    const auto start=std::chrono::steady_clock::now();
    s.original(rdram,ctx,runtime);
    if constexpr(Index>=11 && Index<=13) {
        if(ctx->pc==s_sceneReturn[Index-11]) wotmSceneSection(0);
    }
    s.milliseconds+=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start).count();
    s.calls+=entry;
    ++s.segments;
    s.nonReturns+=(ctx->pc!=returnPc);
    s.instructions+=g_ps2xVuInstr.load(std::memory_order_relaxed)-instructions;
    s.runs+=g_ps2xVuRuns.load(std::memory_order_relaxed)-runs;
}

void wotmDestructionFrame(uint64_t frame) {
    if(!s_destructionCsv) return;
    wotmSceneSection(s_sceneSection);
    if(frame>=s_destructionFirst && frame<=s_destructionLast) {
        for(auto &s:s_destructionScopes) {
            std::fprintf(s_destructionCsv,"%llu,%s,%llu,%.6f,%llu,%llu,%llu,%llu\n",
                frame,s.name,s.calls,s.milliseconds,s.instructions,s.runs,s.nonReturns,s.segments);
        }
        static const char *names[]={"section_other_wall","section_world_wall","section_objects_wall","section_hp_objects_wall"};
        for(unsigned i=0;i<4;++i) {
            const auto &s=s_sceneSections[i];
            std::fprintf(s_destructionCsv,"%llu,%s,0,%.6f,%llu,%llu,0,0\n",
                frame,names[i],s.milliseconds,s.instructions,s.runs);
        }
        if(frame%60u==0u) std::fflush(s_destructionCsv);
    }
    for(auto &s:s_destructionScopes) {
        s.calls=s.nonReturns=s.instructions=s.runs=s.segments=0;
        s.milliseconds=0;
    }
    for(auto &s:s_sceneSections)s={};
    if(frame>=s_destructionLast) {
        std::fclose(s_destructionCsv);
        s_destructionCsv=nullptr;
        std::fprintf(stderr,"[destruction:profile] complete frame=%llu\n",frame);
    }
}

void wotmInstallDestructionProfile(PS2Runtime &runtime) {
    const char *path=std::getenv("PS2X_DESTRUCTION_CSV");
    if(!path || !*path) return;
    if(const char *p=std::getenv("PS2X_DESTRUCTION_START")) s_destructionFirst=std::strtoull(p,nullptr,10);
    if(const char *p=std::getenv("PS2X_DESTRUCTION_END")) s_destructionLast=std::strtoull(p,nullptr,10);
    const PS2Runtime::RecompiledFunction wrappers[] = {
        &wotmDestructionScope<0>, &wotmDestructionScope<1>, &wotmDestructionScope<2>,
        &wotmDestructionScope<3>, &wotmDestructionScope<4>, &wotmDestructionScope<5>,
        &wotmDestructionScope<6>, &wotmDestructionScope<7>, &wotmDestructionScope<8>,
        &wotmDestructionScope<9>, &wotmDestructionScope<10>, &wotmDestructionScope<11>,
        &wotmDestructionScope<12>, &wotmDestructionScope<13>, &wotmDestructionScope<14>
    };
    for(auto &s:s_destructionScopes) {
        s.original=runtime.hasFunction(s.address)?runtime.lookupFunction(s.address):nullptr;
        if(!s.original) {
            std::fprintf(stderr,"[destruction:profile] missing %s at %08x\n",s.name,s.address);
            return;
        }
    }
    s_destructionCsv=std::fopen(path,"w");
    if(!s_destructionCsv) return;
    std::fprintf(s_destructionCsv,"frame,scope,calls,inclusive_ms,vu_instructions,vu_runs,non_returns,segments\n");
    // Dense table contains continuation PCs. Hook matching original pointers
    // throughout the game's code range, otherwise a scheduler yield silently
    // excludes the rest of a long scene traversal from the measurements.
    // Preserve ctx->pc: the original function's switch resumes at that PC.
    for(uint32_t pc=0x100000u;pc<0x260000u;pc+=4u) {
        if(!runtime.hasFunction(pc)) continue;
        const auto original=runtime.lookupFunction(pc);
        for(unsigned i=0;i<sizeof(wrappers)/sizeof(wrappers[0]);++i)
            if(original==s_destructionScopes[i].original) {
                runtime.replaceFunction(pc,wrappers[i]);
                break;
            }
    }
    std::fprintf(stderr,"[destruction:profile] enabled frames=%llu..%llu\n",s_destructionFirst,s_destructionLast);
}
}

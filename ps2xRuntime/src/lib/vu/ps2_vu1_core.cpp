#include "../motion_provenance.inc"
#include "runtime/ps2_vu1.h"
#include "runtime/ps2_perf_clock.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"
#include "ps2_vu1_decode_inl.h"
#include "ps2_vu1_native.h"
#include <native_parameters.inc>

#include <algorithm>
#include <bit>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
#include <intrin.h>
#endif
#include <atomic>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <type_traits>
#include <ps2_log.h>
#include "ps2_vu1_flag_retire.inc"
#include "ps2_vu1_image_cache.inc"

// The XGKICK packet currently being handed to the GS (game thread only). Lets
// the GS-side PS2X_BIGTRI diagnostic attribute a bad primitive to the VU1
// program, PC and VU memory address that produced it. packet is non-null only
// while finishXgkick is submitting.
struct Ps2xVu1KickContext
{
    const uint8_t *packet;
    uint32_t bytes, startPc, pc, src, top, itop;
    uint64_t serial;
};
Ps2xVu1KickContext g_ps2xVu1Kick{};
// FNV-1a of the current VU1 code image (set in rebuildDecodedCodeCache); the
// native program table (ps2_vu1_native.h) is keyed on it.
Vu1NativeStep ps2xVu1BlockPrototype(VU1Interpreter &, Vu1NativeCtx &);
Vu1NativeStep ps2xVu1HeavyBlock(VU1Interpreter &, Vu1NativeCtx &, Vu1NativeProgram);
Vu1NativeStep (*g_ps2xVu1FusedTransform)(VU1Interpreter &, Vu1NativeCtx &)=nullptr;
Vu1NativeStep (*g_ps2xVu1ShaderTransform)(VU1Interpreter &, Vu1NativeCtx &)=nullptr;
bool (*g_ps2xVu1MenuVertex)(VU1Interpreter &, Vu1NativeCtx &)=nullptr;
bool (*g_ps2xVu1MenuLoop)(VU1Interpreter &, Vu1NativeCtx &)=nullptr;
Vu1NativeStep (*g_ps2xVu1FusedTransformSecond)(VU1Interpreter &, Vu1NativeCtx &)=nullptr;
uint64_t g_ps2xVu1FusedHits=0,g_ps2xVu1FusedFallbacks=0;
Vu1NativeStep (*g_ps2xVu1ConvertPair[2])(VU1Interpreter &, Vu1NativeCtx &)={nullptr,nullptr};
uint64_t g_ps2xVu1ObjectHits[4]{},g_ps2xVu1ObjectFallbacks[3]{};
Vu1NativeStep (*g_ps2xVu1ObjectKernel[4])(VU1Interpreter &,Vu1NativeCtx &)={};
Vu1NativeStep (*g_ps2xVu1GameMatrix[3])(VU1Interpreter &,Vu1NativeCtx &)={};
uint64_t g_ps2xVu1MatrixHits=0,g_ps2xVu1MatrixFallbacks=0;
static uint64_t s_vu1ImageHash = 0u;
static const bool s_entryTimingEnabled = std::getenv("PS2X_VU1_ENTRY_TIMING") != nullptr;
static thread_local uint64_t s_entryGifTicks = 0u;
struct Ps2xVu1EntryStat { std::atomic<uint64_t> ticks{0u}, cycles{0u}, calls{0u}, gifTicks{0u}; };
static uint64_t s_entryImages[8]{};
static Ps2xVu1EntryStat s_entryStats[8][2048];
static void ps2xVu1EntryNote(uint64_t image, uint32_t pc, uint64_t ticks, uint64_t cycles, uint64_t gifTicks)
{
    int k = 0;
    for (; k < 8; ++k)
    {
        if (s_entryImages[k] == image) break;
        if (s_entryImages[k] == 0u) { s_entryImages[k] = image; break; }
    }
    if (k == 8) return;
    Ps2xVu1EntryStat &e = s_entryStats[k][(pc / 8u) & 2047u];
    e.ticks.fetch_add(ticks, std::memory_order_relaxed);
    e.cycles.fetch_add(cycles, std::memory_order_relaxed);
    e.calls.fetch_add(1u, std::memory_order_relaxed);
    e.gifTicks.fetch_add(gifTicks, std::memory_order_relaxed);
}
void ps2xVu1EntryReport(double wall)
{
    struct Row { double share; uint64_t image; uint32_t pc; uint64_t calls, cycles, ticks, gifTicks; };
    std::vector<Row> rows;
    uint64_t total = 0u;
    for (int k = 0; k < 8; ++k)
        for (uint32_t i = 0; i < 2048u; ++i)
        {
            Ps2xVu1EntryStat &e = s_entryStats[k][i];
            const uint64_t tk = e.ticks.exchange(0u), cy = e.cycles.exchange(0u), ca = e.calls.exchange(0u);
            const uint64_t gif = e.gifTicks.exchange(0u);
            if (ca) { rows.push_back({0.0, s_entryImages[k], i * 8u, ca, cy, tk, gif}); total += tk; }
        }
    std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) { return a.ticks > b.ticks; });
    for (size_t i = 0; i < rows.size() && i < 8u; ++i)
        std::fprintf(stderr, "[vu1:entries] %4.1f%% img=%04llx pc=0x%04x calls/s=%.0f cycles/call=%.0f ticks/cycle=%.1f gif=%.1f%%\n",
                     100.0 * rows[i].ticks / std::max<uint64_t>(1u, total), (unsigned long long)(rows[i].image >> 48), rows[i].pc,
                     rows[i].calls / wall, double(rows[i].cycles) / rows[i].calls, double(rows[i].ticks) / std::max<uint64_t>(1u, rows[i].cycles),
                     100.0 * rows[i].gifTicks / std::max<uint64_t>(1u, rows[i].ticks));
}
// VU0 micro-code image hash (native VU0 programs are looked up by it too).
static uint64_t s_vu0ImageHash = 0u;
// PS2X_VU0_NATIVE_VERIFY support: forces the next VU0 run through the
// interpreter so ps2_runtime.cpp can compare it with the native program.
thread_local bool t_vu0ForceInterpreter = false;
namespace {
// Power-of-two scaling is exact in the representable conversion range.
// Clamp before scaling to avoid host overflow, then saturate positive lanes.
template<unsigned Scale>
__m128i vu1ConvertVector(const float *source) {
    const __m128 value=_mm_castsi128_ps(vu1n::vuLoadNormalized(source));
    const __m128 lower=_mm_set1_ps(-2147483648.0f/Scale);
    const __m128 upper=_mm_set1_ps(2147483520.0f/Scale);
    const __m128 clipped=_mm_min_ps(_mm_max_ps(value,lower),upper);
    const __m128 scaled=_mm_mul_ps(clipped,_mm_set1_ps(float(Scale)));
    const __m128i positive=_mm_castps_si128(_mm_cmpge_ps(value,_mm_set1_ps(2147483648.0f/Scale)));
    return _mm_or_si128(_mm_cvttps_epi32(scaled),_mm_and_si128(positive,_mm_set1_epi32(0x7fffffff)));
}
struct VuBenchHeader {
    uint64_t magic=0x3152435531565350ull, image=0;
    uint32_t objectSize=sizeof(VU1Interpreter), codeSize=0, dataSize=0, maxCycles=0;
};
struct VuBenchFile { VuBenchHeader header; std::vector<uint8_t> object,code,data; };
thread_local VuBenchFile *t_vuBenchReplay=nullptr;
thread_local bool t_vuBenchPassed=false;
thread_local bool t_qwordReference=false;
thread_local bool t_nativeStoreReference=false;
const bool s_nativeStores=[] {const char *p=std::getenv("PS2X_VU1_NATIVE_STORES");return !p || p[0]!='0';}();
const bool s_nativeTransferStores=[] {const char *p=std::getenv("PS2X_VU1_NATIVE_TRANSFER_STORES");return !p || p[0]!='0';}();
const bool s_batchFlags=[] {const char *p=std::getenv("PS2X_VU1_BATCH_FLAGS");return !p || p[0]!='0';}();
const bool s_batchFlagsVerify=std::getenv("PS2X_VU1_BATCH_FLAGS_VERIFY")!=nullptr;
// Read launch configuration once, outside the per-cycle retirement path.
const bool s_fastPipelineStore=[] {
    const char *p=std::getenv("PS2X_VU1_STORE_COPY");
    return !p || p[0]!='0';
}();
thread_local std::vector<uint8_t> t_stretchInput;
thread_local std::vector<uint8_t> t_stretchMeshInput;
thread_local bool t_stretchSaved=false;
const char *s_stretchPath=std::getenv("PS2X_STRETCH_CHECKPOINT");
struct VuEntryProfile {
    struct Counts { uint64_t calls=0,pairs=0; };
    std::map<std::pair<uint64_t,uint32_t>,Counts> counts;
    uint64_t pairs=0,nextDump=250000000;
    ~VuEntryProfile() { dump(); }
    void dump() {
        std::vector<std::pair<std::pair<uint64_t,uint32_t>,Counts>> sorted(counts.begin(),counts.end());
        std::sort(sorted.begin(),sorted.end(),[](const auto &a,const auto &b){return a.second.pairs>b.second.pairs;});
        uint64_t total=0;for(const auto &row:sorted)total+=row.second.pairs;
        for(size_t i=0;i<std::min<size_t>(sorted.size(),20);++i) {
            const auto &row=sorted[i];
            std::fprintf(stderr,"[vu1:entry] image=%016llx pc=0x%x shader=0x%x calls=%llu pairs=%llu share=%.2f%%\n",
                static_cast<unsigned long long>(row.first.first),row.first.second&0xffffu,row.first.second>>16,
                static_cast<unsigned long long>(row.second.calls),static_cast<unsigned long long>(row.second.pairs),
                total?100.0*row.second.pairs/total:0);
        }
    }
};

}
// Called synchronously while the GS decodes this run's XGKICK packet.
void ps2xSaveStretchCheckpoint()
{
    if(!s_stretchPath || t_stretchSaved || t_stretchInput.empty() || !g_ps2xVu1Kick.packet || g_ps2xVu1Kick.startPc!=0x20)return;
    t_stretchSaved=true;
    if(std::filesystem::exists(s_stretchPath)){std::fprintf(stderr,"[stretch:checkpoint] existing path, skipped\n");return;}
    if(auto *f=std::fopen(s_stretchPath,"wb")) {
        const bool ok=std::fwrite(t_stretchInput.data(),1,t_stretchInput.size(),f)==t_stretchInput.size();std::fclose(f);
        std::fprintf(stderr,"[stretch:checkpoint] %s kick=%llu src=%x path=%s\n",ok?"saved":"FAILED",g_ps2xVu1Kick.serial,g_ps2xVu1Kick.src,s_stretchPath);
        const auto meshPath=std::string(s_stretchPath)+".mesh.vu";
        if(!t_stretchMeshInput.empty())if(auto *mesh=std::fopen(meshPath.c_str(),"wb")){std::fwrite(t_stretchMeshInput.data(),1,t_stretchMeshInput.size(),mesh);std::fclose(mesh);}
        const auto packetPath=std::string(s_stretchPath)+".gif";
        if(auto *packet=std::fopen(packetPath.c_str(),"wb")){std::fwrite(g_ps2xVu1Kick.packet,1,g_ps2xVu1Kick.bytes,packet);std::fclose(packet);}
    }
}
int ps2xReplayVu1Capture(const char *path)
{
    static_assert(std::is_trivially_copyable_v<VU1Interpreter>);
    VuBenchFile f;
    FILE *in=std::fopen(path,"rb");
    if(!in)return 2;
    bool valid=std::fread(&f.header,sizeof(f.header),1,in)==1 &&
        f.header.magic==0x3152435531565350ull && f.header.objectSize==sizeof(VU1Interpreter) &&
        f.header.codeSize==16384 && f.header.dataSize==16384 && f.header.maxCycles>0 && f.header.maxCycles<=16777216;
    if(valid) {
        f.object.resize(f.header.objectSize);f.code.resize(f.header.codeSize);f.data.resize(f.header.dataSize);
        valid=std::fread(f.object.data(),1,f.object.size(),in)==f.object.size() &&
            std::fread(f.code.data(),1,f.code.size(),in)==f.code.size() &&
            std::fread(f.data.data(),1,f.data.size(),in)==f.data.size() && std::fgetc(in)==EOF;
    }
    std::fclose(in);
    uint64_t hash=1469598103934665603ull;
    for(uint8_t b:f.code){hash^=b;hash*=1099511628211ull;}
    if(!valid || hash!=f.header.image || !ps2xVu1NativeLookup(hash)) {
        std::fprintf(stderr,"[vu1:replay] invalid or incompatible checkpoint\n");return 2;
    }
    // GS is an inert callback target: capture-only execution never submits.
    GS gs;
    auto v=std::make_unique<VU1Interpreter>();
    t_vuBenchPassed=false;t_vuBenchReplay=&f;
    v->execute(f.code.data(),f.header.codeSize,f.data.data(),f.header.dataSize,gs,nullptr);
    t_vuBenchReplay=nullptr;
    return t_vuBenchPassed?0:1;
}

// [perf] visibility: how many VU1 program runs took the native path vs the
// interpreter loop, and how many native runs handed back mid-program.
std::atomic<uint64_t> g_ps2xVu1NativeRuns{0u};
std::atomic<uint64_t> g_ps2xVu1InterpRuns{0u};
std::atomic<uint64_t> g_ps2xVu1NativeFallbacks{0u};

// PS2X_VU1_FLIGHT=1: VU1 flight recorder. A ring of the last 1024 executed
// pairs (PC, words, cycle, Q, flags and the values the pair wrote) plus
// XGKICK markers, dumped to vu1_flight.txt by ps2xVu1FlightDump() -- called
// by the GS-side PS2X_BIGTRI detector when a drawn primitive has a garbage
// vertex, so the computation that produced it can be followed step by step.
struct Vu1FlightEntry
{
    uint32_t pc, upper, lower;
    uint64_t cycle;
    float q, i;
    uint32_t clip, mac, status;
    uint8_t uReg, lReg, viReg, accW;
    float uVal[4], lVal[4], acc[4];
    int32_t viVal;
};
namespace
{
    const bool s_vu1Flight = std::getenv("PS2X_VU1_FLIGHT") != nullptr;
    constexpr uint32_t kVu1FlightSize = 1024u;
    Vu1FlightEntry s_vu1FlightRing[kVu1FlightSize];
    uint32_t s_vu1FlightPos = 0u;
}

void ps2xVu1FlightDump(const char *reason)
{
    if (!s_vu1Flight)
        return;
    static uint32_t s_dumps = 0u;
    std::FILE *f = std::fopen("vu1_flight.txt", s_dumps == 0u ? "w" : "a");
    ++s_dumps;
    if (!f)
        return;
    std::fprintf(f, "##### dump %u: %s\n", s_dumps, reason ? reason : "");
    const uint32_t count = std::min<uint32_t>(s_vu1FlightPos, kVu1FlightSize);
    for (uint32_t n = 0; n < count; ++n)
    {
        const Vu1FlightEntry &e = s_vu1FlightRing[(s_vu1FlightPos - count + n) % kVu1FlightSize];
        if (e.pc == 0xFFFFFFFFu)
        {
            std::fprintf(f, "---- XGKICK done serial=%llu src=0x%x bytes=%u\n",
                         static_cast<unsigned long long>(e.cycle), e.upper, e.lower);
            continue;
        }
        std::fprintf(f, "%04x c=%llu U=%08x L=%08x Q=%.8g I=%.8g clip=%06x mac=%04x st=%03x",
                     e.pc, static_cast<unsigned long long>(e.cycle), e.upper, e.lower, e.q, e.i,
                     e.clip & 0xFFFFFFu, e.mac & 0xFFFFu, e.status & 0xFFFu);
        if (e.uReg)
            std::fprintf(f, " vf%u=(%.8g,%.8g,%.8g,%.8g)", e.uReg, e.uVal[0], e.uVal[1], e.uVal[2], e.uVal[3]);
        if (e.accW)
            std::fprintf(f, " acc=(%.8g,%.8g,%.8g,%.8g)", e.acc[0], e.acc[1], e.acc[2], e.acc[3]);
        if (e.lReg)
            std::fprintf(f, " Lvf%u=(%.8g,%.8g,%.8g,%.8g)", e.lReg, e.lVal[0], e.lVal[1], e.lVal[2], e.lVal[3]);
        if (e.viReg)
            std::fprintf(f, " vi%u=%d", e.viReg, e.viVal);
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
// [vu1:prof] nested bucket: time spent handing finished XGKICK packets to
// the GIF/GS frontend (packet parse + raster-worker enqueue). It is measured
// *inside* the `advance` bucket, so it is reported separately rather than
// summed. Before the raster worker this call also contained all of
// rasterization, which is why the pre-worker `advance` share was misleading.
static std::atomic<uint64_t> s_vpGifSubmitTicks{0u};
static const bool s_vpGifOn = std::getenv("PS2X_VU1_PROF") != nullptr;
#endif

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
// [vu1:scope] PS2X_VU1_SCOPE=1 -- inventory of the VU1 microprograms the game
// actually runs, as input for scoping the VU1 recompiler: each distinct code
// image (FNV-1a of VU1 code memory), how often it is entered and at which start
// PCs, instructions executed, highest PC reached, and an opcode histogram. Code
// images are written to PS2X_VU1_DUMP_DIR as vu1prog_<hash>.bin (never to the
// working directory). Only the game thread runs VU code, so no locking.
#include <fstream>
#include <map>
#include <vector>
namespace vu1scope
{
    const bool enabled = std::getenv("PS2X_VU1_SCOPE") != nullptr;

    struct ProgramStats
    {
        uint64_t runs = 0;
        uint64_t instrs = 0;
        uint32_t maxPc = 0;
        std::map<uint32_t, uint64_t> startPcs;
        std::vector<uint64_t> pairCounts;             // executions per pair (pc / 8)
        std::map<uint32_t, uint64_t> jumpTargets;     // pcs reached by a taken jump
    };

    std::map<uint64_t, ProgramStats> programs;
    std::map<uint32_t, uint64_t> upperOps;
    std::map<uint32_t, uint64_t> lowerOps;
    uint64_t currentHash = 0;
    uint64_t lastGeneration = ~0ull;
    const uint8_t *lastCode = nullptr;
    uint32_t lastCodeSize = 0;
    uint64_t imageChanges = 0;
    uint32_t lastPc = ~0u; // previous executed pair in this run (~0u = run start)
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();

    inline uint64_t fnv1a(const uint8_t *p, size_t n)
    {
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < n; ++i)
        {
            h ^= p[i];
            h *= 1099511628211ull;
        }
        return h;
    }

    // Upper: op 0x3C-0x3F is the special table, indexed by (bits 1:0 | bits 10:6 << 2).
    inline uint32_t upperKey(uint32_t u)
    {
        const uint32_t op = u & 0x3Fu;
        return op >= 0x3Cu ? (0x100u | ((u & 3u) | ((u >> 4) & 0x7Cu))) : op;
    }

    // Lower: bits 31:25 select the opcode; 0x40 is the special table (bits 5:0,
    // with 0x3C-0x3F again extended by bits 10:6).
    inline uint32_t lowerKey(uint32_t l)
    {
        const uint32_t hi = (l >> 25) & 0x7Fu;
        if (hi != 0x40u)
            return hi;
        const uint32_t op = l & 0x3Fu;
        return op >= 0x3Cu ? (0x300u | ((l & 3u) | ((l >> 4) & 0x7Cu))) : (0x200u | op);
    }

    void report()
    {
        std::fprintf(stderr, "[vu1:scope] programs=%zu imageChanges=%llu\n", programs.size(),
                     static_cast<unsigned long long>(imageChanges));
        for (const auto &[hash, p] : programs)
        {
            std::vector<std::pair<uint64_t, uint32_t>> starts;
            for (const auto &[pc, n] : p.startPcs)
                starts.emplace_back(n, pc);
            std::sort(starts.rbegin(), starts.rend());
            std::string top;
            char buf[48];
            for (size_t i = 0; i < starts.size() && i < 12u; ++i)
            {
                std::snprintf(buf, sizeof(buf), " 0x%x:%llu", starts[i].second,
                              static_cast<unsigned long long>(starts[i].first));
                top += buf;
            }
            std::fprintf(stderr,
                         "[vu1:scope] prog %016llx runs=%llu instrs=%llu avg=%.0f maxPc=0x%x startPcs=%zu top:%s\n",
                         static_cast<unsigned long long>(hash), static_cast<unsigned long long>(p.runs),
                         static_cast<unsigned long long>(p.instrs),
                         p.runs ? double(p.instrs) / double(p.runs) : 0.0, p.maxPc, p.startPcs.size(), top.c_str());
        }
        for (const auto &[hash, p] : programs)
        {
            if (p.pairCounts.empty() || p.instrs == 0u)
                continue;
            std::vector<std::pair<uint64_t, uint32_t>> hot;
            for (uint32_t i = 0; i < p.pairCounts.size(); ++i)
                if (p.pairCounts[i] != 0u)
                    hot.emplace_back(p.pairCounts[i], i * 8u);
            std::sort(hot.rbegin(), hot.rend());
            std::string line;
            char buf[64];
            uint64_t covered = 0u;
            for (size_t i = 0; i < hot.size() && i < 24u; ++i)
            {
                covered += hot[i].first;
                std::snprintf(buf, sizeof(buf), " 0x%x:%llu", hot[i].second,
                              static_cast<unsigned long long>(hot[i].first));
                line += buf;
            }
            std::fprintf(stderr, "[vu1:scope] hotpairs %016llx distinct=%zu top24=%.1f%%:%s\n",
                         static_cast<unsigned long long>(hash), hot.size(),
                         100.0 * double(covered) / double(p.instrs), line.c_str());
            std::vector<std::pair<uint64_t, uint32_t>> jumps;
            for (const auto &[pc, n] : p.jumpTargets)
                jumps.emplace_back(n, pc);
            std::sort(jumps.rbegin(), jumps.rend());
            line.clear();
            for (size_t i = 0; i < jumps.size() && i < 20u; ++i)
            {
                std::snprintf(buf, sizeof(buf), " 0x%x:%llu", jumps[i].second,
                              static_cast<unsigned long long>(jumps[i].first));
                line += buf;
            }
            std::fprintf(stderr, "[vu1:scope] jumptargets %016llx distinct=%zu:%s\n",
                         static_cast<unsigned long long>(hash), jumps.size(), line.c_str());
        }
        auto dumpHist = [](const char *name, const std::map<uint32_t, uint64_t> &hist)
        {
            std::vector<std::pair<uint64_t, uint32_t>> v;
            for (const auto &[k, n] : hist)
                v.emplace_back(n, k);
            std::sort(v.rbegin(), v.rend());
            std::string line;
            char buf[40];
            for (const auto &[n, k] : v)
            {
                std::snprintf(buf, sizeof(buf), " %x:%llu", k, static_cast<unsigned long long>(n));
                line += buf;
            }
            std::fprintf(stderr, "[vu1:scope] %s distinct=%zu:%s\n", name, v.size(), line.c_str());
        };
        dumpHist("upperOps", upperOps);
        dumpHist("lowerOps", lowerOps);
    }

    void onRun(const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc)
    {
        const uint64_t generation = memory ? memory->getVU1CodeGeneration() : 0u;
        if (generation != lastGeneration || vuCode != lastCode || codeSize != lastCodeSize || !memory)
        {
            lastGeneration = generation;
            lastCode = vuCode;
            lastCodeSize = codeSize;
            const uint64_t hash = fnv1a(vuCode, codeSize);
            if (hash != currentHash)
                ++imageChanges;
            currentHash = hash;
            if (programs.find(hash) == programs.end())
            {
                if (const char *dir = std::getenv("PS2X_VU1_DUMP_DIR"))
                {
                    char name[64];
                    std::snprintf(name, sizeof(name), "/vu1prog_%016llx.bin", static_cast<unsigned long long>(hash));
                    std::ofstream out(std::string(dir) + name, std::ios::binary);
                    out.write(reinterpret_cast<const char *>(vuCode), codeSize);
                }
                programs[hash];
            }
        }
        ProgramStats &p = programs[currentHash];
        ++p.runs;
        ++p.startPcs[pc];
        lastPc = ~0u;

        const auto now = std::chrono::steady_clock::now();
        if (now - lastReport >= std::chrono::seconds(10))
        {
            lastReport = now;
            report();
        }
    }

    inline void onInstr(uint32_t upper, uint32_t lower, uint32_t pc)
    {
        ProgramStats &p = programs[currentHash];
        ++p.instrs;
        if (p.pairCounts.empty())
            p.pairCounts.assign(0x4000u / 8u, 0u);
        if ((pc / 8u) < p.pairCounts.size())
            ++p.pairCounts[pc / 8u];
        if (lastPc != ~0u && pc != lastPc + 8u)
            ++p.jumpTargets[pc];
        lastPc = pc;
        if (pc > p.maxPc)
            p.maxPc = pc;
        ++upperOps[upperKey(upper)];
        ++lowerOps[lowerKey(lower)];
    }
}
#endif

namespace
{
    // Lowest free slot in an occupancy mask covering `count` slots, or -1.
    inline int firstFreeSlot(uint32_t mask, uint32_t count)
    {
        const uint32_t freeBits = ~mask & ((count >= 32u) ? 0xFFFFFFFFu : ((1u << count) - 1u));
        return freeBits != 0u ? std::countr_zero(freeBits) : -1;
    }

    // Stage 1a VU1 fast path (PS2X_VU1_FAST=1). execUpper/execLower already
    // write their results straight into the registers; the exact path then
    // copies the old VF/VI/ACC values back and replays the new ones through
    // the latency queues. The fast path skips that copy-back/replay. Timing is
    // untouched (calculatePairReadyCycle/markPairWrites still run), so Q, P,
    // flag and XGKICK behaviour are exactly the interpreter's, and values match
    // because every declared read of a pending write stalls until it lands.
    // The same-pair shadow rule and the branch-VI rule are kept as they are.
    // PS2X_VU1_FLAG_CHECK=1: keep MAC/status/clip the old eager way in a shadow
    // and compare at every flag read (any build).
    const bool s_flagCheck = std::getenv("PS2X_VU1_FLAG_CHECK") != nullptr;

    const bool s_vu1FastMode = []
    {
        const char *value = std::getenv("PS2X_VU1_FAST");
        return value != nullptr && value[0] == '1';
    }();

    // Differential harness (PS2X_VU1_DIFF=1, diagnostic builds only): each VU1
    // run executes first on a copy in fast mode (GIF output captured, not
    // sent), then for real in exact mode, and the results are compared --
    // registers, Q/P/I/flags, cycle count, VU data memory and GIF packets.
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    const bool s_vu1DiffMode = std::getenv("PS2X_VU1_DIFF") != nullptr;
#else
    const bool s_vu1DiffMode = false;
#endif
    thread_local bool t_vu1InShadow = false;

    // PS2X_VU1_DIFF pair trace: the (pc, cycle-on-entry) sequence of each pass,
    // so a divergence can be located at the instruction that caused it instead
    // of only being visible in the end-of-program state.
    struct Vu1PairTrace
    {
        static constexpr uint32_t kCap = 8192u;
        uint32_t pcs[kCap];
        uint64_t cycles[kCap];
        uint64_t maxReady[kCap];
        uint64_t fdiv[kCap];   // readyCycle, or 0 when not in flight
        uint64_t efu[kCap];    // latest in-flight EFU readyCycle, else 0
        // Architecturally visible values on entry to each pair: where exact and
        // relaxed timing can legitimately disagree (Q from DIV, MAC/CLIP/STATUS).
        uint32_t qBits[kCap];
        uint32_t mac[kCap];
        uint32_t clip[kCap];
        uint32_t status[kCap];
        uint32_t count = 0;
        void clear() { count = 0; }
        void add(uint32_t p, uint64_t c, uint64_t mr, uint64_t fd, uint64_t ef, float q = 0.0f, uint32_t m = 0u,
                 uint32_t cl = 0u, uint32_t st = 0u)
        {
            if (count < kCap)
            {
                pcs[count] = p;
                cycles[count] = c;
                maxReady[count] = mr;
                fdiv[count] = fd;
                efu[count] = ef;
                std::memcpy(&qBits[count], &q, 4);
                mac[count] = m;
                clip[count] = cl;
                status[count] = st;
                ++count;
            }
        }
    };
    Vu1PairTrace g_traceAuth;
    Vu1PairTrace g_traceShadow;

    struct Vu1DiffState
    {
        std::unique_ptr<VU1Interpreter> shadow;
        std::vector<uint8_t> shadowData;
        std::vector<uint8_t> shadowGif;
        std::vector<uint8_t> authGif;
        uint32_t startPc = 0u;
        uint64_t compared = 0u;
        uint64_t mismatched = 0u;
        uint64_t skipped = 0u;
    };

    Vu1DiffState &vu1Diff()
    {
        static Vu1DiffState state;
        return state;
    }

    inline uint64_t codeHash(const uint8_t *p, size_t n)
    {
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < n; ++i)
        {
            h ^= p[i];
            h *= 1099511628211ull;
        }
        return h;
    }
}

// Set by executeVU0Microprogram around the screen-matrix vcallms programs so the
// per-instruction [vu0:i] trace only fires for those.
std::atomic<bool> g_vu0InstrTrace{false};
std::atomic<uint32_t> g_vu0InstrTraceStartPc{0u};
// sourcePc of the hierTrace* that launched the current hierTraverseAsm.
std::atomic<uint32_t> g_travOrigin{0u};

namespace
{
}

VU1Interpreter::VU1Interpreter(Unit unit)
    : m_unit(unit)
{
    reset();
}

void VU1Interpreter::resetScheduler()
{
    m_fdiv = {};
    m_efu = {};
    {
        // Empty the pipelines by mask in both modes: every reader walks the
        // occupancy masks, and every allocation re-initialises its entry, so a
        // freed slot's stale fields are never read. (Zero-filling all ~5 KB of
        // slots cost most of the ~200 ns per-call setup of the ~70k VU0
        // calls/s, and reset() then did it a second time.)
        for (uint32_t live = m_flagMask; live != 0u; live &= live - 1u)
            m_flagPipeline[static_cast<size_t>(std::countr_zero(live))].valid = false;
        for (uint32_t live = m_storeMask; live != 0u; live &= live - 1u)
            m_storePipeline[static_cast<size_t>(std::countr_zero(live))].valid = false;
        for (uint32_t live = m_vfWriteMask; live != 0u; live &= live - 1u)
            m_vfWritePipeline[static_cast<size_t>(std::countr_zero(live))].valid = false;
        for (uint32_t live = m_viWriteMask; live != 0u; live &= live - 1u)
            m_viWritePipeline[static_cast<size_t>(std::countr_zero(live))].valid = false;
        for (uint32_t live = m_accWriteMask; live != 0u; live &= live - 1u)
            m_accWritePipeline[static_cast<size_t>(std::countr_zero(live))].valid = false;
    }
    // Bookkeeping only: `m_xgkick = {}` zero-filled the 64 KB packet buffer on
    // every execute() (~30k/s = ~2 GB/s of memset, VCRUNTIME memset was 5% of
    // the game thread). Packet bytes are always copied in before being read.
    m_xgkick.sourceAddress = 0u;
    m_xgkick.totalBytes = 0u;
    m_xgkick.copiedBytes = 0u;
    m_xgkick.currentTagEnd = 0u;
    m_xgkick.cycleCredit = 0u;
    m_xgkick.issueCycle = 0u;
    m_xgkick.active = false;
    m_xgkick.currentTagEop = false;
    if (!m_immediateWrites)
    {
        m_vfReady = {};
        m_viReady = {};
        m_accReady = {};
        m_vfLatestWrite = {};
        m_viLatestWrite = {};
        m_accLatestWrite = {};
    }
    m_vfPendingMask = 0u;
    m_viPendingMask = 0u;
    m_accPending = false;
    m_pipelineNextReady = ~0ull;
    m_maxReadyCycle = 0;
    m_flagMask = 0u;
    m_storeMask = 0u;
    m_vfWriteMask = 0u;
    m_viWriteMask = 0u;
    m_accWriteMask = 0u;
    m_nextWriteSequence = 0;
    m_efuResourceReady = 0;
    m_workingClip = m_state.clip;
    m_shadowMac = m_state.mac;
    m_shadowStatus = m_state.status;
    m_shadowClip = m_state.clip;
    m_flagOrderHead = 0u;
    m_flagOrderCount = 0u;
    m_flagTailReady = 0u;
    m_viBranchBackupValue = 0;
    m_viBranchBackupReg = 0;
    m_viBranchBackupValid = false;
    m_stopRequested = false;
    m_pendingHaltD = false;
    m_pendingHaltT = false;
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f;
    m_state.q = 1.0f;
    m_state.r = 0x3F800000u;
    m_cycle = 0;
    resetScheduler();
    // The clock just rewound, so stale ready cycles would now lie in the
    // future: clear them (resetScheduler already did when !m_immediateWrites).
    // Pipeline slots need no clearing, see resetScheduler.
    if (m_immediateWrites)
    {
        m_vfReady = {};
        m_viReady = {};
        m_accReady = {};
        m_vfLatestWrite = {};
        m_viLatestWrite = {};
        m_accLatestWrite = {};
    }
}

float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return normalizeOperand(vf[bc & 3u]);
}

float VU1Interpreter::normalizeOperand(float value) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    if (exponent == 0u)
    {
        bits &= 0x80000000u;
    }
    else if (exponent == 0xFFu)
    {
        bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t exponent = (bits >> 23) & 0xFFu;

    laneFlags = sign != 0u ? 0x2u : 0u;
    if (magnitude == 0u)
    {
        laneFlags |= 0x1u;
    }
    else if (exponent == 0u)
    {
        laneFlags |= 0x5u;
        bits = sign;
    }
    else if (exponent == 0xFFu)
    {
        laneFlags |= 0x8u;
        bits = sign | 0x7F7FFFFFu;
    }

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
{
    if (reg == 0u)
        return 0;
    if (m_viBranchBackupValid &&
        m_viBranchBackupReg == reg)
    {
        return m_viBranchBackupValue;
    }
    return m_state.vi[reg];
}

void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8u)
        dst[0] = result[0];
    if (dest & 0x4u)
        dst[1] = result[1];
    if (dest & 0x2u)
        dst[2] = result[2];
    if (dest & 0x1u)
        dst[3] = result[3];
}

void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

void VU1Interpreter::normalizeFmacResult(float *result, uint8_t dest,
                                         uint8_t laneFlags[4])
{
    for (uint32_t component = 0; component < 4u; ++component)
    {
        laneFlags[component] = 0u;
        if ((dest & laneForComponent(component)) == 0u)
            continue;

        long double exactResult = 0.0L;
        if (calculateFmacExactResult(component, exactResult))
        {
            laneFlags[component] = normalizeFmacExactResult(result[component], exactResult);
            continue;
        }

        uint32_t flags = 0u;
        result[component] = normalizeResult(result[component], flags);
        laneFlags[component] = static_cast<uint8_t>(flags);
    }
}

bool VU1Interpreter::calculateFmacExactResult(uint32_t component,
                                               long double &result) const
{
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu
                                ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu))
                                : 0xFFu;
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);

    const auto operand = [this](float value)
    {
        return static_cast<long double>(normalizeOperand(value));
    };
    const auto vs = [&](uint32_t lane)
    {
        return operand(m_state.vf[fs][lane]);
    };
    const auto vt = [&](uint32_t lane)
    {
        return operand(m_state.vf[ft][lane]);
    };
    const auto acc = [&](uint32_t lane)
    {
        return operand(m_state.acc[lane]);
    };

    const long double q = operand(m_state.q);
    const long double i = operand(m_state.i);

    if (op < 0x3Cu)
    {
        if (op <= 0x03u)
            result = vs(component) + vt(op & 3u);
        else if (op <= 0x07u)
            result = vs(component) - vt(op & 3u);
        else if (op <= 0x0Bu)
            result = acc(component) + vs(component) * vt(op & 3u);
        else if (op <= 0x0Fu)
            result = acc(component) - vs(component) * vt(op & 3u);
        else if (op >= 0x18u && op <= 0x1Bu)
            result = vs(component) * vt(op & 3u);
        else
        {
            switch (op)
            {
            case 0x1Cu:
                result = vs(component) * q;
                break;
            case 0x1Eu:
                result = vs(component) * i;
                break;
            case 0x20u:
                result = vs(component) + q;
                break;
            case 0x21u:
                result = acc(component) + vs(component) * q;
                break;
            case 0x22u:
                result = vs(component) + i;
                break;
            case 0x23u:
                result = acc(component) + vs(component) * i;
                break;
            case 0x24u:
                result = vs(component) - q;
                break;
            case 0x25u:
                result = acc(component) - vs(component) * q;
                break;
            case 0x26u:
                result = vs(component) - i;
                break;
            case 0x27u:
                result = acc(component) - vs(component) * i;
                break;
            case 0x28u:
                result = vs(component) + vt(component);
                break;
            case 0x29u:
                result = acc(component) + vs(component) * vt(component);
                break;
            case 0x2Au:
                result = vs(component) * vt(component);
                break;
            case 0x2Cu:
                result = vs(component) - vt(component);
                break;
            case 0x2Du:
                result = acc(component) - vs(component) * vt(component);
                break;
            case 0x2Eu:
            {
                static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
                static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
                result = component == 3u
                             ? 0.0L
                             : acc(component) - vs(left[component]) * vt(right[component]);
                break;
            }
            default:
                return false;
            }
        }
        return true;
    }

    if (special <= 0x03u)
        result = vs(component) + vt(special & 3u);
    else if (special <= 0x07u)
        result = vs(component) - vt(special & 3u);
    else if (special <= 0x0Bu)
        result = acc(component) + vs(component) * vt(special & 3u);
    else if (special <= 0x0Fu)
        result = acc(component) - vs(component) * vt(special & 3u);
    else if (special >= 0x18u && special <= 0x1Bu)
        result = vs(component) * vt(special & 3u);
    else
    {
        switch (special)
        {
        case 0x1Cu:
            result = vs(component) * q;
            break;
        case 0x1Eu:
            result = vs(component) * i;
            break;
        case 0x20u:
            result = vs(component) + q;
            break;
        case 0x21u:
            result = acc(component) + vs(component) * q;
            break;
        case 0x22u:
            result = vs(component) + i;
            break;
        case 0x23u:
            result = acc(component) + vs(component) * i;
            break;
        case 0x24u:
            result = vs(component) - q;
            break;
        case 0x25u:
            result = acc(component) - vs(component) * q;
            break;
        case 0x26u:
            result = vs(component) - i;
            break;
        case 0x27u:
            result = acc(component) - vs(component) * i;
            break;
        case 0x28u:
            result = vs(component) + vt(component);
            break;
        case 0x29u:
            result = acc(component) + vs(component) * vt(component);
            break;
        case 0x2Au:
            result = vs(component) * vt(component);
            break;
        case 0x2Cu:
            result = vs(component) - vt(component);
            break;
        case 0x2Du:
            result = acc(component) - vs(component) * vt(component);
            break;
        case 0x2Eu:
        {
            static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
            static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
            result = component == 3u
                         ? 0.0L
                         : vs(left[component]) * vt(right[component]);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

uint8_t VU1Interpreter::normalizeFmacExactResult(float &value,
                                                  long double exactResult) const
{
    const bool negative = std::signbit(exactResult);
    const long double magnitude = std::fabs(exactResult);
    const long double maximum = static_cast<long double>(std::numeric_limits<float>::max());
    const long double minimum = static_cast<long double>(std::numeric_limits<float>::min());
    uint8_t flags = negative ? 0x2u : 0u;

    uint32_t bits = negative ? 0x80000000u : 0u;
    if (magnitude == 0.0L)
    {
        flags |= 0x1u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude > maximum)
    {
        flags |= 0x8u;
        bits |= 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude < minimum)
    {
        flags |= 0x5u;
        std::memcpy(&value, &bits, sizeof(value));
    }

    return flags;
}

uint32_t VU1Interpreter::calculateFmacProductSticky(uint8_t dest) const
{
    uint32_t extraSticky = 0u;
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
    const bool productSum =
        (op >= 0x08u && op <= 0x0Fu) ||
        op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
        op == 0x29u || op == 0x2Du || op == 0x2Eu ||
        (special >= 0x08u && special <= 0x0Fu) ||
        special == 0x21u || special == 0x23u || special == 0x25u ||
        special == 0x27u || special == 0x29u || special == 0x2Du;
    if (!productSum)
        return 0u;

    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & laneForComponent(component)) == 0u)
            continue;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const uint8_t leftComponent = op == 0x2Eu ? crossLeft[component] : static_cast<uint8_t>(component);
        const float left = normalizeOperand(m_state.vf[fs][leftComponent]);
        float right = 0.0f;
        if ((op >= 0x08u && op <= 0x0Fu) || (special >= 0x08u && special <= 0x0Fu))
        {
            right = normalizeOperand(m_state.vf[ft][(op >= 0x08u && op <= 0x0Fu ? op : special) & 3u]);
        }
        else if (op == 0x21u || op == 0x25u || special == 0x21u || special == 0x25u)
        {
            right = normalizeOperand(m_state.q);
        }
        else if (op == 0x23u || op == 0x27u || special == 0x23u || special == 0x27u)
        {
            right = normalizeOperand(m_state.i);
        }
        else if (op == 0x2Eu)
        {
            right = normalizeOperand(m_state.vf[ft][crossRight[component]]);
        }
        else
        {
            right = normalizeOperand(m_state.vf[ft][component]);
        }

        float product = left * right;
        const long double exactProduct = static_cast<long double>(left) * static_cast<long double>(right);
        const uint8_t productFlags = normalizeFmacExactResult(product, exactProduct);
        // Product-sum instructions report Z/S/U/O from the add/subtract result
        // as current flags, while every product condition accumulates into the
        // corresponding sticky flag.
        extraSticky |= productFlags & 0xFu;
    }
    return extraSticky;
}

void VU1Interpreter::updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest,
                                     uint32_t extraSticky)
{
    if (dest == 0u)
        return;

    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint8_t lane = laneForComponent(component);
        if ((dest & lane) == 0u)
            continue;

        const uint32_t flags = laneFlags[component];
        if ((flags & 0x1u) != 0u)
            mac |= lane;
        if ((flags & 0x2u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 4;
        if ((flags & 0x4u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 8;
        if ((flags & 0x8u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 12;
        status |= flags;
    }

    int freeSlot = firstFreeSlot(m_flagMask, kMaxFlagEntries);
    if (freeSlot < 0)
    {
        retireFlags();
        freeSlot = firstFreeSlot(m_flagMask, kMaxFlagEntries);
    }
    if (freeSlot < 0)
    {
        reportReservedInstruction(true, 0xFFFFFFFFu);
        return;
    }
    FlagPipelineEntry *entry = &m_flagPipeline[static_cast<size_t>(freeSlot)];
    m_flagMask |= 1u << freeSlot;
    pushFlagOrder(static_cast<uint32_t>(freeSlot));

    *entry = {};
    entry->valid = true;
    entry->issueCycle = m_cycle;
    entry->readyCycle = m_cycle + kFmacLatency;
    if (s_flagCheck)
        notePipelineReady(m_cycle + kFmacLatency);
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
}

void VU1Interpreter::applyFmacDest(float *dst, float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDest(dst, result, dest);
}

void VU1Interpreter::applyFmacDestAcc(float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
    applyDestAcc(result, dest);
}

void VU1Interpreter::queueFsset(uint16_t immediate)
{
    for (uint32_t live = m_flagMask; live != 0u; live &= live - 1u)
    {
        FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(std::countr_zero(live))];
        if (entry.issueCycle == m_cycle)
            entry.writesStatus = false;
    }

    if (firstFreeSlot(m_flagMask, kMaxFlagEntries) < 0)
        retireFlags();
    if (const int freeSlot = firstFreeSlot(m_flagMask, kMaxFlagEntries); freeSlot >= 0)
    {
        FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(freeSlot)];
        m_flagMask |= 1u << freeSlot;
        pushFlagOrder(static_cast<uint32_t>(freeSlot));
        {
            entry = {};
            entry.valid = true;
            entry.issueCycle = m_cycle;
            entry.readyCycle = m_cycle + kFmacLatency;
            if (s_flagCheck)
                notePipelineReady(m_cycle + kFmacLatency);
            entry.status = static_cast<uint32_t>(immediate) & 0xFC0u;
            entry.writesSticky = true;
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFFEu);
}

void VU1Interpreter::queueClip(uint32_t clip)
{
    m_workingClip = ((m_workingClip << 6) | (clip & 0x3Fu)) & 0xFFFFFFu;
    if (firstFreeSlot(m_flagMask, kMaxFlagEntries) < 0)
        retireFlags();
    if (const int freeSlot = firstFreeSlot(m_flagMask, kMaxFlagEntries); freeSlot >= 0)
    {
        FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(freeSlot)];
        m_flagMask |= 1u << freeSlot;
        pushFlagOrder(static_cast<uint32_t>(freeSlot));
        {
            entry = {};
            entry.valid = true;
            entry.issueCycle = m_cycle;
            entry.readyCycle = m_cycle + kFmacLatency;
            if (s_flagCheck)
                notePipelineReady(m_cycle + kFmacLatency);
            entry.clip = m_workingClip;
            entry.writesClip = true;
            return;
        }
    }
    reportReservedInstruction(true, 0xFFFFFFFDu);
}

void VU1Interpreter::queueFcset(uint32_t clip)
{
    m_workingClip = clip & 0xFFFFFFu;
    for (uint32_t live = m_flagMask; live != 0u; live &= live - 1u)
    {
        FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(std::countr_zero(live))];
        if (entry.issueCycle == m_cycle)
            entry.writesClip = false;
    }
    if (firstFreeSlot(m_flagMask, kMaxFlagEntries) < 0)
        retireFlags();
    if (const int freeSlot = firstFreeSlot(m_flagMask, kMaxFlagEntries); freeSlot >= 0)
    {
        FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(freeSlot)];
        m_flagMask |= 1u << freeSlot;
        pushFlagOrder(static_cast<uint32_t>(freeSlot));
        {
            entry = {};
            entry.valid = true;
            entry.issueCycle = m_cycle;
            entry.readyCycle = m_cycle + kFmacLatency;
            if (s_flagCheck)
                notePipelineReady(m_cycle + kFmacLatency);
            entry.clip = m_workingClip;
            entry.writesClip = true;
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFFAu);
}

void VU1Interpreter::queueQ(float value, uint32_t latency, uint32_t statusDi)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    m_fdiv.valid = true;
    m_fdiv.readyCycle = m_cycle + latency;
    notePipelineReady(m_cycle + latency);
    m_fdiv.value = value;
    m_fdiv.statusDi = statusDi & 0x30u;
}

void VU1Interpreter::queueP(float value, uint32_t latency)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (!entry.valid)
        {
            entry.valid = true;
            entry.readyCycle = m_cycle + latency;
            notePipelineReady(m_cycle + latency);
            entry.value = value;
            // EFU throughput is one cycle shorter than result visibility.
            m_efuResourceReady = m_cycle + (latency > 0u ? latency - 1u : 0u);
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF9u);
}

void VU1Interpreter::queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    // A fast VU1 store is the pair's sole lower instruction. No memory
    // observer runs between this write and advanceOneCycle[Fast], which
    // retires stores BEFORE progressing PATH1. Writing now preserves the
    // data seen by both the next pair and an active graphics transfer.
    // Older queued stores, VU0 and exact execution keep the timed path.
    if(s_nativeStores && !t_nativeStoreReference && m_unit==Unit::VU1 &&
       m_immediateWrites && (s_nativeTransferStores || !m_xgkick.active) && !m_storeMask &&
       m_activeVuData && address+16u<=m_activeVuDataSize) {
        if(laneMask==0xfu) std::memcpy(m_activeVuData+address,words,16u);
        else for(unsigned lane=0;lane<4;++lane) if(laneMask&(8u>>lane))
            std::memcpy(m_activeVuData+address+lane*4u,words+lane,4u);
        return;
    }

    if (const int freeSlot = firstFreeSlot(m_storeMask, kMaxPendingStores); freeSlot >= 0)
    {
        PendingStore &store = m_storePipeline[static_cast<size_t>(freeSlot)];
        m_storeMask |= 1u << freeSlot;
        {
            store.valid = true;
            store.readyCycle = m_cycle + 1u;
            notePipelineReady(m_cycle + 1u);
            store.address = address;
            store.laneMask = laneMask;
            std::copy(words, words + 4, store.words.begin());
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFFCu);
}

void VU1Interpreter::queueVfWrite(uint8_t reg, uint8_t laneMask,
                                  const float value[4], uint32_t latency)
{
    if (reg == 0u || laneMask == 0u)
        return;
    if (const int freeSlot = firstFreeSlot(m_vfWriteMask, kMaxPendingVfWrites); freeSlot >= 0)
    {
        PendingVfWrite &write = m_vfWritePipeline[static_cast<size_t>(freeSlot)];
        m_vfWriteMask |= 1u << freeSlot;
        {
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            notePipelineReady(m_cycle + latency);
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    m_vfLatestWrite[reg][component] = write.sequence;
            }
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF7u);
}

void VU1Interpreter::queueViWrite(uint8_t reg, int32_t value, uint32_t latency)
{
    if (reg == 0u)
        return;
    if (const int freeSlot = firstFreeSlot(m_viWriteMask, kMaxPendingViWrites); freeSlot >= 0)
    {
        PendingViWrite &write = m_viWritePipeline[static_cast<size_t>(freeSlot)];
        m_viWriteMask |= 1u << freeSlot;
        {
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            notePipelineReady(m_cycle + latency);
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.value = value;
            m_viLatestWrite[reg] = write.sequence;
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF6u);
}

void VU1Interpreter::queueAccWrite(uint8_t laneMask, const float value[4], uint32_t latency)
{
    if (laneMask == 0u)
        return;
    if (const int freeSlot = firstFreeSlot(m_accWriteMask, kMaxPendingAccWrites); freeSlot >= 0)
    {
        PendingAccWrite &write = m_accWritePipeline[static_cast<size_t>(freeSlot)];
        m_accWriteMask |= 1u << freeSlot;
        {
            write = {};
            write.valid = true;
            write.readyCycle = m_cycle + latency;
            notePipelineReady(m_cycle + latency);
            write.sequence = ++m_nextWriteSequence;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    m_accLatestWrite[component] = write.sequence;
            }
            return;
        }
    }
    reportReservedInstruction(true, 0xFFFFFFF5u);
}

void VU1Interpreter::pushFlagOrder(uint32_t slot)
{
    m_flagOrder[(m_flagOrderHead + m_flagOrderCount) % kMaxFlagEntries] = static_cast<uint8_t>(slot);
    ++m_flagOrderCount;
}

void ps2xVu1TracePair(uint32_t pc, uint64_t cycle, uint64_t maxReady, uint64_t fdiv, uint64_t efu,
                      float q, uint32_t mac, uint32_t clip, uint32_t status)
{
    if (s_vu1DiffMode)
        (t_vu1InShadow ? g_traceShadow : g_traceAuth).add(pc, cycle, maxReady, fdiv, efu, q, mac, clip, status);
}

void VU1Interpreter::retireFlags()
{
    // Empty and not-yet-ready queues need no configuration or register work.
    if (!m_flagOrderCount || m_flagPipeline[m_flagOrder[m_flagOrderHead]].readyCycle>m_cycle)
        return;
    if (s_batchFlags && m_flagOrderCount>=4u) {
        // The diagnostic compares the entire entry array, cursor, mask and
        // visible registers against the original loop, including pending tails.
        static thread_local uint64_t checked=0;
        if (s_batchFlagsVerify && (++checked%4096u)==1u) {
            auto expected=m_flagPipeline;
            uint32_t head=m_flagOrderHead,count=m_flagOrderCount,mask=m_flagMask;
            uint32_t mac=m_state.mac,status=m_state.status,clip=m_state.clip;
            while (count) {
                const uint32_t slot=m_flagOrder[head];
                auto &entry=expected[slot];
                if (entry.readyCycle>m_cycle) break;
                head=(head+1u)%kMaxFlagEntries;--count;
                if (entry.writesMac) mac=entry.mac;
                if (entry.writesStatus) {
                    const uint32_t current=entry.status&0xFu;
                    status=(status&0xFF0u)|current|((current|entry.extraSticky)<<6);
                }
                if (entry.writesSticky) status=(status&0x03Fu)|(entry.status&0xFC0u);
                if (entry.writesClip) clip=entry.clip;
                entry.valid=false;mask&=~(1u<<slot);
            }
            wotmVuFlags::retireReady(m_flagPipeline,m_flagOrder,m_flagOrderHead,
                                    m_flagOrderCount,m_flagMask,m_state,m_cycle);
            if (head!=m_flagOrderHead || count!=m_flagOrderCount || mask!=m_flagMask ||
                mac!=m_state.mac || status!=m_state.status || clip!=m_state.clip ||
                std::memcmp(expected.data(),m_flagPipeline.data(),sizeof(expected))) {
                std::fprintf(stderr,"[vu:batch-flags] MISMATCH cycle=%llu\n",(unsigned long long)m_cycle);
                std::abort();
            }
            if ((checked%1048576u)==1u)
                std::fprintf(stderr,"[vu:batch-flags] calls=%llu checked=%llu mismatches=0\n",
                    (unsigned long long)checked,(unsigned long long)((checked+4095u)/4096u));
            return;
        }
        wotmVuFlags::retireReady(m_flagPipeline,m_flagOrder,m_flagOrderHead,
                                m_flagOrderCount,m_flagMask,m_state,m_cycle);
        return;
    }
    // Entries are queued in issue order and all share one latency, so the
    // head of the FIFO is always the oldest; stop at the first not yet ready.
    while (m_flagOrderCount != 0u)
    {
        const uint32_t best = m_flagOrder[m_flagOrderHead];
        FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(best)];
        if (entry.readyCycle > m_cycle)
            return;
        m_flagOrderHead = (m_flagOrderHead + 1u) % kMaxFlagEntries;
        --m_flagOrderCount;
        if (entry.writesMac)
            m_state.mac = entry.mac;
        if (entry.writesStatus)
        {
            const uint32_t current = entry.status & 0xFu;
            m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
        }
        if (entry.writesSticky)
            m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
        if (entry.writesClip)
            m_state.clip = entry.clip;
        // Every allocation resets the whole entry, so freeing only needs the
        // valid bit.
        entry.valid = false;
        m_flagMask &= ~(1u << static_cast<uint32_t>(best));
    }
}

void VU1Interpreter::syncFlagsForRead(const char *op)
{
    retireFlags();
    if (!s_flagCheck)
        return;
    static std::atomic<uint64_t> s_checked{0u}, s_bad{0u};
    const uint64_t n = s_checked.fetch_add(1u, std::memory_order_relaxed) + 1u;
    if (m_state.mac != m_shadowMac || m_state.status != m_shadowStatus || m_state.clip != m_shadowClip)
    {
        const uint64_t b = s_bad.fetch_add(1u, std::memory_order_relaxed) + 1u;
        if (b <= 20u)
            std::fprintf(stderr, "[vu1:flagcheck] MISMATCH #%llu %s pc=0x%x cycle=%llu lazy mac=%x st=%x clip=%x eager mac=%x st=%x clip=%x\n",
                         static_cast<unsigned long long>(b), op, m_state.pc, static_cast<unsigned long long>(m_cycle),
                         m_state.mac, m_state.status, m_state.clip, m_shadowMac, m_shadowStatus, m_shadowClip);
    }
    if ((n % 1000000u) == 0u)
        std::fprintf(stderr, "[vu1:flagcheck] reads=%llu mismatches=%llu\n",
                     static_cast<unsigned long long>(n),
                     static_cast<unsigned long long>(s_bad.load(std::memory_order_relaxed)));
}

void VU1Interpreter::commitReadyPipelines()
{
    // Nothing can retire yet, so the ~51-slot scan below would find nothing.
    // This is the common case on most cycles and is what makes the interpreter
    // affordable; see notePipelineReady() / m_pipelineNextReady.
    const uint64_t cycle = m_cycle;
    if (cycle < m_pipelineNextReady)
        return;

    // No callbacks run during retirement. Keep the minimum in a local until
    // all queues have been visited, avoiding repeated aliased object writes.
    uint64_t nextReady = ~0ull;

    // Flag entries no longer retire here (see retireFlags): they were ~1 per
    // cycle, and every read of MAC/status/clip goes through syncFlagsForRead,
    // so the registers are exact whenever anything can observe them. With the
    // check on, the old per-cycle application runs on a shadow copy (entries
    // stay live until retireFlags frees them).
    if (s_flagCheck)
    {
        for (uint32_t live = m_flagMask; live != 0u; live &= live - 1u)
        {
            FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(std::countr_zero(live))];
            if (entry.readyCycle > cycle)
            {
                nextReady = std::min(nextReady, entry.readyCycle);
                continue;
            }
            if (entry.shadowApplied)
                continue;
            entry.shadowApplied = true;
            if (entry.writesMac)
                m_shadowMac = entry.mac;
            if (entry.writesStatus)
            {
                const uint32_t current = entry.status & 0xFu;
                m_shadowStatus = (m_shadowStatus & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
            }
            if (entry.writesSticky)
                m_shadowStatus = (m_shadowStatus & 0x03Fu) | (entry.status & 0xFC0u);
            if (entry.writesClip)
                m_shadowClip = entry.clip;
        }
    }

    if (m_fdiv.valid)
    {
        if (m_fdiv.readyCycle <= cycle)
        {
            m_state.q = m_fdiv.value;
            const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
            m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
            if (s_flagCheck)
                m_shadowStatus = (m_shadowStatus & 0xFCFu) | currentDi | (currentDi << 6);
            m_fdiv = {};
        }
        else
        {
            nextReady = std::min(nextReady, m_fdiv.readyCycle);
        }
    }

    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (!entry.valid)
            continue;
        if (entry.readyCycle <= cycle)
        {
            m_state.p = entry.value;
            entry = {};
        }
        else
        {
            nextReady = std::min(nextReady, entry.readyCycle);
        }
    }

    for (uint32_t live = m_storeMask; live != 0u; live &= live - 1u)
    {
        const uint32_t slot = static_cast<uint32_t>(std::countr_zero(live));
        PendingStore &store = m_storePipeline[slot];
        if (store.readyCycle > cycle)
        {
            nextReady = std::min(nextReady, store.readyCycle);
            continue;
        }
        if (m_activeVuData && store.address + 16u <= m_activeVuDataSize)
        {
            if (s_fastPipelineStore && !t_qwordReference && store.laneMask==0xFu)
            {
                // All four lanes are replaced at this same commit boundary.
                std::memcpy(m_activeVuData+store.address,store.words.data(),16u);
            }
            else
            {
                uint32_t oldWords[4]{};
                std::memcpy(oldWords, m_activeVuData + store.address, sizeof(oldWords));
                for (uint32_t component = 0; component < 4u; ++component)
                {
                    if ((store.laneMask & laneForComponent(component)) != 0u)
                        oldWords[component] = store.words[component];
                }
                std::memcpy(m_activeVuData + store.address, oldWords, sizeof(oldWords));
            }
        }
        store.valid = false; // queueStore sets every field on allocation
        m_storeMask &= ~(1u << slot);
    }

    for (uint32_t live = m_vfWriteMask; live != 0u; live &= live - 1u)
    {
        const uint32_t slot = static_cast<uint32_t>(std::countr_zero(live));
        PendingVfWrite &write = m_vfWritePipeline[slot];
        if (write.readyCycle > cycle)
        {
            nextReady = std::min(nextReady, write.readyCycle);
            continue;
        }
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_vfLatestWrite[write.reg][component] == write.sequence)
            {
                m_state.vf[write.reg][component] = write.value[component];
            }
        }
        write = {};
        m_vfWriteMask &= ~(1u << slot);
    }

    for (uint32_t live = m_viWriteMask; live != 0u; live &= live - 1u)
    {
        const uint32_t slot = static_cast<uint32_t>(std::countr_zero(live));
        PendingViWrite &write = m_viWritePipeline[slot];
        if (write.readyCycle > cycle)
        {
            nextReady = std::min(nextReady, write.readyCycle);
            continue;
        }
        if (m_viLatestWrite[write.reg] == write.sequence)
            m_state.vi[write.reg] = static_cast<int16_t>(write.value);
        write = {};
        m_viWriteMask &= ~(1u << slot);
    }

    for (uint32_t live = m_accWriteMask; live != 0u; live &= live - 1u)
    {
        const uint32_t slot = static_cast<uint32_t>(std::countr_zero(live));
        PendingAccWrite &write = m_accWritePipeline[slot];
        if (write.readyCycle > cycle)
        {
            nextReady = std::min(nextReady, write.readyCycle);
            continue;
        }
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_accLatestWrite[component] == write.sequence)
            {
                m_state.acc[component] = write.value[component];
            }
        }
        write = {};
        m_accWriteMask &= ~(1u << slot);
    }
    m_pipelineNextReady = nextReady;
}

// PATH1 transfer rate for XGKICK (PS2X_XGKICK_RATE): unset = 1 qword per 2
// VU cycles (previous behaviour), 1 = 1 qword per cycle, 0 = the whole packet
// at the XGKICK itself. WotM's near-plane clipper (prog cc154aeee0cceaab, loop
// 0x2568-0x2728) reuses its output buffers right after each XGKICK; if PATH1
// is modelled slower than the hardware, the tail of a packet is read after the
// next polygon has started overwriting it -> garbage vertex -> the huge
// 'stretched monster' triangles.
namespace
{
    // Default = 1 qword per cycle. Measured with the PS2X_BIGTRI detector in
    // user gameplay against the enemy Congar: 2 cycles/qw -> 4-17k garbage
    // triangles per 5 s from the clipper (stale vertex from the next polygon);
    // instant -> garbage from the main mesh code at 0xfd0, which kicks early and
    // keeps writing the packet while PATH1 streams it; 1 qword/cycle -> 0.
    const uint32_t s_xgkickRateMode = []
    {
        const char *value = std::getenv("PS2X_XGKICK_RATE");
        if (!value || !value[0])
            return 1u;
        return value[0] == '0' ? 0u : (value[0] == '2' ? 2u : 1u);
    }();
    const uint32_t s_xgkickCyclesPerQw = s_xgkickRateMode == 1u ? 1u : 2u;
}

// Direct output of the terminal mesh sequence. Unsupported/mutable packets
// retain the timed XGKICK path. No widely included runtime interface changes.
void ps2xGsSubmitVuMesh(GS &, PS2Memory *, uint64_t, const GSVertex *, const uint8_t *,
                       uint32_t, const uint8_t *, uint32_t);
namespace {
    const bool s_directMeshEnabled = [] { const char *p = std::getenv("PS2X_VU1_DIRECT_MESH"); return !p || p[0] != '0'; }();
    const bool s_directMeshVerify = std::getenv("PS2X_VU1_DIRECT_VERIFY") != nullptr;
    const bool s_directMeshReplay = std::getenv("PS2X_VU1_DIRECT_REPLAY") != nullptr;
    thread_local bool t_directMeshReference = false;
    thread_local bool t_tailReference = false;
    thread_local uint64_t t_tailBatches = 0;
    const bool s_tailBatchEnabled = [] { const char *p = std::getenv("PS2X_VU1_TAIL_BATCH"); return !p || p[0] != '0'; }();
    const bool s_tailBatchVerify = std::getenv("PS2X_VU1_TAIL_VERIFY") != nullptr;
    thread_local uint64_t t_directMeshPrepared = 0;
    struct DirectMesh {
        VU1Interpreter *owner = nullptr;
        uint64_t issue = 0, tag = 0;
        uint32_t count = 0, bytes = 0;
        const uint8_t *raw = nullptr;
        GSVertex vertices[341]{};
        uint8_t adc[341]{};
        std::vector<uint8_t> expected;
    };
    thread_local DirectMesh t_directMesh;
}

void VU1Interpreter::progressXgkick()
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    if (t_directMesh.owner == this && t_directMesh.issue == m_xgkick.issueCycle && !s_directMeshVerify) {
        // The terminal kernel has no further writes. Keep transfer completion
        // timing, but no qword copies, tag walking, or packet allocation.
        ++m_xgkick.cycleCredit;
        const uint32_t qwords = std::min((m_xgkick.totalBytes - m_xgkick.copiedBytes) / 16u,
                                        m_xgkick.cycleCredit / s_xgkickCyclesPerQw);
        m_xgkick.cycleCredit -= qwords * s_xgkickCyclesPerQw;
        if (m_gifCapture && qwords) // expose partial packet state to the replay comparator
            std::memcpy(m_xgkick.packet.data() + m_xgkick.copiedBytes,
                        t_directMesh.raw + m_xgkick.copiedBytes, qwords * 16u);
        m_xgkick.copiedBytes += qwords * 16u;
        if (m_xgkick.copiedBytes == m_xgkick.totalBytes) finishXgkick();
        return;
    }

    // A non-final payload qword needs neither tag decoding nor completion.
    // Keep the same single-cycle read point and zero credit as the generic loop.
    static const bool fastPayload=[] {const char *p=std::getenv("PS2X_VU1_PAYLOAD_FAST");return !p || p[0]!='0';}();
    if (fastPayload && !t_qwordReference && s_xgkickCyclesPerQw == 1u &&
        m_xgkick.cycleCredit == 0u && m_activeVuDataSize == 16384u &&
        m_xgkick.copiedBytes <= XgkickPipeline::kBufferSize - 16u &&
        m_xgkick.copiedBytes + 16u < m_xgkick.currentTagEnd &&
        ((m_xgkick.sourceAddress + m_xgkick.copiedBytes) & 15u) == 0u)
    {
        const uint32_t offset=m_xgkick.copiedBytes;
        const uint32_t source=(m_xgkick.sourceAddress+offset)&16383u;
        std::memcpy(m_xgkick.packet.data()+offset,m_activeVuData+source,16u);
        m_xgkick.copiedBytes=offset+16u;
        return;
    }

    ++m_xgkick.cycleCredit;
    while (m_xgkick.active && m_xgkick.cycleCredit >= s_xgkickCyclesPerQw)
    {
        m_xgkick.cycleCredit -= s_xgkickCyclesPerQw;
        if (m_xgkick.copiedBytes > XgkickPipeline::kBufferSize - 16u)
        {
            reportReservedInstruction(false, 0xFFFFFFFBu);
            m_xgkick.active = false;
            return;
        }

        const uint32_t qwordOffset = m_xgkick.copiedBytes;
        // This used to be a byte loop with a `% m_activeVuDataSize` per byte --
        // 16 integer divisions per qword, on the path that runs every emulated
        // cycle while a kick is in flight. VU data memory always wraps at most
        // once across 16 bytes, so resolve the wrap once and copy in two runs.
        static const bool fastQword=[] {const char *p=std::getenv("PS2X_VU1_QWORD_COPY");return !p || p[0]!='0';}();
        if (fastQword && !t_qwordReference && m_activeVuDataSize == 16384u &&
            ((m_xgkick.sourceAddress + qwordOffset) & 15u) == 0u)
        {
            // Aligned VU1 qwords cannot straddle its 16 KiB ring boundary.
            // Keep the per-cycle read point; only specialize the address/copy.
            const uint32_t source=(m_xgkick.sourceAddress + qwordOffset) & 16383u;
            std::memcpy(m_xgkick.packet.data()+qwordOffset,m_activeVuData+source,16u);
        }
        else if (m_activeVuDataSize >= 16u)
        {
            const uint32_t source = (m_xgkick.sourceAddress + qwordOffset) % m_activeVuDataSize;
            uint8_t *const dst = m_xgkick.packet.data() + qwordOffset;
            const uint32_t firstRun = std::min<uint32_t>(16u, m_activeVuDataSize - source);
            std::memcpy(dst, m_activeVuData + source, firstRun);
            if (firstRun < 16u)
                std::memcpy(dst + firstRun, m_activeVuData, 16u - firstRun);
        }
        else
        {
            for (uint32_t i = 0; i < 16u; ++i)
            {
                const uint32_t source = (m_xgkick.sourceAddress + qwordOffset + i) % m_activeVuDataSize;
                m_xgkick.packet[qwordOffset + i] = m_activeVuData[source];
            }
        }
        m_xgkick.copiedBytes += 16u;

        if (m_xgkick.currentTagEnd == 0u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, m_xgkick.packet.data() + qwordOffset, sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else if (format == 2u)
                tagBytes += static_cast<uint64_t>(nloop) * 16u;
            else
            {
                reportReservedInstruction(false, 0xFFFFFFF8u);
                m_xgkick.active = false;
                return;
            }

            if (tagBytes > XgkickPipeline::kBufferSize - qwordOffset)
            {
                reportReservedInstruction(false, 0xFFFFFFFBu);
                m_xgkick.active = false;
                return;
            }
            m_xgkick.currentTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
            m_xgkick.currentTagEop = ((tagLo >> 15) & 1u) != 0u;
            if (m_xgkick.currentTagEop)
                m_xgkick.totalBytes = m_xgkick.currentTagEnd;
        }

        if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
        {
            if (m_xgkick.currentTagEop)
                finishXgkick();
            else
            {
                // The next transferred qword is another GIFtag.
                m_xgkick.currentTagEnd = 0u;
                m_xgkick.currentTagEop = false;
            }
        }
    }
}

void VU1Interpreter::finishXgkick()
{
    if (!m_xgkick.active)
        return;

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    {
        static std::atomic<uint32_t> s_xgk{0u};
        const uint32_t i = s_xgk.fetch_add(1u, std::memory_order_relaxed);
        if (i < 48u)
        {
            char b[264] = {0}; // 128 bytes * 2 hex chars + NUL (+ slack)
            const uint32_t n = std::min<uint32_t>(m_xgkick.totalBytes, 128u);
            for (uint32_t k = 0; k < n; ++k)
                std::snprintf(b + k * 2, 3, "%02x", m_xgkick.packet[k]);
            std::cerr << "[vu1:xgkick] #" << i << " bytes=" << m_xgkick.totalBytes
                      << " src=0x" << std::hex << m_xgkick.sourceAddress << std::dec
                      << " first128=" << b << std::endl;
        }
    }
#endif
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    const uint64_t gifSubmitT0 = s_vpGifOn ? __rdtsc() : 0u;
#endif
    if (s_vu1Flight && m_unit == Unit::VU1)
    {
        Vu1FlightEntry &e = s_vu1FlightRing[s_vu1FlightPos++ % kVu1FlightSize];
        e = {};
        e.pc = 0xFFFFFFFFu;
        e.cycle = g_ps2xVu1Kick.serial + 1u;
        e.upper = m_xgkick.sourceAddress;
        e.lower = m_xgkick.totalBytes;
    }
    const bool directMesh = t_directMesh.owner == this && t_directMesh.issue == m_xgkick.issueCycle;
    if (directMesh && m_gifCapture && !s_directMeshVerify)
        std::memcpy(m_xgkick.packet.data(), t_directMesh.raw, t_directMesh.bytes); // replay object/GIF comparison
    if (directMesh && s_directMeshVerify) {
        if (t_directMesh.expected.size() != m_xgkick.totalBytes ||
            std::memcmp(t_directMesh.expected.data(), m_xgkick.packet.data(), m_xgkick.totalBytes) != 0) {
            std::fprintf(stderr, "[vu1:direct-mesh] timed packet mismatch pc=%x src=%x\n", m_state.pc, m_xgkick.sourceAddress);
            std::abort();
        }
    }
    g_ps2xVu1Kick.packet = directMesh ? t_directMesh.raw : m_xgkick.packet.data();
    g_ps2xVu1Kick.bytes = m_xgkick.totalBytes;
    g_ps2xVu1Kick.pc = m_state.pc;
    g_ps2xVu1Kick.src = m_xgkick.sourceAddress;
    g_ps2xVu1Kick.top = m_state.top;
    g_ps2xVu1Kick.itop = m_state.itop;
    ++g_ps2xVu1Kick.serial;
    MotionProvenance::kick(this,g_ps2xVu1Kick.serial,m_xgkick.totalBytes,m_state.pc,s_vu1ImageHash);
    if (m_gifCapture)
        m_gifCapture->insert(m_gifCapture->end(), m_xgkick.packet.data(),
                             m_xgkick.packet.data() + m_xgkick.totalBytes);
    if (!m_gifCaptureOnly)
    {
        const uint64_t entryGifStart = s_entryTimingEnabled ? __rdtsc() : 0u;
        if (directMesh) {
            ps2xGsSubmitVuMesh(*m_activeGs, m_activeMemory, t_directMesh.tag, t_directMesh.vertices,
                              t_directMesh.adc, t_directMesh.count, t_directMesh.raw, t_directMesh.bytes);
            static thread_local uint64_t meshes = 0, vertices = 0;
            ++meshes; vertices += t_directMesh.count;
            if (meshes == 1 || meshes % 50000 == 0)
                std::fprintf(stderr, "[vu1:direct-mesh] meshes=%llu vertices=%llu verify=%u mismatches=0\n",
                    (unsigned long long)meshes, (unsigned long long)vertices, s_directMeshVerify ? 1u : 0u);
            t_directMesh.owner = nullptr;
        } else if (m_activeMemory)
            m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), m_xgkick.totalBytes);
        else if (m_activeGs)
            m_activeGs->processGIFPacket(m_xgkick.packet.data(), m_xgkick.totalBytes);
        if (s_entryTimingEnabled) s_entryGifTicks += __rdtsc() - entryGifStart;
    }
    g_ps2xVu1Kick.packet = nullptr;
    if (directMesh) t_directMesh.owner = nullptr;
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    if (s_vpGifOn)
        s_vpGifSubmitTicks.fetch_add(__rdtsc() - gifSubmitT0, std::memory_order_relaxed);
#endif
    m_xgkick.active = false;
}

void VU1Interpreter::startXgkick(uint32_t qwordAddress)
{
    if (m_unit != Unit::VU1 || !m_activeVuData || m_activeVuDataSize < 16u)
        return;

    // A second XGKICK while one is still in flight must preserve the unsent tail
    // of the first packet. Do not advance the complete VU pipeline recursively
    // from startXgkick: native relaxed execution can enter this function from an
    // instruction commit, and re-entering the pipeline here corrupts its state.
    // The normal interpreter readiness check stalls overlapping XGKICKs. This is
    // the safety fallback for native code that has already reached this point.
    if (m_xgkick.active)
    {
        static std::atomic<uint64_t> s_xgkickOverlaps{0u};
        const uint64_t n = s_xgkickOverlaps.fetch_add(1u, std::memory_order_relaxed) + 1u;
        if (n <= 20u || (n % 10000u) == 0u)
            std::fprintf(stderr, "[vu1:xgkick] overlapping XGKICK (#%llu) pc=0x%x copied=%u of %u\n",
                         static_cast<unsigned long long>(n), m_state.pc,
                         m_xgkick.copiedBytes, m_xgkick.totalBytes);
        const uint32_t savedCredit = m_xgkick.cycleCredit;
        m_xgkick.cycleCredit = 0x7FFFFFFFu;
        progressXgkick();
        if (m_xgkick.active)
            m_xgkick.cycleCredit = savedCredit;
    }

    const uint32_t sourceAddress = (qwordAddress * 16u) % m_activeVuDataSize;
    // Reset the bookkeeping only. `m_xgkick = {}` also zero-filled the 64 KB
    // packet buffer on every XGKICK (~8400 cycles measured); every byte that is
    // ever read back (tags in progressXgkick, [0, totalBytes) in finishXgkick)
    // is copied in first, so the old buffer contents are never observed.
    m_xgkick.sourceAddress = 0u;
    m_xgkick.totalBytes = 0u;
    m_xgkick.copiedBytes = 0u;
    m_xgkick.currentTagEnd = 0u;
    m_xgkick.cycleCredit = 0u;
    m_xgkick.issueCycle = 0u;
    m_xgkick.currentTagEop = false;
    m_xgkick.active = true;
    MotionProvenance::issue(this);
    m_xgkick.sourceAddress = sourceAddress;
    m_xgkick.cycleCredit = 1u; // XGKICK's issue cycle counts toward PATH1.
    m_xgkick.issueCycle = m_cycle;
    if (t_directMesh.owner == this) t_directMesh.owner = nullptr;
    // fd0 is followed only by integer updates and E-bit termination. Match the
    // exact suffix as well as the image, and reject any outstanding LSU store.
    // Straight-line native blocks keep the block-entry PC (fc0) until their
    // next control-flow instruction; the scalar interpreter reports fd0.
    // In particular, do not generalize this to the in-place clipper's kicks.
    static constexpr uint64_t terminal[] = {
        0x000002ff800016fcull, 0x000002ff100303e8ull,
        0x400002ff100e7090ull, 0x000002ff8000033cull
    };
    const bool captureDirect = s_directMeshReplay && t_vuBenchReplay && m_gifCaptureOnly;
    const uint8_t *directCode = captureDirect ? t_vuBenchReplay->code.data() : m_cachedVuCode;
    const uint32_t directCodeSize = captureDirect ? t_vuBenchReplay->header.codeSize : m_cachedCodeSize;
    if (s_directMeshEnabled && !t_directMeshReference && s_xgkickRateMode == 1u &&
        ((!m_gifCapture && !m_gifCaptureOnly && m_activeGs) || captureDirect) &&
        m_activeVuDataSize == 16384u && m_storeMask == 0u && (m_state.pc == 0xfd0u || m_state.pc == 0xfc0u) &&
        !m_state.branchPending && directCode && directCodeSize >= 0xff0u &&
        (s_vu1ImageHash == 0x7d7edbc78086ac63ull || s_vu1ImageHash == 0xd166d271aef9b896ull) &&
        std::memcmp(directCode + 0xfd0u, terminal, sizeof(terminal)) == 0) {
        uint64_t tag = 0, regs = 0;
        std::memcpy(&tag, m_activeVuData + sourceAddress, 8);
        std::memcpy(&regs, m_activeVuData + sourceAddress + 8, 8);
        const uint32_t n = uint32_t(tag & 0x7fffu), bytes = 16u + n * 48u;
        const uint32_t prim = uint32_t((tag >> 47) & 7u);
        if ((tag & (1ull << 15)) && (tag & (1ull << 46)) && ((tag >> 58) & 3u) == 0u &&
            (tag >> 60) == 3u && (regs & 0xfffu) == 0x412u && n >= 3u && n <= 341u &&
            sourceAddress + bytes <= 16384u && (prim == 3u || prim == 4u || prim == 5u)) {
            ++t_directMeshPrepared;
            auto &mesh = t_directMesh;
            mesh.owner = this; mesh.issue = m_cycle; mesh.tag = tag;
            mesh.count = n; mesh.bytes = bytes; mesh.raw = m_activeVuData + sourceAddress;
            for (uint32_t i = 0; i < n; ++i) {
                const uint8_t *p = mesh.raw + 16u + i * 48u;
                GSVertex &vertex = mesh.vertices[i];
                std::memcpy(&vertex.s, p, 4); std::memcpy(&vertex.t, p + 4, 4);
                std::memcpy(&vertex.q, p + 8, 4);
                if (vertex.q == 0.0f) vertex.q = 1.0f;
                vertex.r = p[16]; vertex.g = p[20]; vertex.b = p[24]; vertex.a = p[28];
                uint32_t x, y; uint64_t zf;
                std::memcpy(&x, p + 32, 4); std::memcpy(&y, p + 36, 4); std::memcpy(&zf, p + 40, 8);
                vertex.x = float(x & 0xffffu) / 16.0f; vertex.y = float(y & 0xffffu) / 16.0f;
                vertex.z = float((zf >> 4) & 0xffffffu); vertex.fog = uint8_t(zf >> 36);
                mesh.adc[i] = uint8_t((zf >> 47) & 1u);
            }
            if (s_directMeshVerify) mesh.expected.assign(mesh.raw, mesh.raw + bytes);
            else {
                m_xgkick.totalBytes = m_xgkick.currentTagEnd = bytes;
                m_xgkick.currentTagEop = true;
            }
        }
    }
    if (s_xgkickRateMode == 0u)
    {
        // PS2X_XGKICK_RATE=0: transfer the whole packet as VU memory holds it
        // at the XGKICK (credit covers any packet up to the 64 KB buffer).
        m_xgkick.cycleCredit = 0x7FFFFFFFu;
        progressXgkick();
    }
}

void VU1Interpreter::advanceOneCycle()
{
    ++m_cycle;
    m_state.cycles = m_cycle;
    // LSU commits become visible at the cycle boundary before PATH1 consumes
    // its next qword from VU memory.
    commitReadyPipelines();
    progressXgkick();
}

void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    while (m_cycle < targetCycle)
        advanceOneCycle();
}

bool VU1Interpreter::pipelinesPending()
{
    retireFlags();
    if (m_cycle < m_flagTailReady)
        return true;
    if (m_fdiv.valid || m_xgkick.active)
        return true;
    for (const ScalarPipelineEntry &entry : m_efu)
        if (entry.valid)
            return true;
    return (m_flagMask | m_storeMask | m_vfWriteMask | m_viWriteMask | m_accWriteMask) != 0u;
}

void VU1Interpreter::flushPipelines()
{
    while (pipelinesPending()) {
        if (t_directMesh.owner == this && !s_directMeshVerify && m_xgkick.active &&
            m_pipelineNextReady == ~0ull && m_flagTailReady <= m_cycle &&
            !(m_storeMask | m_vfWriteMask | m_viWriteMask | m_accWriteMask | m_flagMask) &&
            !m_flagOrderCount && !m_fdiv.valid && !m_efu[0].valid && !m_efu[1].valid) {
            const uint32_t cycles = ((m_xgkick.totalBytes - m_xgkick.copiedBytes) / 16u) * s_xgkickCyclesPerQw;
            m_cycle += cycles - std::min(cycles, m_xgkick.cycleCredit);
            m_state.cycles = m_cycle;
            m_xgkick.cycleCredit = 0u;
            m_xgkick.copiedBytes = m_xgkick.totalBytes;
            finishXgkick();
        } else if (s_tailBatchEnabled && !t_tailReference && m_xgkick.active &&
                   m_activeVuData && m_activeVuDataSize == 16384u && s_xgkickRateMode != 0u &&
                   m_pipelineNextReady == ~0ull && m_flagTailReady <= m_cycle &&
                   !(m_storeMask | m_vfWriteMask | m_viWriteMask | m_accWriteMask | m_flagMask) &&
                   !m_flagOrderCount && !m_fdiv.valid && !m_efu[0].valid && !m_efu[1].valid &&
                   m_xgkick.cycleCredit < s_xgkickCyclesPerQw &&
                   m_xgkick.currentTagEnd <= XgkickPipeline::kBufferSize &&
                   m_xgkick.currentTagEnd > m_xgkick.copiedBytes + 16u &&
                   ((m_xgkick.sourceAddress | m_xgkick.copiedBytes | m_xgkick.currentTagEnd) & 15u) == 0u) {
            // Only reached after VU execution has ended and all writes/flags
            // have retired. Preserve the already-transferred prefix, including
            // any data that changed while the VU was still running. Leave the
            // final qword and every tag boundary to the original state machine.
            const uint32_t bytes = m_xgkick.currentTagEnd - m_xgkick.copiedBytes - 16u;
            const uint64_t endCycle = m_cycle + (bytes / 16u) * s_xgkickCyclesPerQw - m_xgkick.cycleCredit;
            static thread_local std::unique_ptr<VU1Interpreter> reference;
            const bool check = s_tailBatchVerify && (++t_tailBatches % 1024u == 1u);
            if (check) {
                if (!reference) reference = std::make_unique<VU1Interpreter>(*this);
                else *reference = *this;
                // advanceTo uses the unchanged per-cycle transfer. No GS call
                // can occur: the final qword remains outstanding in both paths.
                reference->advanceTo(endCycle);
            }
            uint32_t remaining = bytes;
            while (remaining) {
                const uint32_t source = (m_xgkick.sourceAddress + m_xgkick.copiedBytes) & 16383u;
                const uint32_t span = std::min(remaining, 16384u - source);
                std::memcpy(m_xgkick.packet.data() + m_xgkick.copiedBytes, m_activeVuData + source, span);
                m_xgkick.copiedBytes += span;
                remaining -= span;
            }
            m_xgkick.cycleCredit = 0u;
            m_cycle = m_state.cycles = endCycle;
            if (check) {
                if (std::memcmp(this, reference.get(), sizeof(*this)) != 0) {
                    std::fprintf(stderr, "[vu1:tail-batch] state/packet mismatch pc=%x bytes=%u\n", m_state.pc, bytes);
                    std::abort();
                }
                static thread_local uint64_t checks = 0;
                if (++checks == 1 || checks % 1000u == 0)
                    std::fprintf(stderr, "[vu1:tail-batch] checked=%llu mismatches=0\n", (unsigned long long)checks);
            }
        } else advanceOneCycle();
    }
}

uint64_t VU1Interpreter::calculatePairReadyCycle(const DecodedInstructionPair &decoded) const
{
    // Every m_vfReady/m_viReady/m_accReady entry is <= m_maxReadyCycle, and the
    // scan below only ever raises `ready` towards one of them. Once the last
    // in-flight write has landed there is no hazard left to find, so skip the
    // ~50-iteration scan entirely -- the common case in straight-line code.
    if (m_maxReadyCycle <= m_cycle)
    {
        // Nothing is in flight at all, so no pending bit can still be true.
        m_vfPendingMask = 0u;
        m_viPendingMask = 0u;
        m_accPending = false;
        // NOTE: no early return. Register hazards need a write in flight, but
        // the Q / P / EFU / XGKICK waits at the end of this function do not --
        // returning here skipped them, so a WAITP with the EFU still busy did
        // not wait at all. Found by the native differential harness, which
        // does these checks unconditionally and so stalled where this did not.
        // With the masks cleared above, the register scan below is skipped
        // anyway, which is all the early return was worth.
    }

    // Full register-hazard scan (the original per-instruction loop). Exact by
    // construction; the fast path below only skips it when the pair reads no
    // register that could still have a write in flight.
    const auto scanRegisterHazards = [this, &decoded]() -> uint64_t
    {
        uint64_t r = m_cycle;
        const InstructionUsage *usages[2] = {
            &decoded.upperUsage,
            &decoded.lowerUsage};
        for (const InstructionUsage *usage : usages)
        {
            for (uint32_t index = 0; index < usage->vfReadCount; ++index)
            {
                const VfAccess &access = usage->vfRead[index];
                for (uint32_t component = 0; component < 4u; ++component)
                {
                    if ((access.lanes & laneForComponent(component)) != 0u)
                        r = std::max(r, m_vfReady[access.reg][component]);
                }
            }
            for (uint32_t viBits = static_cast<uint32_t>(usage->viRead) & 0xFFFEu; viBits != 0u; viBits &= viBits - 1u)
                r = std::max(r, m_viReady[static_cast<size_t>(std::countr_zero(viBits))]);
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((usage->accRead & laneForComponent(component)) != 0u)
                    r = std::max(r, m_accReady[component]);
            }
        }
        return r;
    };

    uint64_t ready = m_cycle;
    const uint32_t vfHit = decoded.vfReadMask & m_vfPendingMask;
    const uint32_t viHit = static_cast<uint32_t>(decoded.viReadMask & m_viPendingMask);
    const bool accHit = decoded.readsAcc && m_accPending;
    if ((vfHit | viHit) != 0u || accHit)
    {
        ready = scanRegisterHazards();
        // Retire pending bits whose last write has landed (m_cycle only grows,
        // so a cleared bit stays correct).
        for (uint32_t bits = vfHit; bits != 0u; bits &= bits - 1u)
        {
            const uint32_t reg = static_cast<uint32_t>(std::countr_zero(bits));
            const auto &lanes = m_vfReady[reg];
            if (std::max(std::max(lanes[0], lanes[1]), std::max(lanes[2], lanes[3])) <= m_cycle)
                m_vfPendingMask &= ~(1u << reg);
        }
        for (uint32_t bits = viHit; bits != 0u; bits &= bits - 1u)
        {
            const uint32_t reg = static_cast<uint32_t>(std::countr_zero(bits));
            if (m_viReady[reg] <= m_cycle)
                m_viPendingMask &= static_cast<uint16_t>(~(1u << reg));
        }
        if (accHit && std::max(std::max(m_accReady[0], m_accReady[1]), std::max(m_accReady[2], m_accReady[3])) <= m_cycle)
            m_accPending = false;
    }
    // PS2X_VU1_STALL_CHECK=1: prove the fast path never skips a real hazard by
    // running the full scan alongside and comparing (any build).
    static const bool s_stallCheck = std::getenv("PS2X_VU1_STALL_CHECK") != nullptr;
    if (s_stallCheck)
    {
        const uint64_t exact = scanRegisterHazards();
        static std::atomic<uint64_t> s_checked{0u}, s_bad{0u};
        const uint64_t n = s_checked.fetch_add(1u, std::memory_order_relaxed) + 1u;
        if (exact != ready)
        {
            const uint64_t b = s_bad.fetch_add(1u, std::memory_order_relaxed) + 1u;
            if (b <= 20u)
                std::fprintf(stderr, "[vu1:stallcheck] MISMATCH #%llu pc=0x%x fast=%llu exact=%llu cycle=%llu vfHit=%x viHit=%x acc=%d\n",
                             static_cast<unsigned long long>(b), m_state.pc,
                             static_cast<unsigned long long>(ready), static_cast<unsigned long long>(exact),
                             static_cast<unsigned long long>(m_cycle), vfHit, viHit, accHit ? 1 : 0);
        }
        if ((n % 5000000u) == 0u)
            std::fprintf(stderr, "[vu1:stallcheck] checked=%llu mismatches=%llu\n",
                         static_cast<unsigned long long>(n),
                         static_cast<unsigned long long>(s_bad.load(std::memory_order_relaxed)));
    }

    if (decoded.lowerUsage.pipeline == PipelineFdiv && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.pipeline == PipelineEfu)
        ready = std::max(ready, m_efuResourceReady);
    if (decoded.lowerUsage.waitQ && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.waitP)
    {
        for (const ScalarPipelineEntry &entry : m_efu)
            if (entry.valid)
                ready = std::max(ready, entry.readyCycle);
    }
    if (decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
        ready = std::max(ready, m_cycle + 1u);
    return ready;
}

void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
{
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u &&
        decoded.suppressedLowerVf != lowerWrite.reg)
    {
        const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                     ? decoded.lowerUsage.vfLatency
                                     : decoded.lowerUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((lowerWrite.lanes & laneForComponent(component)) != 0u)
                {
                    const uint64_t readyAt = m_cycle + latency;
                    m_vfReady[lowerWrite.reg][component] = readyAt;
                    m_vfPendingMask |= 1u << lowerWrite.reg;
                    if (readyAt > m_maxReadyCycle)
                        m_maxReadyCycle = readyAt;
                }
        }
    }

    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    if (upperWrite.reg != 0u)
    {
        const uint32_t latency = decoded.upperUsage.vfLatency != 0u
                                     ? decoded.upperUsage.vfLatency
                                     : decoded.upperUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((upperWrite.lanes & laneForComponent(component)) != 0u)
                {
                    const uint64_t readyAt = m_cycle + latency;
                    m_vfReady[upperWrite.reg][component] = readyAt;
                    m_vfPendingMask |= 1u << upperWrite.reg;
                    if (readyAt > m_maxReadyCycle)
                        m_maxReadyCycle = readyAt;
                }
        }
    }

    for (uint32_t viBits = static_cast<uint32_t>(decoded.lowerUsage.viWrite) & 0xFFFEu; viBits != 0u; viBits &= viBits - 1u)
    {
        const uint32_t reg = static_cast<uint32_t>(std::countr_zero(viBits));
            {
                const uint64_t readyAt = m_cycle + (decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency);
                m_viReady[reg] = readyAt;
                m_viPendingMask |= static_cast<uint16_t>(1u << reg);
                if (readyAt > m_maxReadyCycle)
                    m_maxReadyCycle = readyAt;
            }
    }
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((decoded.upperUsage.accWrite & laneForComponent(component)) != 0u)
            {
                const uint64_t readyAt = m_cycle + kAccForwardLatency;
                m_accReady[component] = readyAt;
                m_accPending = true;
                if (readyAt > m_maxReadyCycle)
                    m_maxReadyCycle = readyAt;
            }
    }
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    uint32_t lower = 0u;
    uint32_t upper = 0u;
    std::memcpy(&lower, vuCode + pc, sizeof(lower));
    std::memcpy(&upper, vuCode + pc + sizeof(lower), sizeof(upper));
    return decodePairWords(upper, lower, m_unit);
}

void VU1Interpreter::rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                             const PS2Memory *memory, uint64_t generation)
{
    const uint32_t pairCount = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);
    static const bool reuse=[] {
        const char *p=std::getenv("PS2X_VU_IMAGE_CACHE");return !p || p[0]!='0';
    }();
    static const bool verify=std::getenv("PS2X_VU_IMAGE_CACHE_VERIFY")!=nullptr;
    // VIF rewrites run between executions. Cache immutable copies instead of
    // trusting pointers or a generation belonging to a previous upload.
    static thread_local wotmVuImages::Cache<DecodedInstructionPair> images;
    static thread_local uint64_t hits=0,misses=0;
    const bool cacheable=reuse && codeSize!=0u && codeSize<=0x4000u;
    if(cacheable) if(const auto *entry=images.find(vuCode,codeSize,uint8_t(m_unit))) {
        if(verify) {
            for(uint32_t i=0;i<pairCount;++i)
                if(!wotmVuImages::samePair(entry->pairs[i],decodeInstructionPair(vuCode,i*8u))) {
                    std::fprintf(stderr,"[vu:image-cache] MISMATCH unit=%u pair=%u\n",unsigned(m_unit),i);
                    std::abort();
                }
            uint64_t hash=1469598103934665603ull;
            for(uint32_t i=0;i<codeSize;++i)hash=(hash^vuCode[i])*1099511628211ull;
            if(hash!=entry->hash)std::abort();
        }
        std::copy_n(entry->pairs.data(),pairCount,m_decodedCodeCache.data());
        m_cachedVuCode=vuCode;m_cachedMemory=memory;m_cachedCodeSize=codeSize;
        m_cachedCodeGeneration=generation;m_decodedCodeCacheValid=true;
        if(m_unit==Unit::VU1)s_vu1ImageHash=entry->hash;else s_vu0ImageHash=entry->hash;
        ++hits;
        if(verify && (hits==1u || hits%5000u==0u))
            std::fprintf(stderr,"[vu:image-cache] hits=%llu misses=%llu mismatches=0\n",
                (unsigned long long)hits,(unsigned long long)misses);
        return;
    }
    if(cacheable)++misses;
    for (uint32_t i = 0; i < pairCount; ++i)
        m_decodedCodeCache[i] = decodeInstructionPair(vuCode, i * 8u);

    m_cachedVuCode = vuCode;
    m_cachedMemory = memory;
    m_cachedCodeSize = codeSize;
    m_cachedCodeGeneration = generation;
    m_decodedCodeCacheValid = true;

    // VU1 recompiler inventory: this runs only when the game rewrites VU1
    // code memory, so hashing the image here is free. Each distinct image is
    // logged once and dumped to vu1_image_<hash>.bin as generator input.
    if (m_unit == Unit::VU0)
    {
        uint64_t hash = 1469598103934665603ull;
        for (uint32_t i = 0; i < codeSize; ++i)
            hash = (hash ^ vuCode[i]) * 1099511628211ull;
        s_vu0ImageHash = hash;
        static uint64_t s_seenVu0[16] = {};
        static uint32_t s_vu0Seen = 0u;
        bool seen = false;
        for (uint32_t i = 0; i < s_vu0Seen && i < 16u; ++i)
            seen |= s_seenVu0[i] == hash;
        if (!seen)
        {
            if (s_vu0Seen < 16u)
                s_seenVu0[s_vu0Seen] = hash;
            const uint32_t n = ++s_vu0Seen;
            std::fprintf(stderr, "[vu0:image] #%u hash=%016llx gen=%llu bytes=%u\n", n,
                         static_cast<unsigned long long>(hash), static_cast<unsigned long long>(generation), codeSize);
            char name[64];
            std::snprintf(name, sizeof(name), "vu0_image_%016llx.bin", static_cast<unsigned long long>(hash));
            if (std::FILE *f = std::fopen(name, "wb"))
            {
                std::fwrite(vuCode, 1u, codeSize, f);
                std::fclose(f);
            }
        }
    }
    if (m_unit == Unit::VU1)
    {
        uint64_t hash = 1469598103934665603ull;
        for (uint32_t i = 0; i < codeSize; ++i)
            hash = (hash ^ vuCode[i]) * 1099511628211ull;
        s_vu1ImageHash = hash;
        // The game swaps between several images many times a second, so log
        // and dump each distinct image once, not on every swap.
        static uint64_t s_seenImages[16] = {};
        static uint32_t s_imagesSeen = 0u;
        bool seen = false;
        for (uint32_t i = 0; i < s_imagesSeen && i < 16u; ++i)
            seen |= s_seenImages[i] == hash;
        if (!seen)
        {
            if (s_imagesSeen < 16u)
                s_seenImages[s_imagesSeen] = hash;
            const uint32_t n = ++s_imagesSeen;
            std::fprintf(stderr, "[vu1:image] #%u hash=%016llx gen=%llu bytes=%u\n", n,
                         static_cast<unsigned long long>(hash), static_cast<unsigned long long>(generation), codeSize);
            char name[64];
            std::snprintf(name, sizeof(name), "vu1_image_%016llx.bin", static_cast<unsigned long long>(hash));
            if (std::FILE *f = std::fopen(name, "wb"))
            {
                std::fwrite(vuCode, 1u, codeSize, f);
                std::fclose(f);
            }
        }
    }
    if(cacheable)images.save(vuCode,codeSize,uint8_t(m_unit),m_decodedCodeCache.data(),
        pairCount,m_unit==Unit::VU1?s_vu1ImageHash:s_vu0ImageHash);
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::getDecodedInstructionPairForPc(
    const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc)
{
    if ((pc & 7u) != 0u)
        return decodeInstructionPair(vuCode, pc);

    const bool trackedVu1Code = memory != nullptr &&
                                ((m_unit == Unit::VU1 && vuCode == memory->getVU1Code()) ||
                                 (m_unit == Unit::VU0 && vuCode == memory->getVU0Code()));
    if (!trackedVu1Code)
        return decodeInstructionPair(vuCode, pc);

    const uint64_t generation = m_unit == Unit::VU1 ? memory->getVU1CodeGeneration() : memory->getVU0CodeGeneration();
    if (!m_decodedCodeCacheValid ||
        m_cachedVuCode != vuCode ||
        m_cachedMemory != memory ||
        m_cachedCodeSize != codeSize ||
        m_cachedCodeGeneration != generation)
    {
        rebuildDecodedCodeCache(vuCode, codeSize, memory, generation);
    }
    const uint32_t pairIndex = pc / 8u;
    if (pairIndex >= kMaxDecodedPairs)
        return decodeInstructionPair(vuCode, pc);
    return m_decodedCodeCache[pairIndex];
}

void VU1Interpreter::reportReservedInstruction(bool upper, uint32_t instruction)
{
    RUNTIME_ERROR(
        "[VU" << (m_unit == Unit::VU1 ? "1" : "0")
              << " reserved " << (upper ? "upper" : "lower")
              << "] cycle=" << m_cycle
              << " pc=0x" << std::hex << m_state.pc
              << " instruction=0x" << instruction
              << std::dec << '\n');
    m_stopRequested = true;
}

void VU1Interpreter::execute(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory,
                             uint32_t startPC, uint32_t top, uint32_t itop,
                             uint32_t maxCycles)
{
    resetScheduler();
    m_state.pc = startPC & microAddressMask();
    if (m_unit == Unit::VU1)
        g_ps2xVu1Kick.startPc = m_state.pc;
    m_state.ebit = false;
    m_state.haltAfterDelaySlot = false;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    {
        static std::atomic<uint32_t> s_ex{0u};
        // VU1 only (16KB micro mem); skip the 4KB VU0 spam.
        // Settled-gate: only the stable screenMain (betweenScreens==0 && currScreen==1)
        // — earlier caps filled with pre-settle frames (stale-diagnostic trap, s13#11).
        const bool execSettled = memory &&
            (memory->read32(0x6F7E8Cu) == 0u && memory->read32(0x6F8464u) == 1u);
        if (codeSize >= 8192u && execSettled)
        {
            const uint32_t i = s_ex.fetch_add(1u, std::memory_order_relaxed);
            if (i < 300u)
            {
                uint64_t w0 = 0, w1 = 0;
                if (startPC + 16u <= codeSize && vuCode)
                {
                    std::memcpy(&w0, vuCode + startPC, 8);
                    std::memcpy(&w1, vuCode + startPC + 8, 8);
                }
                auto dumpVf = [&](int r) {
                    std::cerr << " vf" << r << "=(" << m_state.vf[r][0] << "," << m_state.vf[r][1]
                              << "," << m_state.vf[r][2] << "," << m_state.vf[r][3] << ")";
                };
                const uint64_t cyc0 = m_cycle;
                std::cerr << "[vu1:exec] #" << i << " startPC=0x" << std::hex << startPC << std::dec
                          << " top=" << top << " itop=" << itop;
                if (vuData && dataSize >= 64u)
                {
                    const float *d = reinterpret_cast<const float *>(vuData);
                    // q0..q20 — covers the four V4-32 matrices the WotM menu render
                    // UNPACKs to VU1 data q5/q9/q13/q17 (see [vif1:unpack]).
                    for (int q = 0; q < 21; ++q)
                        std::fprintf(stderr, " q%d=(%.4g,%.4g,%.4g,%.4g)", q,
                                     d[q * 4], d[q * 4 + 1], d[q * 4 + 2], d[q * 4 + 3]);
                }
                // Dump VU1 data mem where VIF1 UNPACKed this frame's geometry
                // (chain pkt #5 VIFcodes target qword ~0x238..0x260).
                for (uint32_t qw : {0x238u, 0x239u, 0x23au, 0x23bu, 0x247u, 0x250u})
                {
                    const uint32_t off = qw * 16u;
                    if (vuData && off + 16u <= dataSize)
                    {
                        const uint8_t *p = vuData + off;
                        std::fprintf(stderr, " q%03x=%02x%02x%02x%02x_%02x%02x%02x%02x_%02x%02x%02x%02x_%02x%02x%02x%02x",
                                     qw, p[0],p[1],p[2],p[3], p[4],p[5],p[6],p[7],
                                     p[8],p[9],p[10],p[11], p[12],p[13],p[14],p[15]);
                    }
                }
                std::cerr << std::endl;
                run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
                std::cerr << "[vu1:exec] #" << i << " END pc=0x" << std::hex << m_state.pc << std::dec
                          << " cycles=" << (m_cycle - cyc0)
                          << " ebit=" << m_state.ebit
                          << " stoppedD=" << m_state.stoppedByD
                          << " stoppedT=" << m_state.stoppedByT;
                dumpVf(1); dumpVf(2); dumpVf(3); dumpVf(4);
                std::cerr << std::endl;
                std::cerr << "[vu1:exec] #" << i << " REGS";
                for (int r : {5, 7, 8, 18, 19, 20, 21, 22, 24, 25, 26})
                    dumpVf(r);
                std::cerr << " Q=" << m_state.q
                          << " ACC=(" << m_state.acc[0] << "," << m_state.acc[1] << ","
                          << m_state.acc[2] << "," << m_state.acc[3] << ")" << std::endl;
                return;
            }
        }
    }
#endif
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::resume(uint8_t *vuCode, uint32_t codeSize,
                            uint8_t *vuData, uint32_t dataSize,
                            GS &gs, PS2Memory *memory,
                            uint32_t top, uint32_t itop, uint32_t maxCycles)
{
    m_state.top = top;
    m_state.itop = itop;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

struct Ps2xPerf { std::atomic<uint64_t> rasterNs, presentNs, vu1Ns, vif1Ns, prims; };
extern Ps2xPerf g_ps2xPerf;
std::atomic<uint64_t> g_ps2xVuInstr{0u};
std::atomic<uint64_t> g_ps2xVuRuns{0u};
namespace
{
    // These shared totals are read only by optional performance reports.
    // Normal play avoids two contended, locked writes per VU0/VU1 run.
    // Keep reporting exact when requested; an explicit override also lets
    // frame-time tests measure shipping execution without collecting VU work.
    const bool s_collectVuWork = [] {
        if (const char* value = std::getenv("PS2X_VU_WORK_COUNTERS")) {
            const bool enabled = value[0] != '0';
            std::fprintf(stderr, "[vu:work-counters] enabled=%u (diagnostic statistics only)\n",
                         unsigned(enabled));
            return enabled;
        }
        return std::getenv("PS2X_PERF") != nullptr ||
               std::getenv("PS2X_FRAME_CSV") != nullptr ||
               std::getenv("PS2X_DESTRUCTION_CSV") != nullptr;
    }();
}

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
// [vu1:prof] — temporary rdtsc breakdown of VU1Interpreter::run's inner loop.
// chrono::now() is ~20-25 ns and would swamp a ~200 ns instruction; rdtsc is
// ~15 cycles, so the proportions stay meaningful. Diag build only; delete once
// the hot spot is settled.
namespace
{
    // flush = the program-end drain after the loop (advanceTo + flushPipelines),
    // which the per-pair buckets never saw.
    enum VuProfBucket { kVpDecode = 0, kVpReady, kVpShadowIn, kVpExec, kVpShadowOut, kVpMark, kVpAdvance, kVpTail, kVpFlush, kVpCount };
    const char *const kVpNames[kVpCount] = {"decode", "ready", "shadowIn", "exec", "shadowOut", "mark", "advance", "tail", "flush"};
    // Emulated cycles vs retired pairs (stalls show up as cycles > pairs).
    uint64_t g_vpCycles = 0u, g_vpPairs = 0u, g_vpFlushCycles = 0u, g_vpRuns = 0u;
    std::atomic<uint64_t> g_vpTicks[kVpCount]{};
    std::atomic<uint64_t> g_vpTotal{0u};

    // Opt-in: set PS2X_VU1_PROF=1. An rdtsc on every phase boundary costs ~13%,
    // which would distort the very numbers it reports, so the diagnostic build
    // pays nothing for this unless it is asked for.
    const bool g_vpOn = std::getenv("PS2X_VU1_PROF") != nullptr;
    inline void vpAdd(int bucket, uint64_t ticks) { g_vpTicks[bucket].fetch_add(ticks, std::memory_order_relaxed); }

    // Per-opcode exec cost. Only the game thread runs VU code, so plain arrays.
    // Upper key: op (<0x3C) or 0x40+special. Lower key: opHi (<0x40),
    // 0x40+funct for lower1 integer ops, 0x80+funct2 for lower1 specials.
    uint64_t g_vpUpTicks[256]{}, g_vpUpCount[256]{};
    uint64_t g_vpLoTicks[256]{}, g_vpLoCount[256]{};
    inline uint32_t vpUpperKey(uint32_t upper)
    {
        const uint32_t op = upper & 0x3Fu;
        return op < 0x3Cu ? op : 0x40u + ((upper & 3u) | ((upper >> 4) & 0x7Cu));
    }
    inline uint32_t vpLowerKey(uint32_t lower)
    {
        const uint32_t opHi = (lower >> 25) & 0x7Fu;
        if (opHi != 0x40u)
            return opHi & 0x3Fu;
        const uint32_t funct = lower & 0x3Fu;
        if ((funct & 0x3Cu) != 0x3Cu)
            return 0x40u + funct;
        return 0x80u + (((lower & 3u) | ((lower >> 4) & 0x7Cu)) & 0x7Fu);
    }
    void vpDumpOps(const char *name, uint64_t *ticks, uint64_t *count, uint64_t total)
    {
        std::vector<std::pair<uint64_t, uint32_t>> order;
        for (uint32_t k = 0; k < 256u; ++k)
            if (count[k] != 0u)
                order.emplace_back(ticks[k], k);
        std::sort(order.rbegin(), order.rend());
        std::string line = std::string("[vu1:ops] ") + name;
        char buf[96];
        for (size_t i = 0; i < order.size() && i < 14u; ++i)
        {
            const uint32_t k = order[i].second;
            std::snprintf(buf, sizeof(buf), " %02x:n=%llu,avg=%.0fcyc,%.1f%%", k,
                          static_cast<unsigned long long>(count[k]),
                          double(ticks[k]) / double(count[k]),
                          total ? 100.0 * double(ticks[k]) / double(total) : 0.0);
            line += buf;
        }
        std::fprintf(stderr, "%s\n", line.c_str());
        for (uint32_t k = 0; k < 256u; ++k)
            ticks[k] = count[k] = 0u;
    }

    void vpMaybeDump()
    {
        static std::atomic<uint64_t> s_last{0u};
        const uint64_t now = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
        uint64_t prev = s_last.load(std::memory_order_relaxed);
        if (now < prev + 5u)
            return;
        if (!s_last.compare_exchange_strong(prev, now, std::memory_order_relaxed))
            return;
        uint64_t total = 0u;
        uint64_t vals[kVpCount];
        for (int i = 0; i < kVpCount; ++i) { vals[i] = g_vpTicks[i].exchange(0u, std::memory_order_relaxed); total += vals[i]; }
        if (total == 0u)
            return;
        std::string line = "[vu1:prof]";
        char buf[64];
        for (int i = 0; i < kVpCount; ++i)
        {
            std::snprintf(buf, sizeof(buf), " %s=%.1f%%", kVpNames[i], 100.0 * double(vals[i]) / double(total));
            line += buf;
        }
        const uint64_t gifSubmit = s_vpGifSubmitTicks.exchange(0u, std::memory_order_relaxed);
        std::snprintf(buf, sizeof(buf), " | gifSubmit(within advance)=%.1f%%", 100.0 * double(gifSubmit) / double(total));
        line += buf;
        if (g_vpPairs != 0u && g_vpRuns != 0u)
        {
            std::snprintf(buf, sizeof(buf), " | cyc/pair=%.2f flushCyc/run=%.0f pairs/run=%.0f",
                          double(g_vpCycles) / double(g_vpPairs), double(g_vpFlushCycles) / double(g_vpRuns),
                          double(g_vpPairs) / double(g_vpRuns));
            line += buf;
        }
        g_vpCycles = g_vpPairs = g_vpFlushCycles = g_vpRuns = 0u;
        std::fprintf(stderr, "%s\n", line.c_str());
        vpDumpOps("upper", g_vpUpTicks, g_vpUpCount, total);
        vpDumpOps("lower", g_vpLoTicks, g_vpLoCount, total);
    }
}
#define VP_BEGIN() uint64_t vpPrev = g_vpOn ? __rdtsc() : 0u
#define VP_MARK(b) do { if (g_vpOn) { const uint64_t vpNow = __rdtsc(); vpAdd((b), vpNow - vpPrev); vpPrev = vpNow; } } while (0)
#else
#define VP_BEGIN() ((void)0)
#define VP_MARK(b) ((void)0)
#endif

// safebuffers: no /GS stack-cookie check on the VU1 hot path (0.9% of the game
// thread in the sampling profile); its local buffers are fixed-size and internal.
#if defined(_MSC_VER)
__declspec(safebuffers)
#endif
void VU1Interpreter::run(uint8_t *vuCode, uint32_t codeSize,
                         uint8_t *vuData, uint32_t dataSize,
                         GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    // TEMP [vu1:entries] per-entry cost census (image, startPc).
    struct EntryScope
    {
        VU1Interpreter *self;
        bool enabled;
        uint32_t pc;
        uint64_t t0, c0, g0;
        uint32_t sel;
        ~EntryScope()
        {
            if (!enabled || self->m_unit != Unit::VU1)
                return;
            // ABCD = ff0 dispatch; DCBA = 1ea0 dispatch. Preserve image identity
            // for the latter so unrelated microprograms cannot merge.
            const uint64_t image = pc == 0xff0u ? 0xABCD000000000000ull :
                pc == 0x1ea0u ? (0xDCBA000000000000ull | (s_vu1ImageHash & 0xffffffffffffull)) : s_vu1ImageHash;
            ps2xVu1EntryNote(image, (pc == 0xff0u || pc == 0x1ea0u) ? sel : pc,
                              __rdtsc() - t0, self->m_cycle - c0, s_entryGifTicks - g0);
        }
    };
    EntryScope entryScope{this, s_entryTimingEnabled, m_state.pc & 0x3FFFu,
                          s_entryTimingEnabled ? __rdtsc() : 0u, m_cycle, s_entryGifTicks,
                          (static_cast<uint16_t>(m_state.vi[13]) * 8u) & 0x3fffu};
    struct PerfScope
    {
        bool enabled;
        Ps2xTscClock::time_point t0;
        ~PerfScope()
        {
            if (!enabled)
                return;
            g_ps2xPerf.vu1Ns.fetch_add(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Ps2xTscClock::now() - t0).count()),
                std::memory_order_relaxed);
        }
    };
    static const bool s_perfTimingEnabled = std::getenv("PS2X_PERF") != nullptr;
    PerfScope perfScope{s_perfTimingEnabled,
                        s_perfTimingEnabled ? Ps2xTscClock::now() : Ps2xTscClock::time_point{}};
    if(t_vuBenchReplay) {
        std::memcpy(this,t_vuBenchReplay->object.data(),sizeof(*this));
        maxCycles=t_vuBenchReplay->header.maxCycles;
        s_vu1ImageHash=t_vuBenchReplay->header.image;
        m_cachedVuCode=nullptr;m_cachedMemory=nullptr;m_decodedCodeCacheValid=false;
        m_gifCapture=nullptr;
    }
    m_activeVuData = vuData;
    m_activeVuDataSize = dataSize;
    m_activeGs = &gs;
    m_activeMemory = memory;

    if (s_collectVuWork) g_ps2xVuRuns.fetch_add(1u, std::memory_order_relaxed);
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    if (vu1scope::enabled && m_unit == Unit::VU1)
        vu1scope::onRun(vuCode, codeSize, memory, m_state.pc);
#endif
    // The harness always runs the real pass exact and the shadow pass fast.
    if (!t_vu1InShadow && !t_vuBenchReplay)
        m_immediateWrites = s_vu1FastMode && !s_vu1DiffMode;
    bool diffArmed = false;
    if (s_vu1DiffMode && m_unit == Unit::VU1 && !t_vu1InShadow)
    {
        diffBegin(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
        diffArmed = true;
    }
    const int previousRoundingMode = std::fegetround();
    const bool useVuRounding = std::fesetround(FE_TOWARDZERO) == 0;
    const uint64_t budgetEnd = m_cycle + maxCycles;
    bool programEnded = false;
    // Decoded pairs by reference: getDecodedInstructionPairForPc returns the
    // ~100-byte pair by value and re-checks the code generation on every
    // instruction. VU code cannot change while run() executes (VIF MPG writes
    // happen on this thread, between runs), so validate/rebuild the cache once
    // here and index it directly; anything unusual falls back to the old call.
    bool directDecode = false;
    if ((m_state.pc & 7u) == 0u && m_state.pc + 8u <= codeSize && memory != nullptr &&
        ((m_unit == Unit::VU1 && vuCode == memory->getVU1Code()) ||
         (m_unit == Unit::VU0 && vuCode == memory->getVU0Code())))
    {
        (void)getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc);
        directDecode = m_decodedCodeCacheValid && m_cachedVuCode == vuCode &&
                       m_cachedMemory == memory && m_cachedCodeSize == codeSize;
    }
    if(t_vuBenchReplay)directDecode=true;
    DecodedInstructionPair decodedFallback;
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    const uint64_t vpCycleAtStart = m_cycle;
#endif
    // Instructions retired this run; published once at the end instead of a
    // locked add per instruction.
    uint64_t instrCount = 0u;
    // VU1 native programs: the same per-instruction sequence as the loop
    // below, generated per code image with every decode-derived quantity a
    // compile-time constant (ps2_vu1_native.h). Exact by construction -- it
    // calls the interpreter's own execUpper/execLower/timing code -- and
    // gated on the diagnostics that need the interpreter loop.
    // PS2X_VU1_NATIVE=0 disables.
    static const bool s_nativeEnabled = []
    {
        const char *value = std::getenv("PS2X_VU1_NATIVE");
        const bool off = value != nullptr && value[0] == '0';
        // The self-check modes compare inside the interpreter's own paths, so
        // they run the interpreter (the native path is validated by the diag
        // build's PS2X_VU1_DIFF harness instead).
        return !off && std::getenv("PS2X_VU1_FMAC_SLOW") == nullptr && std::getenv("PS2X_VU1_PROF") == nullptr &&
               std::getenv("PS2X_VU1_FLAG_CHECK") == nullptr && std::getenv("PS2X_VU1_STALL_CHECK") == nullptr;
    }();
    bool nativeHandled = false;
    // PS2X_VU1_DIFF (diag builds): the authoritative pass is the exact
    // interpreter and the shadow pass runs fast mode, so native code runs in
    // the shadow only. The harness then compares native against exact.
    const bool nativeAllowedHere = s_vu1DiffMode ? t_vu1InShadow : !diffArmed;
    // VU0 micro-programs (vcallms) run native code too: a small dedicated
    // dispatch, kept apart from the VU1 path's image-specific blocks and bench
    // hooks. PS2X_VU0_NATIVE=0 disables; t_vu0ForceInterpreter is set by the
    // PS2X_VU0_NATIVE_VERIFY comparison in ps2_runtime.cpp.
    static const bool s_vu0NativeEnabled = []
    {
        const char *value = std::getenv("PS2X_VU0_NATIVE");
        return value == nullptr || value[0] != '0';
    }();
    if (s_nativeEnabled && s_vu0NativeEnabled && m_unit == Unit::VU0 && !t_vu0ForceInterpreter &&
        directDecode && m_immediateWrites)
    {
        if (const Vu1NativeProgram program = ps2xVu1NativeLookup(s_vu0ImageHash))
        {
            Vu1NativeCtx nctx{vuData, dataSize, codeSize, &gs, memory, budgetEnd, 0u};
            const Vu1NativeStep step = program(*this, nctx);
            instrCount += nctx.instrCount;
            if (step == Vu1NativeStep::Ended)
            {
                programEnded = true;
                nativeHandled = true;
            }
            else if (step == Vu1NativeStep::Stop)
            {
                nativeHandled = true;
            }
        }
    }
    if (s_nativeEnabled && m_unit == Unit::VU1 && directDecode && m_immediateWrites &&
        !s_vu1Flight && nativeAllowedHere)
    {
        if (const Vu1NativeProgram program = ps2xVu1NativeLookup(s_vu1ImageHash))
        {
            // Opt-in isolated replay: the live interpreter and DMEM are never
            // modified; XGKICK packets are captured, not submitted to rendering.
            static const bool benchEnabled = std::getenv("PS2X_VU1_BENCH") != nullptr;
            extern std::atomic<uint64_t> g_ps2xWotmCompletedFrames;
            static std::map<std::pair<uint64_t,uint32_t>,unsigned> benchSamples;
            static unsigned benchTotal = 0;
            static const uint64_t benchFirstFrame=[] {const char *p=std::getenv("PS2X_VU1_BENCH_START_FRAME");return p?std::strtoull(p,nullptr,10):1600ull;}();
            static const double benchFirstSeconds=[] {const char *p=std::getenv("PS2X_VU1_BENCH_START_SECONDS");return p?std::strtod(p,nullptr):0.0;}();
            static const auto benchEpoch=std::chrono::steady_clock::now();

            const auto benchKey = std::make_pair(s_vu1ImageHash,m_state.pc);
            const uint32_t captureShader=(m_state.pc==0xff0u || m_state.pc==0x1ea0u) ? ((static_cast<uint16_t>(m_state.vi[13])*8u)&0x3fffu) : 0u;
            static const uint32_t captureOnlyShader=[] {const char *p=std::getenv("PS2X_VU1_BENCH_SHADER");return p?uint32_t(std::strtoul(p,nullptr,0)):0u;}();
            const auto captureKey=std::make_pair(s_vu1ImageHash,m_state.pc|(captureShader<<16));
            static const bool liveVerifyEnabled=[] { const char *v=std::getenv("PS2X_VU1_BLOCK_VERIFY");return v && v[0]!='0'; }();
            static const bool liveBlockEnabled=[] { const char *v=std::getenv("PS2X_VU1_BLOCK");return !v || v[0]!='0'; }();
            static bool liveBlockFailed=false;
            // Both registered images have identical instruction pairs and timing
            // metadata throughout the compact blocks (all below 0x34f8).
            static const bool menuBlocks=[] {const char *p=std::getenv("PS2X_VU1_MENU_BLOCKS");return !p || p[0]!='0';}();
            const bool compactImage=s_vu1ImageHash==0x7d7edbc78086ac63ull ||
                (menuBlocks && s_vu1ImageHash==0xd166d271aef9b896ull);
            const bool blockCandidate=compactImage && m_state.pc==0x20;
            const bool heavyCandidate=compactImage && (m_state.pc==0xff0 || m_state.pc==0x1ea0);
            if(s_stretchPath && !t_stretchSaved && !t_vuBenchReplay && !t_vu1InShadow) {
                t_stretchInput.clear();
                const uint64_t frame=g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);
                if(s_vu1ImageHash==0x7d7edbc78086ac63ull && g_ps2xVu1Kick.startPc==0x20 && frame>=1600 && frame<=2200) {
                    auto snapshot=std::make_unique<VU1Interpreter>(*this);
                    snapshot->m_cachedVuCode=nullptr;snapshot->m_cachedMemory=nullptr;
                    snapshot->m_activeVuData=nullptr;snapshot->m_activeGs=nullptr;snapshot->m_activeMemory=nullptr;snapshot->m_gifCapture=nullptr;
                    VuBenchHeader h;h.image=s_vu1ImageHash;h.codeSize=codeSize;h.dataSize=dataSize;h.maxCycles=maxCycles;
                    t_stretchInput.resize(sizeof(h)+sizeof(*snapshot)+codeSize+dataSize);
                    auto *out=t_stretchInput.data();std::memcpy(out,&h,sizeof(h));out+=sizeof(h);
                    std::memcpy(out,snapshot.get(),sizeof(*snapshot));out+=sizeof(*snapshot);
                    std::memcpy(out,vuCode,codeSize);out+=codeSize;std::memcpy(out,vuData,dataSize);
                    if(blockCandidate)t_stretchMeshInput=t_stretchInput;
                }
            }
            static const bool liveHeavyEnabled=[] { const char *v=std::getenv("PS2X_VU1_HEAVY_BLOCK");return !v || v[0]!='0'; }();
            static uint64_t heavyVerifyCandidates=0;
            // Verify the initial calls and sample the high-frequency paths.
            const bool verifyHeavy=liveVerifyEnabled && heavyCandidate && !t_vuBenchReplay &&
                (++heavyVerifyCandidates<=64 || heavyVerifyCandidates%64==0);
            const bool verifyBlock=liveVerifyEnabled && (blockCandidate || verifyHeavy) && !liveBlockFailed && !t_vuBenchReplay;
#include "ps2_vu1_object_kernels.inc"
#include "ps2_vu1_menu_kernel.inc"
#include "ps2_vu1_menu_loop.inc"
            if(!g_ps2xVu1ShaderTransform)g_ps2xVu1ShaderTransform=+[](VU1Interpreter &v,Vu1NativeCtx &c) {
                // 1a60..1a70: MADDAx/MADDAy/MADDz, with integer lower halves.
                constexpr uint32_t U0=locallyGeneratedVu::upper<0x0060u>,L0=locallyGeneratedVu::lower<0x1a60u>,U1=locallyGeneratedVu::upper<0x0068u>,L1=locallyGeneratedVu::lower<0x00c0u>,U2=locallyGeneratedVu::upper<0x00d0u>,L2=locallyGeneratedVu::lower<0x0130u>;
                const auto fallback=[&] {
                    ++g_ps2xVu1MatrixFallbacks;
                    auto r=Vu1NativeAccess::stepPair<U0,L0,true,true,true,true,true>(v,c);
                    if(r!=Vu1NativeStep::Continue)return r;
                    r=Vu1NativeAccess::stepPair<U1,L1,true,true,true,true,true>(v,c);
                    if(r!=Vu1NativeStep::Continue)return r;
                    return Vu1NativeAccess::stepPair<U2,L2,true,true,true,true,true>(v,c);
                };
                if(v.m_pipelineNextReady<=v.m_cycle+3 || v.m_xgkick.active || !v.m_immediateWrites)return fallback();
                const __m128 column=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[20]));
                const __m128 x=_mm_shuffle_ps(column,column,0),y=_mm_shuffle_ps(column,column,0x55),z=_mm_shuffle_ps(column,column,0xaa);
                const __m128 a=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[1]));
                const __m128 b=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[2]));
                const __m128 d=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[3]));
                const __m128 p0=_mm_mul_ps(a,x),p1=_mm_mul_ps(b,y),p2=_mm_mul_ps(d,z);
                const __m128 r0=_mm_add_ps(_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.acc)),p0);
                const __m128 r1=_mm_add_ps(r0,p1),r2=_mm_add_ps(r1,p2);
                const auto sum=[](__m128 p,__m128 result,__m128 l,__m128 r){const auto zero=vu1n::vuZeroLanes(l)|vu1n::vuZeroLanes(r);return (vu1n::vuOrdinaryLanes(p)|zero)&(vu1n::vuOrdinaryLanes(result)|(zero&vu1n::vuZeroLanes(result)));};
                if((sum(p0,r0,a,x)&sum(p1,r1,b,y)&sum(p2,r2,d,z))!=15)return fallback();
                ++g_ps2xVu1MatrixHits;
                const auto lower=[&]<uint32_t U,uint32_t L>() {
                    constexpr auto decoded=Vu1NativeAccess::PairDecode<U,L>::d;
                    constexpr uint32_t bits=uint32_t(decoded.lowerUsage.viWrite)&0xfffeu;
                    constexpr uint8_t reg=bits?uint8_t(std::countr_zero(bits)):0;
                    const int32_t oldVi=reg?v.m_state.vi[reg]:0;
                    Vu1NativeAccess::execLowerN<U,L>(v,c);
                    v.m_viBranchBackupValid=false;
                    if constexpr(reg && decoded.lowerUsage.delaysNextBranchRead)v.recordViWriteForBranch(reg,oldVi);
                };
                // Preserve even the inactive branch-backup metadata produced
                // by the middle integer write, just as three native pairs do.
                lower.operator()<U0,L0>();++v.m_cycle;
                lower.operator()<U1,L1>();++v.m_cycle;
                lower.operator()<U2,L2>();
                _mm_storeu_ps(v.m_state.acc,r1);_mm_storeu_ps(v.m_state.vf[(U2>>6)&31],r2);
                v.m_currentUpperInstruction=U2;
                v.m_state.vf[0][0]=v.m_state.vf[0][1]=v.m_state.vf[0][2]=0;v.m_state.vf[0][3]=1;v.m_state.vi[0]=0;
                c.instrCount+=3;Vu1NativeAccess::advanceOneCycleFast(v);return Vu1NativeStep::Continue;
            };
            if(!g_ps2xVu1GameMatrix[0]) {
                const auto triple=[]<uint32_t U0,uint32_t L0,uint32_t U1,uint32_t L1,uint32_t U2,uint32_t L2,bool Acc>(VU1Interpreter &v,Vu1NativeCtx &c)->Vu1NativeStep {
                    static_assert(vu1n::fmacForm((U0&3)|((U0>>4)&0x7c)).kind==(Acc?vu1n::FmacKind::Mul:vu1n::FmacKind::Madd));
                    static_assert(((U0>>11)&31)==15 && ((U1>>11)&31)==16 && ((U2>>11)&31)==11);
                    static_assert(((U0>>16)&31)==((U1>>16)&31) && ((U0>>16)&31)==((U2>>16)&31));
                    const auto fallback=[&] {
                        ++g_ps2xVu1MatrixFallbacks;
                        auto r=Vu1NativeAccess::stepPair<U0,L0,true,true,true,true,true>(v,c);
                        if(r!=Vu1NativeStep::Continue)return r;
                        r=Vu1NativeAccess::stepPair<U1,L1,true,true,true,true,true>(v,c);
                        if(r!=Vu1NativeStep::Continue)return r;
                        return Vu1NativeAccess::stepPair<U2,L2,true,true,true,true,true>(v,c);
                    };
                    if(v.m_pipelineNextReady<=v.m_cycle+3 || v.m_xgkick.active || !v.m_immediateWrites)return fallback();
                    const __m128 column=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[(U0>>16)&31]));
                    const __m128 x=_mm_shuffle_ps(column,column,0),y=_mm_shuffle_ps(column,column,0x55),z=_mm_shuffle_ps(column,column,0xaa);
                    const __m128 a=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[15]));
                    const __m128 b=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[16]));
                    const __m128 d=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[11]));
                    const __m128 p0=_mm_mul_ps(a,x),p1=_mm_mul_ps(b,y),p2=_mm_mul_ps(d,z);
                    // The first group starts with MADDAx (not MULAx): its
                    // incoming ACC contains the translation term from 1ba0.
                    const __m128 r0=Acc?p0:_mm_add_ps(_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.acc)),p0);
                    const __m128 r1=_mm_add_ps(r0,p1),r2=_mm_add_ps(r1,p2);
                    const auto product=[](__m128 p,__m128 l,__m128 r) {return vu1n::vuOrdinaryLanes(p)|((vu1n::vuZeroLanes(l)|vu1n::vuZeroLanes(r))&vu1n::vuZeroLanes(p));};
                    const auto sum=[](__m128 p,__m128 result,__m128 l,__m128 r) {
                        const auto zero=vu1n::vuZeroLanes(l)|vu1n::vuZeroLanes(r);
                        return (vu1n::vuOrdinaryLanes(p)|zero)&(vu1n::vuOrdinaryLanes(result)|(zero&vu1n::vuZeroLanes(result)));
                    };
                    if(((Acc?product(p0,a,x):sum(p0,r0,a,x))&sum(p1,r1,b,y)&sum(p2,r2,d,z))!=15)return fallback();
                    ++g_ps2xVu1MatrixHits;

                    // These lower halves only load data or issue DIV. Preserve
                    // their issue cycles; none observes intermediate MAC/ACC.
                    Vu1NativeAccess::execLowerN<U0,L0>(v,c);++v.m_cycle;
                    Vu1NativeAccess::execLowerN<U1,L1>(v,c);++v.m_cycle;
                    Vu1NativeAccess::execLowerN<U2,L2>(v,c);
                    if constexpr(Acc)_mm_storeu_ps(v.m_state.acc,r2);
                    else {_mm_storeu_ps(v.m_state.acc,r1);_mm_storeu_ps(v.m_state.vf[(U2>>6)&31],r2);}
                    v.m_currentUpperInstruction=U2;v.m_viBranchBackupValid=false;
                    v.m_state.vf[0][0]=v.m_state.vf[0][1]=v.m_state.vf[0][2]=0;v.m_state.vf[0][3]=1;v.m_state.vi[0]=0;
                    c.instrCount+=3;Vu1NativeAccess::advanceOneCycleFast(v);
                    return Vu1NativeStep::Continue;
                };
                g_ps2xVu1GameMatrix[0]=+[](VU1Interpreter &v,Vu1NativeCtx &c){return decltype(triple){}.operator()<locallyGeneratedVu::upper<0x0e58u>, locallyGeneratedVu::lower<0x1ba8u>,locallyGeneratedVu::upper<0x0e60u>, locallyGeneratedVu::lower<0x1bb0u>,locallyGeneratedVu::upper<0x0e68u>, locallyGeneratedVu::lower<0x1bb8u>,false>(v,c);};
                g_ps2xVu1GameMatrix[1]=+[](VU1Interpreter &v,Vu1NativeCtx &c){return decltype(triple){}.operator()<locallyGeneratedVu::upper<0x0e70u>, locallyGeneratedVu::lower<0x1bc0u>,locallyGeneratedVu::upper<0x0e78u>, locallyGeneratedVu::lower<0x1bc8u>,locallyGeneratedVu::upper<0x0e80u>, locallyGeneratedVu::lower<0x1bd0u>,true>(v,c);};
                g_ps2xVu1GameMatrix[2]=+[](VU1Interpreter &v,Vu1NativeCtx &c){return decltype(triple){}.operator()<locallyGeneratedVu::upper<0x0e98u>, locallyGeneratedVu::lower<0x1be8u>,locallyGeneratedVu::upper<0x0ea0u>, locallyGeneratedVu::lower<0x0008u>,locallyGeneratedVu::upper<0x0ea8u>, locallyGeneratedVu::lower<0x0008u>,true>(v,c);};
            }
            if(!g_ps2xVu1ConvertPair[0]) {
                const auto convert=[]<uint32_t U,uint32_t L,unsigned Scale,bool ToFloat=false>(VU1Interpreter &v,Vu1NativeCtx &ctx)->Vu1NativeStep {
                    constexpr auto d=Vu1NativeAccess::PairDecode<U,L>::d;
                    static_assert(!d.iBit && !d.upperVfShadowReg);
                    static_assert(((U>>21)&15)==15 && ((U>>16)&31)!=0);
                    if(v.m_cycle>=v.m_pipelineNextReady)v.commitReadyPipelines();
                    v.m_currentUpperInstruction=U;
                    constexpr uint32_t viBits=uint32_t(d.lowerUsage.viWrite)&0xfffeu;
                    constexpr unsigned writtenVi=viBits?std::countr_zero(viBits):0;
                    const int32_t oldVi=writtenVi?v.m_state.vi[writtenVi]:0;
                    __m128i result;
                    if constexpr(ToFloat)result=_mm_castps_si128(_mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(v.m_state.vf[(U>>11)&31]))),_mm_set1_ps(1.0f/Scale)));
                    else result=vu1ConvertVector<Scale>(v.m_state.vf[(U>>11)&31]);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(v.m_state.vf[(U>>16)&31]),result);
                    Vu1NativeAccess::execLowerN<U,L>(v,ctx);
                    v.m_viBranchBackupValid=false;++ctx.instrCount;
                    if constexpr(writtenVi && d.lowerUsage.delaysNextBranchRead)v.recordViWriteForBranch(writtenVi,oldVi);
                    v.m_state.vf[0][0]=v.m_state.vf[0][1]=v.m_state.vf[0][2]=0.0f;
                    v.m_state.vf[0][3]=1.0f;v.m_state.vi[0]=0;
                    Vu1NativeAccess::advanceOneCycleFast(v);
                    return Vu1NativeStep::Continue;
                };
                g_ps2xVu1ConvertPair[0]=+[](VU1Interpreter &v,Vu1NativeCtx &c){return decltype(convert){}.operator()<locallyGeneratedVu::upper<0x1978u>,locallyGeneratedVu::lower<0x0168u>,1>(v,c);};
                g_ps2xVu1ConvertPair[1]=+[](VU1Interpreter &v,Vu1NativeCtx &c){
                    // Two adjacent conversions; retain both lower halves and
                    // cycle boundaries, including the ILW branch-read backup.
                    auto r=decltype(convert){}.operator()<locallyGeneratedVu::upper<0x01c8u>,locallyGeneratedVu::lower<0x01d0u>,4096,true>(v,c);
                    if(r!=Vu1NativeStep::Continue)return r;
                    return decltype(convert){}.operator()<locallyGeneratedVu::upper<0x19b0u>,locallyGeneratedVu::lower<0x0008u>,16>(v,c);
                };
            }
            if(!g_ps2xVu1FusedTransform && (liveBlockEnabled || benchEnabled || t_vuBenchReplay)) {
                // Defined in member scope to access pipeline state without
                // changing a widely included header. Private runtime bridge.
                const auto fused=[]<uint32_t L0,uint32_t L1,uint32_t L2,uint32_t U3,uint32_t L3,unsigned Dest>(VU1Interpreter &v,Vu1NativeCtx &ctx)->Vu1NativeStep {
                    const auto fallback=[&]() {
                        ++g_ps2xVu1FusedFallbacks;
                        Vu1NativeStep r;
                        if((r=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x03e8u>,L0,true,true,true,false,true>(v,ctx))!=Vu1NativeStep::Continue)return r;
                        if((r=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x03f0u>,L1,true,true,true,false,true>(v,ctx))!=Vu1NativeStep::Continue)return r;
                        if((r=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x03f8u>,L2,true,true,true,false,true>(v,ctx))!=Vu1NativeStep::Continue)return r;
                        return Vu1NativeAccess::stepPair<U3,L3,true,true,true,false,true>(v,ctx);
                    };
                    if(v.m_pipelineNextReady!=~0ull || v.m_xgkick.active || v.m_flagOrderCount!=0 || !v.m_immediateWrites)return fallback();
                    const __m128 xyz=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[5]));
                    const __m128 x=_mm_shuffle_ps(xyz,xyz,0x00), y=_mm_shuffle_ps(xyz,xyz,0x55), z=_mm_shuffle_ps(xyz,xyz,0xaa);
                    const __m128 a=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[21]));
                    const __m128 b=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[22]));
                    const __m128 d=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[23]));
                    const __m128 e=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[24]));
                    const __m128 p0=_mm_mul_ps(a,x),p1=_mm_mul_ps(b,y),p2=_mm_mul_ps(d,z);
                    const __m128 r1=_mm_add_ps(p0,p1),r2=_mm_add_ps(r1,p2),r3=_mm_add_ps(r2,e);
                    const auto ordinaryProduct=[](__m128 product,__m128 left,__m128 right) {
                        return vu1n::vuOrdinaryLanes(product) | ((vu1n::vuZeroLanes(left)|vu1n::vuZeroLanes(right)) & vu1n::vuZeroLanes(product));
                    };
                    const auto ordinarySum=[](__m128 product,__m128 result,__m128 left,__m128 right) {
                        const uint32_t zeros=vu1n::vuZeroLanes(left)|vu1n::vuZeroLanes(right);
                        return (vu1n::vuOrdinaryLanes(product)|zeros) & (vu1n::vuOrdinaryLanes(result)|(zeros & vu1n::vuZeroLanes(result)));
                    };
                    if((ordinaryProduct(p0,a,x)&ordinarySum(p1,r1,b,y)&ordinarySum(p2,r2,d,z)&ordinarySum(e,r3,e,_mm_set1_ps(1.0f)))!=15u)return fallback();
                    ++g_ps2xVu1FusedHits;
                    // No flag or ACC consumers exist in these four LQ lower
                    // halves. Accumulate all sticky S/Z contributions, publish
                    // the final MAC/current flags, and retain the final ACC.
                    const uint32_t stickySign=(_mm_movemask_ps(p0)|_mm_movemask_ps(p1)|_mm_movemask_ps(r1)|_mm_movemask_ps(p2)|_mm_movemask_ps(r2)|_mm_movemask_ps(e))?2u:0u;
                    const uint32_t stickyZero=(vu1n::vuZeroLanes(p0)|vu1n::vuZeroLanes(p1)|vu1n::vuZeroLanes(r1)|vu1n::vuZeroLanes(p2)|vu1n::vuZeroLanes(r2)|vu1n::vuZeroLanes(e))?1u:0u;
                    uint8_t f[4];const uint32_t neg=_mm_movemask_ps(r3),zero=vu1n::vuZeroLanes(r3);
                    for(unsigned i=0;i<4;++i)f[i]=uint8_t(((neg>>i)&1u)*2u+((zero>>i)&1u));
                    // All source rows were captured before these loads replace
                    // vf21/22/23. Empty pipelines make their cycle position
                    // unobservable; the four elapsed cycles remain identical.
                    // LQD decrements VI8 and leaves branch-backup metadata even
                    // after subsequent non-branch instructions invalidate it.
                    const int32_t oldVi8=v.m_state.vi[8];
                    Vu1NativeAccess::execLowerN<locallyGeneratedVu::upper<0x03e8u>,L0>(v,ctx);
                    if constexpr(L0==locallyGeneratedVu::lower<0x0410u>)v.recordViWriteForBranch(8,oldVi8);
                    Vu1NativeAccess::execLowerN<locallyGeneratedVu::upper<0x03f0u>,L1>(v,ctx);
                    Vu1NativeAccess::execLowerN<locallyGeneratedVu::upper<0x03f8u>,L2>(v,ctx);
                    Vu1NativeAccess::execLowerN<U3,L3>(v,ctx);
                    v.m_cycle+=3;
                    Vu1NativeAccess::updateFmacFlagsC<15,true,false>(v,f,stickySign|stickyZero);
                    ++v.m_cycle;v.m_state.cycles=v.m_cycle;
                    v.m_currentUpperInstruction=U3;
                    _mm_storeu_ps(v.m_state.acc,r2);_mm_storeu_ps(v.m_state.vf[Dest],r3);
                    v.m_viBranchBackupValid=false;
                    v.m_state.vf[0][0]=v.m_state.vf[0][1]=v.m_state.vf[0][2]=0.0f;v.m_state.vf[0][3]=1.0f;v.m_state.vi[0]=0;
                    ctx.instrCount+=4;
                    return Vu1NativeStep::Continue;
                };
                g_ps2xVu1FusedTransform=+[](VU1Interpreter &v,Vu1NativeCtx &c) {
                    return decltype(fused){}.operator()<locallyGeneratedVu::lower<0x03e8u>,locallyGeneratedVu::lower<0x03f0u>,locallyGeneratedVu::lower<0x03f8u>,locallyGeneratedVu::upper<0x0400u>,locallyGeneratedVu::lower<0x0400u>,6>(v,c);
                };
                g_ps2xVu1FusedTransformSecond=+[](VU1Interpreter &v,Vu1NativeCtx &c) {
                    return decltype(fused){}.operator()<locallyGeneratedVu::lower<0x0410u>,locallyGeneratedVu::lower<0x0418u>,locallyGeneratedVu::lower<0x0420u>,locallyGeneratedVu::upper<0x0428u>,locallyGeneratedVu::lower<0x0428u>,7>(v,c);
                };
            }

            if (verifyBlock || t_vuBenchReplay || (benchEnabled && !t_vu1InShadow && !s_vu1DiffMode &&
                g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed)>=benchFirstFrame &&
                std::chrono::duration<double>(std::chrono::steady_clock::now()-benchEpoch).count()>=benchFirstSeconds &&
                (!captureOnlyShader || captureShader==captureOnlyShader) && benchTotal<64 && benchSamples[captureKey]<2))
            {
                ++benchSamples[captureKey]; ++benchTotal;
                const auto savedKick=g_ps2xVu1Kick;
                auto clone=std::make_unique<VU1Interpreter>(*this);
                auto expected=std::make_unique<VU1Interpreter>(*this);
                std::vector<uint8_t> input(vuData,vuData+dataSize), data(input);
                std::vector<uint8_t> gif, expectedGif, expectedData;
                gif.reserve(262144);
                double bestNs=std::numeric_limits<double>::max(), totalNs=0, baselineNs=0, prototypeNs=0;
                uint64_t pairs=0; bool matches=true, stateMatch=true, dataMatch=true, gifMatch=true; size_t objectDiff=0;
                Vu1NativeStep expectedStep=Vu1NativeStep::Fallback;
                if(t_vuBenchReplay && std::getenv("PS2X_VU1_MATRIX_TEST")) {
                    uint32_t random=0x2378129bu;unsigned failures=0;
                    const uint32_t special[]={0,0x80000000u,1,0x807fffffu,0x00800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,0x7fc00000u,0x3f800000u};
                    for(unsigned test=0;test<12288;++test) {
                        *clone=*this;data=input;gif.clear();
                        clone->m_activeVuData=data.data();clone->m_activeMemory=nullptr;clone->m_activeGs=nullptr;
                        clone->m_gifCapture=&gif;clone->m_gifCaptureOnly=true;
                        clone->m_pipelineNextReady=~0ull;clone->m_xgkick.active=false;clone->m_flagOrderCount=0;clone->m_fdiv.valid=false;
                        const auto bits=[&] {random^=random<<13;random^=random>>17;random^=random<<5;return test%3==0?special[random%std::size(special)]:(test%3==1?random:(random&0x807fffffu)|((120u+random%14)<<23));};
                        for(unsigned reg:{1u,2u,3u,11u,12u,15u,16u,17u,20u,21u,30u})for(unsigned lane=0;lane<4;++lane){auto b=bits();std::memcpy(&clone->m_state.vf[reg][lane],&b,4);}
                        for(auto &a:clone->m_state.acc){auto b=bits();std::memcpy(&a,&b,4);}
                        if(test%7==0) {clone->m_fdiv.valid=true;clone->m_fdiv.readyCycle=clone->m_cycle+1;clone->m_fdiv.value=1.5f;clone->m_fdiv.statusDi=0;clone->m_pipelineNextReady=clone->m_cycle+1;}
                        auto seed=std::make_unique<VU1Interpreter>(*clone);
                        Vu1NativeCtx a{data.data(),dataSize,codeSize,&gs,nullptr,budgetEnd,0};
                        Vu1NativeStep want=Vu1NativeStep::Continue;
                        switch((test/3)%4) {
                        case 0:
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0e58u>, locallyGeneratedVu::lower<0x1ba8u>, true, true, true, true, true>(*clone,a);
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0e60u>, locallyGeneratedVu::lower<0x1bb0u>, true, true, true, true, true>(*clone,a);
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0e68u>, locallyGeneratedVu::lower<0x1bb8u>, true, true, true, true, true>(*clone,a);
                            break;
                        case 1:
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0e70u>, locallyGeneratedVu::lower<0x1bc0u>, true, true, true, true, true>(*clone,a);
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0e78u>, locallyGeneratedVu::lower<0x1bc8u>, true, true, true, true, true>(*clone,a);
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0e80u>, locallyGeneratedVu::lower<0x1bd0u>, true, true, true, true, true>(*clone,a);
                            break;
                        case 2:
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0e98u>, locallyGeneratedVu::lower<0x1be8u>, true, true, true, true, true>(*clone,a);
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0ea0u>, locallyGeneratedVu::lower<0x0008u>, true, true, true, true, true>(*clone,a);
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0ea8u>, locallyGeneratedVu::lower<0x0008u>, true, true, true, true, true>(*clone,a);
                            break;
                        case 3:
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0060u>,locallyGeneratedVu::lower<0x1a60u>,true,true,true,true,true>(*clone,a);
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0068u>,locallyGeneratedVu::lower<0x00c0u>,true,true,true,true,true>(*clone,a);
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x00d0u>,locallyGeneratedVu::lower<0x0130u>,true,true,true,true,true>(*clone,a);
                            break;
                        }
                        *expected=*clone;expectedData=data;expectedGif=gif;
                        *clone=*seed;data=input;gif.clear();
                        Vu1NativeCtx b{data.data(),dataSize,codeSize,&gs,nullptr,budgetEnd,0};
                        auto got=((test/3)%4==3)?g_ps2xVu1ShaderTransform(*clone,b):g_ps2xVu1GameMatrix[(test/3)%4](*clone,b);
                        for(auto *obj:{clone.get(),expected.get()}) {
                            constexpr size_t used=offsetof(ScalarPipelineEntry,valid)+sizeof(bool);
                            std::memset(reinterpret_cast<uint8_t*>(&obj->m_fdiv)+used,0,sizeof(ScalarPipelineEntry)-used);
                            for(auto &e:obj->m_efu)std::memset(reinterpret_cast<uint8_t*>(&e)+used,0,sizeof(ScalarPipelineEntry)-used);
                        }
                        if(got!=want || a.instrCount!=b.instrCount || data!=expectedData || gif!=expectedGif || std::memcmp(clone.get(),expected.get(),sizeof(*clone))) {
                            if(failures<5) {
                                std::fprintf(stderr,"[vu1:matrix-test] mismatch test=%u\n",test);
                                const auto *x=reinterpret_cast<const uint8_t*>(clone.get()),*y=reinterpret_cast<const uint8_t*>(expected.get());
                                unsigned shown=0;for(size_t k=0;k<sizeof(*clone)&&shown<8;++k)if(x[k]!=y[k]){std::fprintf(stderr," offset=%zu got=%02x want=%02x\n",k,x[k],y[k]);++shown;}
                            }
                            ++failures;
                        }
                    }
                    std::fprintf(stderr,"[vu1:matrix-test] cases=12288 failures=%u fast=%llu fallback=%llu\n",failures,g_ps2xVu1MatrixHits,g_ps2xVu1MatrixFallbacks);
                    if(failures){g_ps2xVu1Kick=savedKick;if(useVuRounding && previousRoundingMode!=-1)std::fesetround(previousRoundingMode);return;}
                }
                if(t_vuBenchReplay && std::getenv("PS2X_VU1_CONVERT_TEST")) {
                    uint32_t random=0x49c8261du;unsigned failures=0;
                    const uint32_t special[]={0,0x80000000u,1,0x807fffffu,0x00800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,0x7fc00000u,0xffc00000u,0x4effffffu,0x4f000000u,0xcf000000u,0x4d7fffffu,0x4d800000u,0xcd800000u};
                    for(unsigned test=0;test<262144;++test) {
                        float source[4];int32_t actual[4];
                        for(unsigned lane=0;lane<4;++lane) {
                            random^=random<<13;random^=random>>17;random^=random<<5;
                            uint32_t bits=test<1024?special[random%std::size(special)]:random;
                            std::memcpy(&source[lane],&bits,4);
                        }
                        for(unsigned scale:{1u,16u}) {
                            const __m128i result=scale==1?vu1ConvertVector<1>(source):vu1ConvertVector<16>(source);
                            _mm_storeu_si128(reinterpret_cast<__m128i*>(actual),result);
                            for(unsigned lane=0;lane<4;++lane)if(actual[lane]!=Vu1NativeAccess::floatToInt(Vu1NativeAccess::normOp(source[lane]),float(scale)))++failures;
                        }
                    }
                    // Compare complete pair state too, including a Q result
                    // becoming ready between the adjacent conversions.
                    for(unsigned test=0;test<8192;++test) {
                        *clone=*this;data=input;gif.clear();
                        clone->m_activeVuData=data.data();clone->m_activeMemory=nullptr;clone->m_activeGs=nullptr;
                        clone->m_gifCapture=&gif;clone->m_gifCaptureOnly=true;
                        for(unsigned reg:{9u,12u,24u})for(unsigned lane=0;lane<4;++lane) {
                            random^=random<<13;random^=random>>17;random^=random<<5;
                            const uint32_t bits=test<1024?special[random%std::size(special)]:random;
                            std::memcpy(&clone->m_state.vf[reg][lane],&bits,4);
                        }
                        if(test%7==0) {
                            clone->m_fdiv.valid=true;clone->m_fdiv.readyCycle=clone->m_cycle+1;
                            clone->m_fdiv.value=1.5f;clone->m_fdiv.statusDi=0;
                            clone->m_pipelineNextReady=std::min(clone->m_pipelineNextReady,clone->m_cycle+1);
                        }
                        auto seed=std::make_unique<VU1Interpreter>(*clone);
                        Vu1NativeCtx a{data.data(),dataSize,codeSize,&gs,nullptr,budgetEnd,0};
                        Vu1NativeStep want;
                        if(test&1u) {
                            want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x01c8u>,locallyGeneratedVu::lower<0x01d0u>,true,true,true,false,true>(*clone,a);
                            if(want==Vu1NativeStep::Continue)want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x19b0u>,locallyGeneratedVu::lower<0x0008u>,true,true,true,false,true>(*clone,a);
                        } else want=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x1978u>,locallyGeneratedVu::lower<0x0168u>,false,true,true,false,true>(*clone,a);
                        *expected=*clone;expectedData=data;expectedGif=gif;
                        *clone=*seed;data=input;gif.clear();
                        Vu1NativeCtx b{data.data(),dataSize,codeSize,&gs,nullptr,budgetEnd,0};
                        auto got=g_ps2xVu1ConvertPair[test&1u](*clone,b);
                        for(auto *obj:{clone.get(),expected.get()}) {
                            constexpr size_t used=offsetof(ScalarPipelineEntry,valid)+sizeof(bool);
                            std::memset(reinterpret_cast<uint8_t*>(&obj->m_fdiv)+used,0,sizeof(ScalarPipelineEntry)-used);
                            for(auto &e:obj->m_efu)std::memset(reinterpret_cast<uint8_t*>(&e)+used,0,sizeof(ScalarPipelineEntry)-used);
                        }
                        if(got!=want || a.instrCount!=b.instrCount || data!=expectedData || gif!=expectedGif || std::memcmp(clone.get(),expected.get(),sizeof(*clone))) {
                            if(failures<5)std::fprintf(stderr,"[vu1:convert-test] pair mismatch test=%u\n",test);
                            ++failures;
                        }
                    }
                    std::fprintf(stderr,"[vu1:convert-test] lanes=2097152 pairs=8192 failures=%u\n",failures);
                    if(failures) {g_ps2xVu1Kick=savedKick;if(useVuRounding && previousRoundingMode!=-1)std::fesetround(previousRoundingMode);return;}
                }
                if(t_vuBenchReplay && std::getenv("PS2X_VU1_FUSED_TEST")) {
                    uint32_t random=0x52a194bdu;unsigned failures=0;
                    const uint32_t special[]={0,0x80000000u,1,0x80000001u,0x007fffffu,0x00800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,0x7fc00000u,0x3f800000u};
                    for(unsigned test=0;test<4096;++test) {
                        *clone=*this;data=input;
                        clone->m_activeVuData=data.data();clone->m_activeMemory=nullptr;clone->m_activeGs=nullptr;
                        clone->m_gifCapture=&gif;clone->m_gifCaptureOnly=true;
                        clone->m_pipelineNextReady=~0ull;clone->m_xgkick.active=false;clone->m_flagOrderCount=0;
                        clone->m_fdiv.valid=false;
                        for(unsigned reg:{5u,21u,22u,23u,24u})for(unsigned lane=0;lane<4;++lane) {
                            random^=random<<13;random^=random>>17;random^=random<<5;
                            uint32_t bits;
                            if(test%3==0)bits=special[random%12];
                            else if(test%3==1)bits=random;
                            else bits=(random&0x807fffffu)|((120u+(random%14u))<<23);
                            std::memcpy(&clone->m_state.vf[reg][lane],&bits,4);
                        }
                        // Exercise the outstanding-Q guard as well as ordinary
                        // inputs, exceptional magnitudes, signed zero and NaN.
                        if(test%7==0) {
                            clone->m_fdiv.valid=true;clone->m_fdiv.readyCycle=clone->m_cycle+1;
                            clone->m_fdiv.value=1.5f;clone->m_fdiv.statusDi=0;
                            clone->m_pipelineNextReady=clone->m_cycle+1;
                        }
                        auto seed=std::make_unique<VU1Interpreter>(*clone);
                        Vu1NativeCtx q{data.data(),dataSize,codeSize,&gs,nullptr,budgetEnd,0};
                        Vu1NativeStep r;
                        if(test&1u) {
                        r=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x03e8u>,locallyGeneratedVu::lower<0x0410u>,true,true,true,false,true>(*clone,q);
                        if(r==Vu1NativeStep::Continue)r=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x03f0u>,locallyGeneratedVu::lower<0x0418u>,true,true,true,false,true>(*clone,q);
                        if(r==Vu1NativeStep::Continue)r=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x03f8u>,locallyGeneratedVu::lower<0x0420u>,true,true,true,false,true>(*clone,q);
                        if(r==Vu1NativeStep::Continue)r=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0428u>,locallyGeneratedVu::lower<0x0428u>,true,true,true,false,true>(*clone,q);
                        } else {
                        r=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x03e8u>,locallyGeneratedVu::lower<0x03e8u>,true,true,true,false,true>(*clone,q);
                        if(r==Vu1NativeStep::Continue)r=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x03f0u>,locallyGeneratedVu::lower<0x03f0u>,true,true,true,false,true>(*clone,q);
                        if(r==Vu1NativeStep::Continue)r=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x03f8u>,locallyGeneratedVu::lower<0x03f8u>,true,true,true,false,true>(*clone,q);
                        if(r==Vu1NativeStep::Continue)r=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0400u>,locallyGeneratedVu::lower<0x0400u>,true,true,true,false,true>(*clone,q);
                        }
                        *expected=*clone;expectedData=data;*clone=*seed;data=input;
                        Vu1NativeCtx z{data.data(),dataSize,codeSize,&gs,nullptr,budgetEnd,0};
                        auto hitsBefore=g_ps2xVu1FusedHits;
                        auto got=(test&1u?g_ps2xVu1FusedTransformSecond:g_ps2xVu1FusedTransform)(*clone,z);
                        // The pending-Q case clears a scalar queue entry via
                        // aggregate assignment, leaving its tail padding free.
                        for(auto *obj:{clone.get(),expected.get()}) {
                            constexpr size_t used=offsetof(ScalarPipelineEntry,valid)+sizeof(bool);
                            std::memset(reinterpret_cast<uint8_t*>(&obj->m_fdiv)+used,0,sizeof(ScalarPipelineEntry)-used);
                            for(auto &e:obj->m_efu)std::memset(reinterpret_cast<uint8_t*>(&e)+used,0,sizeof(ScalarPipelineEntry)-used);
                        }
                        if(got!=r || q.instrCount!=z.instrCount || data!=expectedData || std::memcmp(clone.get(),expected.get(),sizeof(*clone))) {
                            if(failures<5) {
                                size_t at=0;auto *a=reinterpret_cast<const uint8_t*>(clone.get());auto *b=reinterpret_cast<const uint8_t*>(expected.get());
                                while(at<sizeof(*clone) && a[at]==b[at])++at;
                                std::fprintf(stderr,"[vu1:fused-diff] test=%u fast=%u offset=%zu stateOffset=%zu stateSize=%zu mac=%x/%x status=%x/%x data=%u\n",test,g_ps2xVu1FusedHits!=hitsBefore,at,offsetof(VU1Interpreter,m_state),sizeof(VU1State),clone->m_state.mac,expected->m_state.mac,clone->m_state.status,expected->m_state.status,data==expectedData);
                            }
                            ++failures;
                        }
                    }
                    std::fprintf(stderr,"[vu1:fused-test] cases=4096 failures=%u fast=%llu fallback=%llu\n",failures,static_cast<unsigned long long>(g_ps2xVu1FusedHits),static_cast<unsigned long long>(g_ps2xVu1FusedFallbacks));
                    if(failures) {
                        g_ps2xVu1Kick=savedKick;
                        if(useVuRounding && previousRoundingMode!=-1)std::fesetround(previousRoundingMode);
                        return;
                    }
                }
#include "ps2_vu1_object_tests.inc"
                // Replay-only longer measurements avoid reporting speedups
                // from a handful of microsecond samples. Keep live checks cheap.
                const unsigned measuredRepetitions = [&] {
                    const char *p = t_vuBenchReplay ? std::getenv("PS2X_VU1_BENCH_REPETITIONS") : nullptr;
                    return p ? (std::clamp<unsigned>(static_cast<unsigned>(std::strtoul(p,nullptr,10)),2u,65536u) & ~1u) : 16u;
                }();
                for(unsigned repetition=0;repetition<(verifyBlock?2u:measuredRepetitions+1u);++repetition)
                {
                    *clone=*this; data=input; gif.clear();
                    clone->m_activeVuData=data.data();
                    clone->m_activeMemory=nullptr;
                    clone->m_activeGs=nullptr;
                    clone->m_gifCapture=&gif;
                    clone->m_gifCaptureOnly=true;
                    Vu1NativeCtx testCtx{data.data(),dataSize,codeSize,&gs,nullptr,budgetEnd,0u};
                    const auto begin=std::chrono::steady_clock::now();
                    const bool heavyReplay=(t_vuBenchReplay || verifyHeavy) && heavyCandidate;
                    const bool prototype=(((t_vuBenchReplay || verifyBlock) && blockCandidate) || heavyReplay) && (repetition&1u);
                    // Differential replay compares the original transfer with
                    // the specialized transfer, including partial pipeline state.
                    static const bool verifyQword=std::getenv("PS2X_VU1_QWORD_VERIFY")!=nullptr;
                    t_qwordReference=verifyQword && !(repetition&1u);
                    t_directMeshReference = s_directMeshReplay && !(repetition & 1u);
                    t_tailReference = !(repetition & 1u);
                    t_nativeStoreReference=!(repetition&1u);
                    auto result=prototype?(heavyReplay?ps2xVu1HeavyBlock(*clone,testCtx,program):ps2xVu1BlockPrototype(*clone,testCtx)):program(*clone,testCtx);

                    // For short-budget semantic tests, resume both stopped
                    // states through the unchanged native program. This checks
                    // that omitted intermediate flags cannot affect later work.
                    if(t_vuBenchReplay && std::getenv("PS2X_VU1_RESUME_TEST") && result==Vu1NativeStep::Stop && !clone->m_stopRequested) {
                        testCtx.budgetEnd=clone->m_cycle+16777216u;
                        result=program(*clone,testCtx);
                    }
                    if(result==Vu1NativeStep::Ended)
                    {
                        clone->advanceTo(clone->m_maxReadyCycle);
                        clone->flushPipelines();
                    }
                    t_directMeshReference = false;
                    t_qwordReference = false;
                    t_nativeStoreReference = false;
                    t_tailReference = false;
                    const double ns=double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count());
                    // In the game-math experiment, only unused STATUS and
                    // inactive flag history may differ. Registers, active flags,
                    // scalar/store pipelines, cycles, memory and GIF still compare.
                    static const bool gameMathCompare=[] {const char *p=std::getenv("PS2X_VU1_GAME_MATH");return !p || p[0]!='0';}();
                    if(gameMathCompare && heavyCandidate) {
                        clone->m_state.status=0;
                        for(auto &entry:clone->m_flagPipeline)if(!entry.valid)std::memset(&entry,0,sizeof(entry));
                        if(!clone->m_flagOrderCount){clone->m_flagOrderHead=0;std::memset(clone->m_flagOrder.data(),0,sizeof(clone->m_flagOrder));}
                        if(clone->m_flagTailReady<=clone->m_cycle)clone->m_flagTailReady=0;
                    }
                    // FlagPipelineEntry has two tail-padding bytes. Queue
                    // assignments do not define them; they are not VU state.
                    for(auto &entry:clone->m_flagPipeline) {
                        constexpr size_t used=offsetof(FlagPipelineEntry,shadowApplied)+sizeof(bool);
                        std::memset(reinterpret_cast<uint8_t*>(&entry)+used,0,sizeof(entry)-used);
                    }
                    // Scalar/store/write queue entries also have tail
                    // padding; compare all fields, never uninitialised padding.
                    const auto clearTail=[](auto &entry,size_t used) {
                        std::memset(reinterpret_cast<uint8_t*>(&entry)+used,0,sizeof(entry)-used);
                    };
                    clearTail(clone->m_fdiv,offsetof(ScalarPipelineEntry,valid)+sizeof(bool));
                    for(auto &e:clone->m_efu)clearTail(e,offsetof(ScalarPipelineEntry,valid)+sizeof(bool));
                    for(auto &e:clone->m_storePipeline) {
                        // Retirement leaves old payload/address bytes in free
                        // slots; queueStore overwrites every field on reuse.
                        // Native kernels can bypass these slots entirely. Only
                        // a valid pending store is observable future VU state.
                        if (!e.valid) std::memset(&e,0,sizeof(e));
                        else clearTail(e,offsetof(PendingStore,valid)+sizeof(bool));
                    }
                    for(auto &e:clone->m_vfWritePipeline)clearTail(e,offsetof(PendingVfWrite,valid)+sizeof(bool));
                    for(auto &e:clone->m_viWritePipeline)clearTail(e,offsetof(PendingViWrite,valid)+sizeof(bool));
                    for(auto &e:clone->m_accWritePipeline)clearTail(e,offsetof(PendingAccWrite,valid)+sizeof(bool));
                    if(repetition==0)
                    {
                        *expected=*clone; expectedData=data; expectedGif=gif;
                        expectedStep=result; pairs=testCtx.instrCount;
                    }
                    else
                    {
                        bestNs=std::min(bestNs,ns); totalNs+=ns; if(prototype)prototypeNs+=ns;else baselineNs+=ns;
                        // Objects start as exact copies and use the same buffer
                        // pointers. Queue padding is normalized above.
                        stateMatch=stateMatch && std::memcmp(&clone->m_state,&expected->m_state,sizeof(VU1State))==0;
                        dataMatch=dataMatch && data==expectedData; gifMatch=gifMatch && gif==expectedGif;
                        const bool objectMatch=std::memcmp(clone.get(),expected.get(),sizeof(VU1Interpreter))==0;
                        if(!objectMatch && !objectDiff) {
                            const auto *a=reinterpret_cast<const uint8_t*>(clone.get());
                            const auto *b=reinterpret_cast<const uint8_t*>(expected.get());
                            while(objectDiff<sizeof(VU1Interpreter) && a[objectDiff]==b[objectDiff])++objectDiff;
                            if(std::getenv("PS2X_VU1_OBJECT_PROFILE")) {
                                const char *field="inactive metadata";
                                Vu1NativeAccess::hleSameState(*clone,*expected,&field);
                                std::fprintf(stderr,"[vu1:object-diff] field=%s offset=%zu tail=%zu upper=%zu shadow=%zu backup=%zu\n",field,objectDiff,
                                    offsetof(VU1Interpreter,m_flagTailReady),offsetof(VU1Interpreter,m_currentUpperInstruction),
                                    offsetof(VU1Interpreter,m_shadowMac),offsetof(VU1Interpreter,m_viBranchBackupValue));
                            }
                        }
                        matches=matches && result==expectedStep && pairs==testCtx.instrCount && objectMatch && dataMatch && gifMatch;
                    }
                }
                if (t_vuBenchReplay && s_directMeshReplay)
                    std::fprintf(stderr, "[vu1:direct-replay] nativeMeshes=%llu\n", (unsigned long long)t_directMeshPrepared);
                if(!t_vuBenchReplay && !verifyBlock) if(const char *dir=std::getenv("PS2X_VU1_BENCH_DIR");dir && *dir) {
                    *clone=*this;
                    // Same-build developer format: remove all host pointers.
                    clone->m_cachedVuCode=nullptr;clone->m_cachedMemory=nullptr;
                    clone->m_activeVuData=nullptr;clone->m_activeGs=nullptr;clone->m_activeMemory=nullptr;
                    clone->m_gifCapture=nullptr;
                    VuBenchHeader h;h.image=s_vu1ImageHash;h.codeSize=codeSize;h.dataSize=dataSize;h.maxCycles=maxCycles;
                    char name[96];std::snprintf(name,sizeof(name),"sample-%02u-%016llx-%04x.vu",benchTotal,static_cast<unsigned long long>(h.image),m_state.pc);
                    const auto path=std::filesystem::path(dir)/name;
                    if(!std::filesystem::exists(path)) if(FILE *out=std::fopen(path.string().c_str(),"wb")) {
                        bool ok=std::fwrite(&h,sizeof(h),1,out)==1 && std::fwrite(clone.get(),sizeof(*clone),1,out)==1 &&
                            std::fwrite(vuCode,1,codeSize,out)==codeSize && std::fwrite(input.data(),1,dataSize,out)==dataSize;
                        ok=std::fclose(out)==0 && ok;
                        std::fprintf(stderr,"[vu1:checkpoint] %s %s\n",ok?"saved":"FAILED",path.string().c_str());
                    }
                }
                if(t_vuBenchReplay) if(const char *path=std::getenv("PS2X_VU1_REPLAY_GIF")) {
                    if(auto *out=std::fopen(path,"wb")) {std::fwrite(expectedGif.data(),1,expectedGif.size(),out);std::fclose(out);}
                    const auto dataPath=std::string(path)+".dmem";
                    if(auto *out=std::fopen(dataPath.c_str(),"wb")) {std::fwrite(expectedData.data(),1,expectedData.size(),out);std::fclose(out);}
                }
                g_ps2xVu1Kick=savedKick;
                if(!verifyBlock) std::fprintf(stderr,"[vu1:bench] sample=%u image=%016llx pc=0x%x shader=0x%x pairs=%llu gif=%zu best=%.0fns mean=%.0fns ns/pair=%.2f repeatMatch=%u state=%u data=%u gifMatch=%u objectDiff=%zu storeOffset=%zu flagOffset=%zu\n",
                    benchTotal,static_cast<unsigned long long>(s_vu1ImageHash),m_state.pc,captureShader,
                    static_cast<unsigned long long>(pairs),expectedGif.size(),bestNs,totalNs/measuredRepetitions,
                    pairs?bestNs/pairs:0,matches?1u:0u,stateMatch?1u:0u,dataMatch?1u:0u,gifMatch?1u:0u,objectDiff,offsetof(VU1Interpreter,m_storePipeline),offsetof(VU1Interpreter,m_flagPipeline));
                if(prototypeNs>0 && !verifyBlock) std::fprintf(stderr,"[vu1:block] baseline=%.0fns prototype=%.0fns speedup=%.3f match=%u fused=%llu fallback=%llu\n",baselineNs/(measuredRepetitions/2),prototypeNs/(measuredRepetitions/2),baselineNs/prototypeNs,matches?1u:0u,static_cast<unsigned long long>(g_ps2xVu1FusedHits),static_cast<unsigned long long>(g_ps2xVu1FusedFallbacks));
                if(verifyBlock) {
                    if(!matches)liveBlockFailed=true;
                    if(verifyHeavy) {
                        static uint64_t heavyVerified=0;++heavyVerified;
                        if(heavyVerified==1 || heavyVerified%1000==0 || !matches)
                            std::fprintf(stderr,"[vu1:heavy-verify] calls=%llu pc=0x%x match=%u state=%u data=%u gif=%u disabled=%u\n",static_cast<unsigned long long>(heavyVerified),m_state.pc,matches,stateMatch,dataMatch,gifMatch,liveBlockFailed);
                    }
                    if(benchTotal<=3 || benchTotal%1000==0 || !matches)
                        std::fprintf(stderr,"[vu1:block-verify] calls=%u match=%u state=%u data=%u gif=%u disabled=%u\n",benchTotal,matches,stateMatch,dataMatch,gifMatch,liveBlockFailed);
                }
                if(t_vuBenchReplay) {
                    t_vuBenchPassed=matches;
                    if(useVuRounding && previousRoundingMode!=-1)std::fesetround(previousRoundingMode);
                    return;
                }
            }
            // Reuse the pre-execution selector for both indirect entry routes.
            const uint32_t profileSelector=heavyCandidate ? captureShader : 0u;
            Vu1NativeCtx nctx{vuData, dataSize, codeSize, &gs, memory, budgetEnd, 0u};
            const bool useHeavy=liveBlockEnabled && liveHeavyEnabled && heavyCandidate && !liveBlockFailed;
            const bool useBlock=liveBlockEnabled && blockCandidate && !liveBlockFailed;
            Vu1NativeStep step=useHeavy?ps2xVu1HeavyBlock(*this,nctx,program):(useBlock?ps2xVu1BlockPrototype(*this,nctx):program(*this,nctx));
            if(useHeavy) {
                static uint64_t heavyCalls=0;++heavyCalls;
                if(heavyCalls==1 || heavyCalls%100000==0)
                    std::fprintf(stderr,"[vu1:heavy-live] calls=%llu\n",static_cast<unsigned long long>(heavyCalls));
            }
            if(useBlock && step==Vu1NativeStep::Fallback)step=program(*this,nctx);
            if(useBlock) {
                static uint64_t liveCalls=0;++liveCalls;
                if(liveCalls==1 || liveCalls%5000==0)
                    std::fprintf(stderr,"[vu1:block-live] calls=%llu fused=%llu fallback=%llu\n",static_cast<unsigned long long>(liveCalls),static_cast<unsigned long long>(g_ps2xVu1FusedHits),static_cast<unsigned long long>(g_ps2xVu1FusedFallbacks));
            }
            static const bool entryProfileEnabled=std::getenv("PS2X_VU1_ENTRY_PROFILE")!=nullptr;
            static const uint64_t profileStart=[] {const char *p=std::getenv("PS2X_VU1_PROFILE_START_FRAME");return p?std::strtoull(p,nullptr,10):1600ull;}();
            static const double profileSeconds=[] {const char *p=std::getenv("PS2X_VU1_PROFILE_START_SECONDS");return p?std::strtod(p,nullptr):0.0;}();
            if(entryProfileEnabled && !t_vuBenchReplay &&
               g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed)>=profileStart &&
               std::chrono::duration<double>(std::chrono::steady_clock::now()-benchEpoch).count()>=profileSeconds) {
                static VuEntryProfile profile;
                auto &row=profile.counts[{benchKey.first,benchKey.second|(profileSelector<<16)}];++row.calls;row.pairs+=nctx.instrCount;
                profile.pairs+=nctx.instrCount;
                if(profile.pairs>=profile.nextDump) { profile.dump();profile.nextDump+=250000000; }
            }
            instrCount += nctx.instrCount;
            const uint64_t nr = g_ps2xVu1NativeRuns.fetch_add(1u, std::memory_order_relaxed) + 1u;
            if (nr == 1u)
                std::fprintf(stderr, "[vu1:native] engaged for image %016llx\n",
                             static_cast<unsigned long long>(s_vu1ImageHash));
            if (step == Vu1NativeStep::Ended)
            {
                programEnded = true;
                nativeHandled = true;
            }
            else if (step == Vu1NativeStep::Stop)
            {
                nativeHandled = true;
            }
            else
            {
                const uint64_t nf = g_ps2xVu1NativeFallbacks.fetch_add(1u, std::memory_order_relaxed) + 1u;
                if (nf <= 5u)
                    std::fprintf(stderr, "[vu1:native] fallback to interpreter at pc=0x%x (#%llu)\n",
                                 m_state.pc, static_cast<unsigned long long>(nf));
            }
        }
    }
    if (!nativeHandled && m_unit == Unit::VU1)
        g_ps2xVu1InterpRuns.fetch_add(1u, std::memory_order_relaxed);
    // Relaxed VU0 (opt-in, PS2X_VU0_RELAX=1 -- BROKEN: menus invisible, crash at pre-level cutscene).
    // Same rule as the relaxed VU1 native programs: no hazard stalls, and after
    // every pair the Q/P/store/XGKICK pipelines and queued flag updates complete
    // at once. PCSX2's VU recompilers behave this way and WotM renders
    // correctly there; our cycle-exact model stretched deep skeleton bones.
    static const bool s_vu0Relaxed = []
    {
        const char *value = std::getenv("PS2X_VU0_RELAX");
        return value != nullptr && value[0] == '1'; // opt-in: default-on hid the menus and crashed at the pre-level cutscene
    }();
    static const bool s_vu1RelaxedRef = []
    {
        const char *value = std::getenv("PS2X_VU1_RELAX");
        return value != nullptr && value[0] == '1';
    }();
    const bool relaxedHere = (s_vu0Relaxed && m_unit == Unit::VU0) || (s_vu1RelaxedRef && m_unit == Unit::VU1);
    const auto relaxSettle = [this]()
    {
        while (m_pipelineNextReady != ~0ull || m_xgkick.active)
            advanceOneCycle();
        while (m_flagOrderCount != 0u)
        {
            const uint32_t best = m_flagOrder[m_flagOrderHead];
            FlagPipelineEntry &entry = m_flagPipeline[static_cast<size_t>(best)];
            m_flagOrderHead = (m_flagOrderHead + 1u) % kMaxFlagEntries;
            --m_flagOrderCount;
            if (entry.writesMac)
                m_state.mac = entry.mac;
            if (entry.writesStatus)
            {
                const uint32_t current = entry.status & 0xFu;
                m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
            }
            if (entry.writesSticky)
                m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
            if (entry.writesClip)
                m_state.clip = entry.clip;
            entry.valid = false;
            m_flagMask &= ~(1u << best);
        }
    };
    while (!nativeHandled && m_cycle < budgetEnd && !m_stopRequested)
    {
        VP_BEGIN();
        commitReadyPipelines();
        if (m_state.pc + 8u > codeSize)
            break;

        const uint32_t pairIndex = m_state.pc / 8u;
        const DecodedInstructionPair &decoded =
            (directDecode && (m_state.pc & 7u) == 0u && pairIndex < kMaxDecodedPairs)
                ? m_decodedCodeCache[pairIndex]
                : (decodedFallback = getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc));
        VP_MARK(kVpDecode);
        if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved)
        {
            reportReservedInstruction(decoded.upperUsage.reserved, decoded.upperUsage.reserved ? decoded.upper : decoded.lower);
            break;
        }

        uint64_t readyCycle = relaxedHere ? m_cycle : calculatePairReadyCycle(decoded);
        // Relaxed VU1 intentionally skips register/FMAC latency stalls, but
        // XGKICK is a single PATH1 transfer resource, not an arithmetic result.
        // A second XGKICK must wait for the first packet to drain.  Skipping
        // this check reached startXgkick with an active transfer, where the
        // emergency fallback copied the remaining packet instantaneously and
        // changed the timing of VU memory reuse (visible as intermittent bad
        // triangles in effects and close-up geometry).
        if (relaxedHere && decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
            readyCycle = m_cycle + 1u;
        while (readyCycle > m_cycle)
        {
            if (readyCycle >= budgetEnd)
            {
                advanceTo(budgetEnd);
                break;
            }
            advanceTo(readyCycle);
            readyCycle = calculatePairReadyCycle(decoded);
        }
        if (m_cycle >= budgetEnd)
            break;
        VP_MARK(kVpReady);

        if (s_vu1DiffMode)
        {
            uint64_t efuReady = 0u;
            for (const ScalarPipelineEntry &e : m_efu)
                if (e.valid)
                    efuReady = std::max(efuReady, e.readyCycle);
            retireFlags(); // the values a flag read would see now
            (t_vu1InShadow ? g_traceShadow : g_traceAuth)
                .add(m_state.pc, m_cycle, m_maxReadyCycle, m_fdiv.valid ? m_fdiv.readyCycle : 0u, efuReady,
                     m_state.q, m_state.mac, m_state.clip, m_state.status);
        }

        uint8_t writtenVi = 0u;
        int32_t oldVi = 0;
        if (const uint32_t viBits = static_cast<uint32_t>(decoded.lowerUsage.viWrite) & 0xFFFEu; viBits != 0u)
        {
            const uint32_t reg = static_cast<uint32_t>(std::countr_zero(viBits));
            writtenVi = static_cast<uint8_t>(reg);
            oldVi = m_state.vi[reg];
        }

        const VfAccess upperWrite = decoded.upperUsage.vfWrite;
        const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
        const bool hasUpperWrite = !m_immediateWrites && upperWrite.reg != 0u;
        const bool hasLowerWrite = !m_immediateWrites && lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg;
        const bool hasDistinctLowerWrite = hasLowerWrite && (!hasUpperWrite || lowerWrite.reg != upperWrite.reg);
        float oldUpperVf[4]{};
        float newUpperVf[4]{};
        float oldLowerVf[4]{};
        float newLowerVf[4]{};
        float oldAcc[4]{};
        float newAcc[4]{};
        if (hasUpperWrite)
            std::memcpy(oldUpperVf, m_state.vf[upperWrite.reg], sizeof(oldUpperVf));
        if (hasDistinctLowerWrite)
            std::memcpy(oldLowerVf, m_state.vf[lowerWrite.reg], sizeof(oldLowerVf));
        if (!m_immediateWrites && decoded.upperUsage.accWrite != 0u)
            std::memcpy(oldAcc, m_state.acc, sizeof(oldAcc));
        VP_MARK(kVpShadowIn);

        if (decoded.iBit)
        {
            execUpper(decoded.upper);
            float immediate = 0.0f;
            std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
            m_state.i = normalizeOperand(immediate);
        }
        else if (decoded.upperVfShadowReg != 0u)
        {
            float oldVf[4]{};
            float upperVf[4]{};
            std::memcpy(oldVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(oldVf));
            execUpper(decoded.upper);
            std::memcpy(upperVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(upperVf));
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        oldVf,
                        sizeof(oldVf));
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        upperVf,
                        sizeof(upperVf));
        }
        else
        {
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
            if (g_vpOn)
            {
                const uint64_t t0 = __rdtsc();
                execUpper(decoded.upper);
                const uint64_t t1 = __rdtsc();
                execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
                const uint64_t t2 = __rdtsc();
                const uint32_t uk = vpUpperKey(decoded.upper), lk = vpLowerKey(decoded.lower);
                g_vpUpTicks[uk] += t1 - t0;
                ++g_vpUpCount[uk];
                g_vpLoTicks[lk] += t2 - t1;
                ++g_vpLoCount[lk];
            }
            else
#endif
            {
                execUpper(decoded.upper);
                execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
            }
        }

        VP_MARK(kVpExec);
        m_viBranchBackupValid = false;
        if (s_vu1Flight && m_unit == Unit::VU1)
        {
            Vu1FlightEntry &e = s_vu1FlightRing[s_vu1FlightPos++ % kVu1FlightSize];
            e.pc = m_state.pc;
            e.upper = decoded.upper;
            e.lower = decoded.lower;
            e.cycle = m_cycle;
            e.q = m_state.q;
            e.i = m_state.i;
            retireFlags();
            e.clip = m_state.clip;
            e.mac = m_state.mac;
            e.status = m_state.status;
            e.uReg = decoded.upperUsage.vfWrite.reg;
            if (e.uReg)
                std::memcpy(e.uVal, m_state.vf[e.uReg], sizeof(e.uVal));
            e.accW = decoded.upperUsage.accWrite;
            std::memcpy(e.acc, m_state.acc, sizeof(e.acc));
            e.lReg = decoded.iBit ? 0u : decoded.lowerUsage.vfWrite.reg;
            if (e.lReg)
                std::memcpy(e.lVal, m_state.vf[e.lReg], sizeof(e.lVal));
            e.viReg = writtenVi;
            e.viVal = writtenVi ? m_state.vi[writtenVi] : 0;
        }

        if (hasUpperWrite)
        {
            std::memcpy(newUpperVf, m_state.vf[upperWrite.reg], sizeof(newUpperVf));
            std::memcpy(m_state.vf[upperWrite.reg], oldUpperVf, sizeof(oldUpperVf));
            const uint32_t latency =
                decoded.upperUsage.vfLatency != 0u
                    ? decoded.upperUsage.vfLatency
                    : decoded.upperUsage.latency;
            queueVfWrite(upperWrite.reg, upperWrite.lanes, newUpperVf, latency);
        }
        if (hasDistinctLowerWrite)
        {
            std::memcpy(newLowerVf, m_state.vf[lowerWrite.reg], sizeof(newLowerVf));
            std::memcpy(m_state.vf[lowerWrite.reg], oldLowerVf, sizeof(oldLowerVf));
            const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                         ? decoded.lowerUsage.vfLatency
                                         : decoded.lowerUsage.latency;
            queueVfWrite(lowerWrite.reg, lowerWrite.lanes, newLowerVf, latency);
        }
        if (!m_immediateWrites && decoded.upperUsage.accWrite != 0u)
        {
            std::memcpy(newAcc, m_state.acc, sizeof(newAcc));
            std::memcpy(m_state.acc, oldAcc, sizeof(oldAcc));
            // ACC is forwarded to the next upper instruction. Its arithmetic
            // flags still use the normal four-cycle FMAC timeline.
            queueAccWrite(decoded.upperUsage.accWrite, newAcc,
                          kAccForwardLatency);
        }
        if (!m_immediateWrites && writtenVi != 0u)
        {
            const int32_t newVi = m_state.vi[writtenVi];
            m_state.vi[writtenVi] = oldVi;
            const uint32_t latency =
                decoded.lowerUsage.viLatency != 0u
                    ? decoded.lowerUsage.viLatency
                    : decoded.lowerUsage.latency;
            queueViWrite(writtenVi, newVi, latency);
        }

        ++instrCount;
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
        if (vu1scope::enabled && m_unit == Unit::VU1)
            vu1scope::onInstr(decoded.upper, decoded.lower, m_state.pc);
#endif
        VP_MARK(kVpShadowOut);
        if (relaxedHere)
            relaxSettle();
        else
            markPairWrites(decoded);
        VP_MARK(kVpMark);
        if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead)
            recordViWriteForBranch(writtenVi, oldVi);

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
        extern std::atomic<bool> g_vu0InstrTrace;
        extern std::atomic<uint32_t> g_vu0InstrTraceStartPc;
        if (codeSize < 8192u && g_vu0InstrTrace.load(std::memory_order_relaxed) &&
            g_vu0InstrTraceStartPc.load(std::memory_order_relaxed) == 0x1A8u)
        {
            static std::atomic<uint32_t> s_vu0i{0u};
            if (s_vu0i.fetch_add(1u, std::memory_order_relaxed) < 300u)
            {
                std::fprintf(stderr,
                    "[vu0:i] pc=0x%x U=%08x L=%08x iBit=%d uW=r%d(l%x) lW=r%d(l%x) accW=%x "
                    "| vf9=(%.4g,%.4g,%.4g,%.4g) vf10=(%.4g,%.4g,%.4g,%.4g) vf11=(%.4g,%.4g,%.4g,%.4g) "
                    "vf31=(%.4g,%.4g,%.4g,%.4g) ACC=(%.4g,%.4g,%.4g,%.4g) Q=%.4g I=%.4g\n",
                    m_state.pc, decoded.upper, decoded.lower, (int)decoded.iBit,
                    (int)decoded.upperUsage.vfWrite.reg, (unsigned)decoded.upperUsage.vfWrite.lanes,
                    (int)decoded.lowerUsage.vfWrite.reg, (unsigned)decoded.lowerUsage.vfWrite.lanes,
                    (unsigned)decoded.upperUsage.accWrite,
                    m_state.vf[9][0], m_state.vf[9][1], m_state.vf[9][2], m_state.vf[9][3],
                    m_state.vf[10][0], m_state.vf[10][1], m_state.vf[10][2], m_state.vf[10][3],
                    m_state.vf[11][0], m_state.vf[11][1], m_state.vf[11][2], m_state.vf[11][3],
                    m_state.vf[31][0], m_state.vf[31][1], m_state.vf[31][2], m_state.vf[31][3],
                    m_state.acc[0], m_state.acc[1], m_state.acc[2], m_state.acc[3],
                    m_state.q, m_state.i);
            }
        }
#endif

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
        // [vu1:i] per-instruction trace of the settled menu vertex program(s).
        // Armed once the first settled (screenMain) VU1 run happens; spans that
        // execute() plus the following resume() continuations until the budget
        // drains. Cap keeps it bounded. .cpp-only diag (s13#12).
        {
            static std::atomic<int> s_vu1iBudget{-1};
            // Arm on the PERSPECTIVE-DIVIDE transform loop (progs 0x6c8 / 0x7c8 /
            // 0x19e8), not the 0x1868 MVP-setup prog — that's where the DIV / Q /
            // FTOI screen-coord pack lives (s13#17).
            if (s_vu1iBudget.load(std::memory_order_relaxed) < 0 && codeSize >= 8192u && memory &&
                ((m_state.pc >= 0x6c0u && m_state.pc < 0x820u) ||
                 (m_state.pc >= 0x19e0u && m_state.pc < 0x1c00u)) &&
                memory->read32(0x6F7E8Cu) == 0u && memory->read32(0x6F8464u) == 1u)
                s_vu1iBudget.store(2200, std::memory_order_relaxed);
            int rem = s_vu1iBudget.load(std::memory_order_relaxed);
            if (rem > 0 && codeSize >= 8192u)
            {
                s_vu1iBudget.store(rem - 1, std::memory_order_relaxed);
                std::fprintf(stderr,
                    "[vu1:i] pc=0x%x U=%08x L=%08x uW=r%d(%x) lW=r%d(%x) accW=%x"
                    " vi=[%d %d %d %d %d %d %d %d]"
                    " vf7=(%.5g,%.5g,%.5g,%.5g) vf8=(%.5g,%.5g,%.5g,%.5g)"
                    " vf9=(%.5g,%.5g,%.5g,%.5g) vf10=(%.5g,%.5g,%.5g,%.5g)"
                    " vf12=(%.5g,%.5g,%.5g,%.5g) vf17=(%.5g,%.5g,%.5g,%.5g)"
                    " vf19=(%.5g,%.5g,%.5g,%.5g) vf22=(%.5g,%.5g,%.5g,%.5g)"
                    " vf30=(%.5g,%.5g,%.5g,%.5g) vf31=(%.5g,%.5g,%.5g,%.5g)"
                    " I=%.6g Q=%.6g ACC=(%.6g,%.6g,%.6g,%.6g)\n",
                    m_state.pc, decoded.upper, decoded.lower,
                    (int)decoded.upperUsage.vfWrite.reg, (unsigned)decoded.upperUsage.vfWrite.lanes,
                    (int)decoded.lowerUsage.vfWrite.reg, (unsigned)decoded.lowerUsage.vfWrite.lanes,
                    (unsigned)decoded.upperUsage.accWrite,
                    m_state.vi[1], m_state.vi[2], m_state.vi[3], m_state.vi[4],
                    m_state.vi[5], m_state.vi[6], m_state.vi[7], m_state.vi[8],
                    m_state.vf[7][0], m_state.vf[7][1], m_state.vf[7][2], m_state.vf[7][3],
                    m_state.vf[8][0], m_state.vf[8][1], m_state.vf[8][2], m_state.vf[8][3],
                    m_state.vf[9][0], m_state.vf[9][1], m_state.vf[9][2], m_state.vf[9][3],
                    m_state.vf[10][0], m_state.vf[10][1], m_state.vf[10][2], m_state.vf[10][3],
                    m_state.vf[12][0], m_state.vf[12][1], m_state.vf[12][2], m_state.vf[12][3],
                    m_state.vf[17][0], m_state.vf[17][1], m_state.vf[17][2], m_state.vf[17][3],
                    m_state.vf[19][0], m_state.vf[19][1], m_state.vf[19][2], m_state.vf[19][3],
                    m_state.vf[22][0], m_state.vf[22][1], m_state.vf[22][2], m_state.vf[22][3],
                    m_state.vf[30][0], m_state.vf[30][1], m_state.vf[30][2], m_state.vf[30][3],
                    m_state.vf[31][0], m_state.vf[31][1], m_state.vf[31][2], m_state.vf[31][3],
                    m_state.i, m_state.q, m_state.acc[0], m_state.acc[1], m_state.acc[2], m_state.acc[3]);
            }
        }
#endif

        m_state.vf[0][0] = 0.0f;
        m_state.vf[0][1] = 0.0f;
        m_state.vf[0][2] = 0.0f;
        m_state.vf[0][3] = 1.0f;
        m_state.vi[0] = 0;

        uint32_t nextPc = m_state.pc + 8u;
        if (nextPc >= codeSize)
            nextPc = 0u;
        m_state.pc = nextPc;

        if (m_state.branchPending)
        {
            if (m_state.branchDelay == 0u)
            {
                m_state.pc = m_state.branchTarget & microAddressMask();
                m_state.branchPending = false;
            }
            else
            {
                --m_state.branchDelay;
            }
        }

        const bool dHalt = decoded.dBit && m_state.dBitEnabled;
        const bool tHalt = decoded.tBit && m_state.tBitEnabled;
        const bool haltBit = dHalt || tHalt;
        const bool haltBranch = haltBit && decoded.lowerUsage.pipeline == PipelineBranch;

        if (m_state.haltAfterDelaySlot)
        {
            m_state.stoppedByD = m_pendingHaltD;
            m_state.stoppedByT = m_pendingHaltT;
            programEnded = true;
        }
        else if (m_state.ebit)
            programEnded = true;
        else if (haltBit && !haltBranch)
        {
            m_state.stoppedByD = dHalt;
            m_state.stoppedByT = tHalt;
            programEnded = true;
        }
        else if (decoded.eBit)
            m_state.ebit = true;
        else if (haltBranch)
        {
            m_state.haltAfterDelaySlot = true;
            m_pendingHaltD = dHalt;
            m_pendingHaltT = tHalt;
        }

        VP_MARK(kVpTail);
        advanceOneCycle();
        VP_MARK(kVpAdvance);
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
        if (g_vpOn)
            vpMaybeDump();
#endif
        if (programEnded)
            break;
    }

    if (s_collectVuWork) g_ps2xVuInstr.fetch_add(instrCount, std::memory_order_relaxed);
    // A VU1 program that stops on the cycle budget instead of its E-bit is left
    // mid-draw (possibly mid-XGKICK), so the GS sees a truncated packet and the
    // next MSCAL starts over: garbage strips that come and go. Report it.
    if (!programEnded && m_unit == Unit::VU1)
    {
        static std::atomic<uint64_t> s_budgetStops{0u};
        const uint64_t n = s_budgetStops.fetch_add(1u, std::memory_order_relaxed) + 1u;
        if (n <= 20u || (n % 1000u) == 0u)
            std::fprintf(stderr, "[vu1:budget] program stopped by cycle budget (#%llu) pc=0x%x budget=%llu\n",
                         static_cast<unsigned long long>(n), m_state.pc,
                         static_cast<unsigned long long>(maxCycles));
    }
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    const uint64_t vpFlushT0 = g_vpOn ? __rdtsc() : 0u;
    const uint64_t vpCycleBeforeFlush = m_cycle;
#endif
    if (programEnded)
    {
        // The exact path's flush ticks on until its queued VF/VI/ACC writes land;
        // the fast path never queued them. m_maxReadyCycle is the cycle the last
        // of those writes lands (same latencies the queues use), so stepping to
        // it first ends the program on the same cycle, with flags, Q/P and
        // XGKICK advancing cycle by cycle exactly as in the exact path. Found by
        // the differential harness: fast runs ended 2-3 cycles early.
        if (m_immediateWrites)
            advanceTo(m_maxReadyCycle);
        flushPipelines();
        m_state.ebit = false;
        m_state.haltAfterDelaySlot = false;
        m_pendingHaltD = false;
        m_pendingHaltT = false;
    }
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    if (g_vpOn && !t_vu1InShadow)
    {
        vpAdd(kVpFlush, __rdtsc() - vpFlushT0);
        g_vpCycles += m_cycle - vpCycleAtStart;
        g_vpFlushCycles += m_cycle - vpCycleBeforeFlush;
        g_vpPairs += instrCount;
        ++g_vpRuns;
    }
#endif
    if (diffArmed)
        diffFinish(vuData, dataSize, programEnded);
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    {
        auto validMask = [](const auto &slots)
        {
            uint32_t mask = 0u;
            for (size_t i = 0; i < slots.size(); ++i)
                if (slots[i].valid)
                    mask |= 1u << i;
            return mask;
        };
        const uint32_t f = validMask(m_flagPipeline), st = validMask(m_storePipeline),
                       vf = validMask(m_vfWritePipeline), vi = validMask(m_viWritePipeline),
                       ac = validMask(m_accWritePipeline);
        if (f != m_flagMask || st != m_storeMask || vf != m_vfWriteMask || vi != m_viWriteMask || ac != m_accWriteMask)
        {
            static std::atomic<uint32_t> s_maskErrors{0u};
            if (s_maskErrors.fetch_add(1u, std::memory_order_relaxed) < 8u)
                std::fprintf(stderr, "[vu1:mask] OCCUPANCY MASK MISMATCH flag=%x/%x store=%x/%x vf=%x/%x vi=%x/%x acc=%x/%x\n",
                             f, m_flagMask, st, m_storeMask, vf, m_vfWriteMask, vi, m_viWriteMask, ac, m_accWriteMask);
        }
    }
#endif
    m_state.cycles = m_cycle;
    if (useVuRounding && previousRoundingMode != -1)
        std::fesetround(previousRoundingMode);
}

// ---------------------------------------------------------------------------
// Differential harness for the Stage 1a fast path (PS2X_VU1_DIFF=1).
// ---------------------------------------------------------------------------

void VU1Interpreter::diffBegin(uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData, uint32_t dataSize,
                               GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    Vu1DiffState &d = vu1Diff();
    if (!d.shadow)
        d.shadow = std::make_unique<VU1Interpreter>(Unit::VU1);
    *d.shadow = *this;
    d.startPc = m_state.pc;
    d.shadowData.assign(vuData, vuData + dataSize);
    d.shadowGif.clear();
    d.authGif.clear();
    d.shadow->m_immediateWrites = true;
    d.shadow->m_gifCapture = &d.shadowGif;
    d.shadow->m_gifCaptureOnly = true;
    g_traceAuth.clear();
    g_traceShadow.clear();
    t_vu1InShadow = true;
    d.shadow->run(vuCode, codeSize, d.shadowData.data(), dataSize, gs, memory, maxCycles);
    t_vu1InShadow = false;
    // Capture the real pass's packets too (it still sends them to the GS).
    m_gifCapture = &d.authGif;
    m_gifCaptureOnly = false;
}

void VU1Interpreter::diffFinish(uint8_t *vuData, uint32_t dataSize, bool programEnded)
{
    Vu1DiffState &d = vu1Diff();
    m_gifCapture = nullptr;
    if (!programEnded)
    {
        // Budget-limited: the exact pass still has queued writes in flight, so
        // the two states are not comparable yet.
        ++d.skipped;
        return;
    }
    ++d.compared;
    const VU1Interpreter &s = *d.shadow;
    auto bits = [](float f)
    {
        uint32_t u = 0u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    };
    char what[200] = {0};
    for (int r = 0; r < 32 && !what[0]; ++r)
        for (int c = 0; c < 4; ++c)
            if (bits(m_state.vf[r][c]) != bits(s.m_state.vf[r][c]))
            {
                std::snprintf(what, sizeof(what), "vf%d.%c exact=%08x fast=%08x", r, "xyzw"[c],
                              bits(m_state.vf[r][c]), bits(s.m_state.vf[r][c]));
                break;
            }
    for (int r = 0; r < 16 && !what[0]; ++r)
        if (m_state.vi[r] != s.m_state.vi[r])
            std::snprintf(what, sizeof(what), "vi%d exact=%d fast=%d", r, m_state.vi[r], s.m_state.vi[r]);
    for (int c = 0; c < 4 && !what[0]; ++c)
        if (bits(m_state.acc[c]) != bits(s.m_state.acc[c]))
            std::snprintf(what, sizeof(what), "acc.%c exact=%08x fast=%08x", "xyzw"[c],
                          bits(m_state.acc[c]), bits(s.m_state.acc[c]));
    if (!what[0] && bits(m_state.q) != bits(s.m_state.q))
        std::snprintf(what, sizeof(what), "Q exact=%08x fast=%08x", bits(m_state.q), bits(s.m_state.q));
    if (!what[0] && bits(m_state.p) != bits(s.m_state.p))
        std::snprintf(what, sizeof(what), "P exact=%08x fast=%08x", bits(m_state.p), bits(s.m_state.p));
    if (!what[0] && bits(m_state.i) != bits(s.m_state.i))
        std::snprintf(what, sizeof(what), "I exact=%08x fast=%08x", bits(m_state.i), bits(s.m_state.i));
    static const bool s_diffRelaxedFlags = []
    {
        const char *value = std::getenv("PS2X_VU1_RELAX");
        const char *values = std::getenv("PS2X_VU1_DIFF_VALUES"); // compare values only (exact interpreter vs relaxed native)
        return (value != nullptr && value[0] == '1') || (values != nullptr && values[0] == '1');
    }();
    if (!what[0] && !s_diffRelaxedFlags && m_state.mac != s.m_state.mac)
        std::snprintf(what, sizeof(what), "MAC exact=%x fast=%x", m_state.mac, s.m_state.mac);
    if (!what[0] && !s_diffRelaxedFlags && m_state.status != s.m_state.status)
        std::snprintf(what, sizeof(what), "STATUS exact=%x fast=%x", m_state.status, s.m_state.status);
    if (!what[0] && m_state.clip != s.m_state.clip)
        std::snprintf(what, sizeof(what), "CLIP exact=%x fast=%x", m_state.clip, s.m_state.clip);
    if (!what[0] && m_state.pc != s.m_state.pc)
        std::snprintf(what, sizeof(what), "PC exact=0x%x fast=0x%x", m_state.pc, s.m_state.pc);
    // PS2X_VU1_RELAX=1: the native programs were generated in relaxed mode
    // (no cycle-accurate pipeline timing), so cycle counts legitimately differ;
    // everything observable (registers, VU memory, GIF output) must still match.
    static const bool s_diffRelaxed = []
    {
        const char *value = std::getenv("PS2X_VU1_RELAX");
        const char *values = std::getenv("PS2X_VU1_DIFF_VALUES"); // compare values only (exact interpreter vs relaxed native)
        return (value != nullptr && value[0] == '1') || (values != nullptr && values[0] == '1');
    }();
    if (!what[0] && !s_diffRelaxed && m_cycle != s.m_cycle)
        std::snprintf(what, sizeof(what), "cycle exact=%llu fast=%llu",
                      static_cast<unsigned long long>(m_cycle), static_cast<unsigned long long>(s.m_cycle));
    if (!what[0] && std::memcmp(vuData, d.shadowData.data(), dataSize) != 0)
    {
        uint32_t off = 0u;
        while (off < dataSize && vuData[off] == d.shadowData[off])
            ++off;
        std::snprintf(what, sizeof(what), "VU data mem first diff at 0x%x", off);
    }
    if (!what[0] && (d.authGif.size() != d.shadowGif.size() ||
                     std::memcmp(d.authGif.data(), d.shadowGif.data(), d.authGif.size()) != 0))
    {
        size_t off = 0u;
        const size_t n = std::min(d.authGif.size(), d.shadowGif.size());
        while (off < n && d.authGif[off] == d.shadowGif[off])
            ++off;
        std::snprintf(what, sizeof(what), "GIF output exact=%zu bytes fast=%zu bytes first diff at %zu",
                      d.authGif.size(), d.shadowGif.size(), off);
    }
    // Per-program tally (image hash, start PC): which programs differ at all,
    // printed every 20000 compares -- decides relaxed vs exact per program.
    {
        static std::mutex s_tallyMutex;
        static std::map<std::pair<uint64_t, uint32_t>, std::pair<uint64_t, uint64_t>> s_tally;
        static std::map<std::pair<uint64_t, uint32_t>, std::string> s_firstWhat;
        const uint64_t hash = m_cachedVuCode ? codeHash(m_cachedVuCode, m_cachedCodeSize) : 0u;
        std::lock_guard<std::mutex> lk(s_tallyMutex);
        auto &e = s_tally[{hash, d.startPc}];
        ++e.first;
        if (what[0])
        {
            ++e.second;
            s_firstWhat.emplace(std::make_pair(hash, d.startPc), std::string(what));
        }
        if (((d.compared + 1u) % 20000u) == 0u)
        {
            std::fprintf(stderr, "[vu1:tally] --- per-program (image startPc: mismatched/compared first-diff) ---\n");
            for (const auto &kv : s_tally)
            {
                auto fw = s_firstWhat.find(kv.first);
                std::fprintf(stderr, "[vu1:tally] %016llx 0x%04x: %llu/%llu %s\n",
                             static_cast<unsigned long long>(kv.first.first), kv.first.second,
                             static_cast<unsigned long long>(kv.second.second),
                             static_cast<unsigned long long>(kv.second.first),
                             fw != s_firstWhat.end() ? fw->second.c_str() : "");
            }
        }
    }
    if (what[0])
    {
        ++d.mismatched;
        if (d.mismatched <= 4u)
        {
            // The chain above names only the FIRST differing field; a timing
            // difference (cycle count) or a Q difference behind a VF mismatch
            // would otherwise stay hidden. List the rest independently.
            char also[400] = {0};
            int alen = 0;
            auto note = [&](const char *tag)
            {
                if (alen < static_cast<int>(sizeof(also)) - 130)
                    alen += std::snprintf(also + alen, sizeof(also) - static_cast<size_t>(alen), " %s", tag);
            };
            if (bits(m_state.q) != bits(s.m_state.q))
                note("Q");
            if (m_state.mac != s.m_state.mac)
                note("MAC");
            if (m_state.status != s.m_state.status)
                note("STATUS");
            if (m_state.clip != s.m_state.clip)
                note("CLIP");
            if (m_state.pc != s.m_state.pc)
                note("PC");
            if (m_cycle != s.m_cycle)
            {
                char cyc[48];
                std::snprintf(cyc, sizeof(cyc), "CYCLE(exact=%llu fast=%llu)",
                              static_cast<unsigned long long>(m_cycle), static_cast<unsigned long long>(s.m_cycle));
                note(cyc);
            }
            if (std::memcmp(vuData, d.shadowData.data(), dataSize) != 0)
            {
                // Which qwords differ: the addresses say whether a queued store
                // landed on one side but not the other before a load.
                uint32_t nDiff = 0u;
                char dm[96];
                int dl = std::snprintf(dm, sizeof(dm), "DMEM(qw");
                for (uint32_t qw = 0; qw * 16u < dataSize; ++qw)
                {
                    if (std::memcmp(vuData + qw * 16u, d.shadowData.data() + qw * 16u, 16u) != 0)
                    {
                        if (nDiff < 6u && dl < static_cast<int>(sizeof(dm)) - 8)
                            dl += std::snprintf(dm + dl, sizeof(dm) - static_cast<size_t>(dl), " %u", qw);
                        ++nDiff;
                    }
                }
                std::snprintf(dm + dl, sizeof(dm) - static_cast<size_t>(dl), " n=%u)", nDiff);
                note(dm);
            }
            // Every differing VF lane (the chain above names only the first).
            {
                char vfl[120];
                int vl = std::snprintf(vfl, sizeof(vfl), "VF[");
                uint32_t nLanes = 0u;
                for (int r = 0; r < 32; ++r)
                    for (int c = 0; c < 4; ++c)
                        if (bits(m_state.vf[r][c]) != bits(s.m_state.vf[r][c]))
                        {
                            if (nLanes < 6u && vl < static_cast<int>(sizeof(vfl)) - 12)
                                vl += std::snprintf(vfl + vl, sizeof(vfl) - static_cast<size_t>(vl), " vf%d.%c", r, "xyzw"[c]);
                            ++nLanes;
                        }
                std::snprintf(vfl + vl, sizeof(vfl) - static_cast<size_t>(vl), " n=%u]", nLanes);
                note(vfl);
            }
            if (d.authGif.size() != d.shadowGif.size() ||
                std::memcmp(d.authGif.data(), d.shadowGif.data(), d.authGif.size()) != 0)
                note("GIF");
            // Locate the first place the two passes diverged in either the
            // instruction sequence or the cycle each instruction started on.
            {
                const uint32_t n = std::min(g_traceAuth.count, g_traceShadow.count);
                uint32_t at = 0u;
                while (at < n && g_traceAuth.pcs[at] == g_traceShadow.pcs[at] &&
                       g_traceAuth.cycles[at] == g_traceShadow.cycles[at])
                    ++at;
                if (at > 0u)
                {
                    const uint32_t k = at - 1u; // the pair whose processing diverged
                    std::fprintf(stderr, "[vu1:diff]   stall inputs at pc=0x%04x: auth maxReady=%llu fdiv=%llu efu=%llu | shadow maxReady=%llu fdiv=%llu efu=%llu\n",
                                 g_traceAuth.pcs[k],
                                 static_cast<unsigned long long>(g_traceAuth.maxReady[k]),
                                 static_cast<unsigned long long>(g_traceAuth.fdiv[k]),
                                 static_cast<unsigned long long>(g_traceAuth.efu[k]),
                                 static_cast<unsigned long long>(g_traceShadow.maxReady[k]),
                                 static_cast<unsigned long long>(g_traceShadow.fdiv[k]),
                                 static_cast<unsigned long long>(g_traceShadow.efu[k]));
                }
                {
                    // First pair, on a shared path, that reads Q / MAC / CLIP / STATUS
                    // while exact and relaxed disagree on that value.
                    const uint32_t shared = std::min(g_traceAuth.count, g_traceShadow.count);
                    for (uint32_t k = 0; k < shared; ++k)
                    {
                        if (g_traceAuth.pcs[k] != g_traceShadow.pcs[k])
                        {
                            std::fprintf(stderr, "[vu1:value]   control flow splits at index %u: auth pc=0x%04x shadow pc=0x%04x\n",
                                         k, g_traceAuth.pcs[k], g_traceShadow.pcs[k]);
                            break;
                        }
                        const uint32_t pc = g_traceAuth.pcs[k];
                        if (!m_cachedVuCode || pc + 8u > m_cachedCodeSize)
                            continue;
                        uint32_t lower = 0u, upper = 0u;
                        std::memcpy(&lower, m_cachedVuCode + pc, 4);
                        std::memcpy(&upper, m_cachedVuCode + pc + 4u, 4);
                        const bool iBit = (upper >> 31) & 1u;
                        const uint32_t op = upper & 0x3Fu;
                        const uint32_t sel = op >= 0x3Cu ? ((upper & 3u) | ((upper >> 4) & 0x7Cu)) : op;
                        const bool readsQ = sel == 0x1Cu || sel == 0x20u || sel == 0x21u || sel == 0x24u || sel == 0x25u;
                        const uint32_t lop = iBit ? 0xFFu : (lower >> 25);
                        const bool readsMac = lop == 0x18u || lop == 0x1Au || lop == 0x1Bu;
                        const bool readsClip = lop == 0x10u || lop == 0x12u || lop == 0x13u || lop == 0x1Cu;
                        const bool readsStatus = lop >= 0x14u && lop <= 0x17u;
                        const char *what2 = nullptr;
                        if (readsQ && g_traceAuth.qBits[k] != g_traceShadow.qBits[k]) what2 = "Q";
                        else if (readsMac && g_traceAuth.mac[k] != g_traceShadow.mac[k]) what2 = "MAC";
                        else if (readsClip && (g_traceAuth.clip[k] & 0xFFFFFFu) != (g_traceShadow.clip[k] & 0xFFFFFFu)) what2 = "CLIP";
                        else if (readsStatus && g_traceAuth.status[k] != g_traceShadow.status[k]) what2 = "STATUS";
                        if (what2)
                        {
                            float qa = 0, qs = 0;
                            std::memcpy(&qa, &g_traceAuth.qBits[k], 4);
                            std::memcpy(&qs, &g_traceShadow.qBits[k], 4);
                            std::fprintf(stderr,
                                "[vu1:value]   first differing READ at index %u pc=0x%04x U=%08x L=%08x reads %s: exact Q=%g mac=%x clip=%x st=%x | relaxed Q=%g mac=%x clip=%x st=%x\n",
                                k, pc, upper, lower, what2, qa, g_traceAuth.mac[k], g_traceAuth.clip[k], g_traceAuth.status[k],
                                qs, g_traceShadow.mac[k], g_traceShadow.clip[k], g_traceShadow.status[k]);
                            break;
                        }
                    }
                }
                std::fprintf(stderr, "[vu1:diff]   trace: auth=%u pairs shadow=%u pairs; first divergence at index %u\n",
                             g_traceAuth.count, g_traceShadow.count, at);
                const uint32_t from = at > 3u ? at - 3u : 0u;
                for (uint32_t i = from; i < std::min(n, at + 4u); ++i)
                    std::fprintf(stderr, "[vu1:diff]     [%u] auth pc=0x%04x cyc=%llu | shadow pc=0x%04x cyc=%llu%s\n",
                                 i, g_traceAuth.pcs[i], static_cast<unsigned long long>(g_traceAuth.cycles[i]),
                                 g_traceShadow.pcs[i], static_cast<unsigned long long>(g_traceShadow.cycles[i]),
                                 i == at ? "   <-- first divergence" : "");
            }
            std::fprintf(stderr, "[vu1:diff] MISMATCH #%llu prog=%016llx startPc=0x%x %s also:[%s ]\n",
                         static_cast<unsigned long long>(d.mismatched),
                         static_cast<unsigned long long>(m_cachedVuCode ? codeHash(m_cachedVuCode, m_cachedCodeSize) : 0u),
                         d.startPc, what, also);
        }
    }
    if ((d.compared % 5000u) == 0u)
        std::fprintf(stderr, "[vu1:diff] compared=%llu mismatched=%llu skipped(budget)=%llu\n",
                     static_cast<unsigned long long>(d.compared), static_cast<unsigned long long>(d.mismatched),
                     static_cast<unsigned long long>(d.skipped));
}

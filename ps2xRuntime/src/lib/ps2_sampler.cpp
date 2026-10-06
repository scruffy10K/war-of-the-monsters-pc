// PS2X_SAMPLE=1: in-process sampling profiler for the game thread (Windows).
//
// A background thread suspends the game thread about once per millisecond,
// reads its instruction pointer and resumes it. Samples are bucketed by
// image-relative address (RVA) and written every 10 s to PS2X_SAMPLE_OUT
// (default: ps2x_samples.txt in the working directory) as "rva count" lines,
// most frequent first. Map RVAs to functions with the linker map (/MAP) --
// see scratchpad map_samples.py. The rdtsc profilers inflate small costs;
// this does not touch the sampled code at all.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib") // timeBeginPeriod: 1 ms sampling resolution

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

extern std::atomic<uint64_t> g_ps2xWotmCompletedFrames;

namespace
{
    // Nearest exported symbol at or below `rva` in a loaded module (walks the
    // in-memory PE export table), so DLL samples read "NtYieldExecution+0x14"
    // instead of a bare offset. Good enough for system/CRT DLLs.
    std::string nearestExport(HMODULE mod, uint32_t rva)
    {
        if (!mod)
            return "?";
        const auto *base = reinterpret_cast<const uint8_t *>(mod);
        const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
        if (dos->e_magic != IMAGE_DOS_SIGNATURE)
            return "?";
        const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS *>(base + dos->e_lfanew);
        const IMAGE_DATA_DIRECTORY &dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
        if (dir.VirtualAddress == 0u || dir.Size == 0u)
            return "?";
        const auto *exp = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY *>(base + dir.VirtualAddress);
        const auto *names = reinterpret_cast<const DWORD *>(base + exp->AddressOfNames);
        const auto *ordinals = reinterpret_cast<const WORD *>(base + exp->AddressOfNameOrdinals);
        const auto *functions = reinterpret_cast<const DWORD *>(base + exp->AddressOfFunctions);
        uint32_t bestRva = 0u;
        const char *bestName = nullptr;
        for (DWORD i = 0; i < exp->NumberOfNames; ++i)
        {
            const uint32_t fnRva = functions[ordinals[i]];
            if (fnRva <= rva && fnRva >= bestRva)
            {
                bestRva = fnRva;
                bestName = reinterpret_cast<const char *>(base + names[i]);
            }
        }
        if (!bestName)
            return "?";
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s+0x%x", bestName, rva - bestRva);
        return buf;
    }

    std::atomic<HANDLE> g_sampleTarget{nullptr};
    std::atomic<bool> g_samplerStarted{false};

    void samplerMain()
    {
        const char *outPath = std::getenv("PS2X_SAMPLE_OUT");
        if (!outPath || !outPath[0])
            outPath = "ps2x_samples.txt";
        const uintptr_t imageBase = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
        std::unordered_map<uint32_t, uint32_t> hist;
        std::map<HMODULE, uint32_t> moduleHist;
        std::map<std::pair<HMODULE, uint32_t>, uint32_t> moduleRvaHist;
        uint64_t total = 0u, outside = 0u;
        auto lastWrite = std::chrono::steady_clock::now();
        const char *startValue=std::getenv("PS2X_SAMPLE_START_FRAME");
        const char *endValue=std::getenv("PS2X_SAMPLE_END_FRAME");
        const uint64_t startFrame=startValue?std::strtoull(startValue,nullptr,10):0;
        const uint64_t endFrame=endValue?std::strtoull(endValue,nullptr,10):~0ull;
        uint64_t firstSampleFrame=0,lastSampleFrame=0;
        timeBeginPeriod(1);
        for (;;)
        {
            Sleep(1);
            const uint64_t sampleFrame=g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);
            if(sampleFrame<startFrame || sampleFrame>endFrame) continue;
            HANDLE target = g_sampleTarget.load(std::memory_order_acquire);
            if (!target)
                continue;
            if (SuspendThread(target) == static_cast<DWORD>(-1))
                continue;
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_CONTROL;
            const BOOL ok = GetThreadContext(target, &ctx);
            ResumeThread(target);
            if (!ok)
                continue;
            if(!total) firstSampleFrame=sampleFrame;
            lastSampleFrame=sampleFrame;
            ++total;
            const uintptr_t rip = static_cast<uintptr_t>(ctx.Rip);
            if (rip >= imageBase && rip - imageBase < 0x7FFFFFFFu)
                ++hist[static_cast<uint32_t>(rip - imageBase)];
            else
            {
                ++outside; // DLLs (CRT, ffmpeg, kernel): attribute to the module + RVA
                HMODULE mod = nullptr;
                if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                       reinterpret_cast<LPCWSTR>(rip), &mod) &&
                    mod)
                {
                    ++moduleHist[mod];
                    ++moduleRvaHist[{mod, static_cast<uint32_t>(rip - reinterpret_cast<uintptr_t>(mod))}];
                }
                else
                    ++moduleHist[nullptr];
            }
            const auto now = std::chrono::steady_clock::now();
            if (now - lastWrite >= std::chrono::seconds(10))
            {
                lastWrite = now;
                std::vector<std::pair<uint32_t, uint32_t>> order(hist.begin(), hist.end());
                std::sort(order.begin(), order.end(),
                          [](const auto &a, const auto &b) { return a.second > b.second; });
                // Write to a temporary file and swap it in: rewriting outPath in
                // place truncates it first, so a process that exits mid-write
                // (PS2X_EXIT_AFTER) left an empty profile behind.
                const std::string tmpPath = std::string(outPath) + ".tmp";
                if (std::FILE *f = std::fopen(tmpPath.c_str(), "w"))
                {
                    std::fprintf(f, "# total=%llu outside_image=%llu base=%llx\n",
                                 static_cast<unsigned long long>(total), static_cast<unsigned long long>(outside),
                                 static_cast<unsigned long long>(imageBase));
                    std::fprintf(f,"# frames=%llu..%llu\n",firstSampleFrame,lastSampleFrame);
                    // Samples outside the exe, per DLL (and its hottest offsets).
                    for (const auto &[mod, n] : moduleHist)
                    {
                        char name[MAX_PATH] = "?";
                        if (mod)
                            GetModuleFileNameA(mod, name, MAX_PATH);
                        std::fprintf(f, "# module %u %s\n", n, name);
                    }
                    std::vector<std::pair<uint32_t, std::pair<HMODULE, uint32_t>>> modRvas;
                    for (const auto &[key, n] : moduleRvaHist)
                        modRvas.push_back({n, key});
                    std::sort(modRvas.rbegin(), modRvas.rend());
                    for (size_t i = 0; i < modRvas.size() && i < 25u; ++i)
                    {
                        char name[MAX_PATH] = "?";
                        GetModuleFileNameA(modRvas[i].second.first, name, MAX_PATH);
                        const char *base = std::strrchr(name, '\\');
                        std::fprintf(f, "# modrva %u %s+%x %s\n", modRvas[i].first, base ? base + 1 : name,
                                     modRvas[i].second.second,
                                     nearestExport(modRvas[i].second.first, modRvas[i].second.second).c_str());
                    }
                    for (size_t i = 0; i < order.size() && i < 20000u; ++i)
                        std::fprintf(f, "%x %u\n", order[i].first, order[i].second);
                    std::fclose(f);
                    MoveFileExA(tmpPath.c_str(), outPath, MOVEFILE_REPLACE_EXISTING);
                }
            }
        }
    }
}

// Select one real execution thread; the EE may mostly wait on the VU worker.
// Default remains the game thread. Profiling stays explicitly opt-in.
static void registerSampleThread(const char* role)
{
    static const bool s_enabled = []
    {
        const char *value = std::getenv("PS2X_SAMPLE");
        return value != nullptr && value[0] == '1';
    }();
    if (!s_enabled)
        return;
    const char* selected=std::getenv("PS2X_SAMPLE_TARGET");
    if(std::strcmp(role,selected && *selected ? selected : "game")!=0)
        return;
    HANDLE self = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &self,
                         THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0))
        return;
    g_sampleTarget.store(self, std::memory_order_release);
    bool expected = false;
    if (g_samplerStarted.compare_exchange_strong(expected, true))
    {
        std::thread(samplerMain).detach();
        std::fprintf(stderr, "[sample] sampling the %s thread (PS2X_SAMPLE_OUT or ps2x_samples.txt)\n",role);
    }
}
void ps2xSamplerRegisterGameThread(){registerSampleThread("game");}
void ps2xSamplerRegisterVu1Thread(){registerSampleThread("vu1");}
#else
void ps2xSamplerRegisterGameThread() {}
void ps2xSamplerRegisterVu1Thread() {}
#endif

#include "Common.h"
#include "Interrupt.h"


#include <cstdlib>
#include "wotm_native_field.inc"

extern uint64_t ps2xPacingDispatchNs();
extern thread_local uint64_t t_ps2xDeadlineSleepNs, t_ps2xClippedNs;
extern std::atomic<uint64_t> g_ps2xEeIdleWaitNs;
extern std::atomic<uint64_t> g_ps2xVu1DrainWaitNs, g_ps2xRasterIdleWaitNs, g_ps2xFinishWaitNs;

extern bool ps2xNativePacingEnabled();
namespace {
    // Diagnostic replay only: remove host-speed-dependent simulation catch-up.
    // Never enabled by the play launcher; require an explicit input replay too.
    const bool s_benchFixedFields=[] {
        const char *p=std::getenv("PS2X_BENCH_FIXED_FIELDS");
        return p && p[0]=='1' && std::getenv("PS2X_PAD_REPLAY");
    }();
    thread_local uint64_t s_pacingFrameTick=0, s_pacingDeliveredTick=0;
    thread_local std::chrono::steady_clock::time_point s_pacingFrameTime;
}
void ps2xPacingFrameComplete(PS2Runtime *runtime) {
    if (!ps2xNativePacingEnabled()) return;
    s_pacingFrameTick=s_pacingDeliveredTick ? s_pacingDeliveredTick : runtime->eeScheduler().currentVSyncTick();
    s_pacingDeliveredTick=0;
    s_pacingFrameTime=std::chrono::steady_clock::now();
}

// Opt-in tracing only; all calls occur on the EE executor thread.
void ps2xTracePacing(const char *event, uint8_t *rdram, PS2Runtime *runtime, uint32_t ra)
{
    static const char *path = std::getenv("PS2X_PACING_CSV");
    if (!path || !*path) return;
    static std::FILE *file = [] (const char *p) {
        auto *f = std::fopen(p, "w");
        if (f) std::fprintf(f, "event,ns,ra,tick,fields,rate,target,dispatch_ns,deadline_sleep_ns,idle_ns,clipped_ns,worker_wait_ns,raster_wait_ns,finish_wait_ns\n");
        return f;
    }(path);
    if (!file) return;
    static const auto start = std::chrono::steady_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
    auto read = [rdram](uint32_t address) { uint32_t x=0; std::memcpy(&x,rdram+address,sizeof(x)); return x; };
    std::fprintf(file,"%s,%lld,%x,%llu,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",event,static_cast<long long>(ns),ra,
        static_cast<unsigned long long>(runtime->eeScheduler().currentVSyncTick()),
        read(0x6f8db4u),read(0x6f8db8u),read(0x6f8dd8u),
        static_cast<unsigned long long>(ps2xPacingDispatchNs()),
        static_cast<unsigned long long>(t_ps2xDeadlineSleepNs),
        static_cast<unsigned long long>(g_ps2xEeIdleWaitNs.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(t_ps2xClippedNs),
        static_cast<unsigned long long>(g_ps2xVu1DrainWaitNs.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_ps2xRasterIdleWaitNs.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_ps2xFinishWaitNs.load(std::memory_order_relaxed)));
    static unsigned rows=0;
    if (++rows%120u==0u) std::fflush(file);
}

namespace ps2_syscalls
{
    namespace
    {
        EeScheduler &scheduler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
        {
            EeScheduler &result = runtime->eeScheduler();
            result.bindMainContextForSyscall(*ctx, rdram);
            return result;
        }

        void setCauseEnabled(uint8_t *rdram,
                             R5900Context *ctx,
                             PS2Runtime *runtime,
                             bool dmac,
                             bool enabled)
        {
            setReturnS32(ctx,
                         scheduler(rdram, ctx, runtime)
                             .setIrqCauseEnabled(dmac, getRegU32(ctx, 4), enabled));
        }

        void addHandler(uint8_t *rdram,
                        R5900Context *ctx,
                        PS2Runtime *runtime,
                        bool dmac)
        {
            // The EE kernel runs INTC/DMAC handlers on its own interrupt stack, never
            // on the registering thread's. Recording the caller's $sp here made the
            // handler's frame land inside that thread's *live* stack (WotM: the
            // VBLANK handler installed by the FMV player clobbered main's saved $ra
            // → main "returned" to 0 and the intro hung). sp=0 makes the scheduler
            // use its dedicated invocation stack.
            const int id = scheduler(rdram, ctx, runtime)
                               .addIrqHandler(dmac,
                                              getRegU32(ctx, 4),
                                              getRegU32(ctx, 5),
                                              getRegU32(ctx, 6) != 0u,
                                              getRegU32(ctx, 7),
                                              getRegU32(ctx, 28),
                                              0u);
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
            std::fprintf(stderr, "[irq:add] %s cause=%u handler=0x%x arg=0x%x append=%d -> id=%d\n",
                         dmac ? "dmac" : "intc", getRegU32(ctx, 4), getRegU32(ctx, 5),
                         getRegU32(ctx, 7), (int)(getRegU32(ctx, 6) != 0u), id);
#endif
            setReturnS32(ctx, id);
        }

        void removeHandler(uint8_t *rdram,
                           R5900Context *ctx,
                           PS2Runtime *runtime,
                           bool dmac)
        {
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
            std::fprintf(stderr, "[irq:del] %s cause=%u id=%d\n",
                         dmac ? "dmac" : "intc", getRegU32(ctx, 4),
                         static_cast<int>(getRegU32(ctx, 5)));
#endif
            setReturnS32(ctx,
                         scheduler(rdram, ctx, runtime)
                             .removeIrqHandler(dmac,
                                               getRegU32(ctx, 4),
                                               static_cast<int>(getRegU32(ctx, 5))));
        }

        void setHandlerEnabled(uint8_t *rdram,
                               R5900Context *ctx,
                               PS2Runtime *runtime,
                               bool dmac,
                               bool enabled)
        {
            setReturnS32(ctx,
                         scheduler(rdram, ctx, runtime)
                             .setIrqHandlerEnabled(dmac,
                                                   static_cast<int>(getRegU32(ctx, 5)),
                                                   enabled));
        }
    }

    void dispatchDmacHandlersForCause(uint8_t *, PS2Runtime *runtime, uint32_t cause)
    {
        runtime->eeScheduler().dispatchIrq(true, cause);
    }

    uint64_t GetCurrentVSyncTick(PS2Runtime *runtime)
    {
        return runtime->eeScheduler().currentVSyncTick();
    }

    void WaitVSyncTick(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, int fixedResult)
    {
        EeScheduler &ee = scheduler(rdram, ctx, runtime);
        static const bool trace = std::getenv("PS2X_PACING_CSV") != nullptr;
        const uint32_t ra = getRegU32(ctx,31);
        uint32_t target=0; std::memcpy(&target,rdram+0x6f8dd8u,sizeof(target));
        // Only the native game's render-configuration handshake is adapted.
        // Its ordinary frame wait below still delivers the 60 Hz gameplay tick.
        static const bool nativeField=[] {
            const char *gpu=std::getenv("PS2X_GS_GPU");
            const char *p=std::getenv("PS2X_NATIVE_FIELD_SYNC");
            return gpu && gpu[0]=='1' && (!p || p[0]!='0');
        }();
        if (ra==0x001884b0u && nativeField && ps2xNativePacingEnabled()) {
            const auto *saved=getEeGuestStruct<uint32_t>(rdram,getRegU32(ctx,29));
            const bool recent=s_pacingFrameTick &&
                std::chrono::steady_clock::now()-s_pacingFrameTime < std::chrono::milliseconds(100);
            int result=0;
            if (saved && wotmNativeConfigField(true,ra,fixedResult,target,recent,*saved,result)) {
                if (trace) ps2xTracePacing("field-config",rdram,runtime,ra);
                setReturnS32(ctx,result);
                return;
            }
        }
        const bool frameWait=ra==0x2234f8u || ra==0x22350cu;
        if (frameWait && ps2xNativePacingEnabled() && target==60u && s_pacingFrameTick &&
            std::chrono::steady_clock::now()-s_pacingFrameTime < std::chrono::milliseconds(100)) {
            const uint64_t previous=s_pacingFrameTick;
            s_pacingFrameTick=0; // Consume once; pause/menu loops must not reuse it.
            auto finish=[rdram,runtime,previous,ra](R5900Context &resumed) {
                const uint64_t delivered=runtime->eeScheduler().currentVSyncTick();
                const uint64_t elapsed=delivered-previous;
                s_pacingDeliveredTick=delivered;
                // timer1Handler converts s0 into ceil(s0/9619) simulation fields.
                // Supply elapsed fields, preserving the retail count/update path.
                const uint32_t fields=s_benchFixedFields ? 1u : static_cast<uint32_t>(std::min<uint64_t>(15u,std::max<uint64_t>(1u,elapsed)));
                SET_GPR_U32(&resumed,16,fields*9619u);
                ps2xTracePacing("resume",rdram,runtime,ra);
            };
            ps2xTracePacing("wait",rdram,runtime,ra);
            const uint64_t tick=ee.currentVSyncTick();
            if (tick>previous) {
                setReturnS32(ctx,fixedResult>=0 ? fixedResult : static_cast<int>((tick-1u)&1u));
                finish(*ctx);
                return;
            }
            ee.waitVSync(previous,fixedResult,std::move(finish));
        }

        // Retail pause/menu waits invalidate any pending native frame handoff.
        // Otherwise the first frame after a long pause could reuse an old tick.
        if (frameWait) s_pacingDeliveredTick=0;
        if (trace && (ra==0x2234f8u || ra==0x22350cu || ra==0x223624u || ra==0x223638u)) {
            ps2xTracePacing("wait",rdram,runtime,ra);
            ee.waitVSync(ee.currentVSyncTick(),fixedResult,[rdram,runtime,ra](R5900Context &) {
                ps2xTracePacing("resume",rdram,runtime,ra);
            });
        }
        ee.waitVSync(ee.currentVSyncTick(), fixedResult);
    }

    void SetVSyncFlag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t flagAddress = getRegU32(ctx, 4);
        const uint32_t tickAddress = getRegU32(ctx, 5);
        if ((flagAddress != 0u && !getEeGuestStruct<uint32_t>(rdram, flagAddress)) ||
            (tickAddress != 0u && !getEeGuestStruct<uint64_t>(rdram, tickAddress)))
        {
            setReturnS32(ctx, KE_ERROR);
            return;
        }
        scheduler(rdram, ctx, runtime).setVSyncFlag(flagAddress, tickAddress);
        setReturnS32(ctx, KE_OK);
    }

    void EnableIntc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setCauseEnabled(rdram, ctx, runtime, false, true);
    }

    void iEnableIntc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        EnableIntc(rdram, ctx, runtime);
    }

    void DisableIntc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setCauseEnabled(rdram, ctx, runtime, false, false);
    }

    void iDisableIntc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        DisableIntc(rdram, ctx, runtime);
    }

    void AddIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        addHandler(rdram, ctx, runtime, false);
    }

    void AddIntcHandler2(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        AddIntcHandler(rdram, ctx, runtime);
    }

    void RemoveIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        removeHandler(rdram, ctx, runtime, false);
    }

    void AddDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        addHandler(rdram, ctx, runtime, true);
    }

    void AddDmacHandler2(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        AddDmacHandler(rdram, ctx, runtime);
    }

    void RemoveDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        removeHandler(rdram, ctx, runtime, true);
    }

    void EnableIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setHandlerEnabled(rdram, ctx, runtime, false, true);
    }

    void DisableIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setHandlerEnabled(rdram, ctx, runtime, false, false);
    }

    void EnableDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setHandlerEnabled(rdram, ctx, runtime, true, true);
    }

    void DisableDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setHandlerEnabled(rdram, ctx, runtime, true, false);
    }

    void EnableDmac(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setCauseEnabled(rdram, ctx, runtime, true, true);
    }

    void iEnableDmac(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        EnableDmac(rdram, ctx, runtime);
    }

    void DisableDmac(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setCauseEnabled(rdram, ctx, runtime, true, false);
    }

    void iDisableDmac(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        DisableDmac(rdram, ctx, runtime);
    }
}

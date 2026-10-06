#pragma once
// Cheap clock for the always-on [perf] timers on hot paths (one read per VU1
// run, VU0 call, GIF packet, VIF1 transfer: ~500k reads/s in fights).
// steady_clock goes through QueryPerformanceCounter (~0.8% of the game thread);
// __rdtsc is a single instruction. Ticks are scaled to nanoseconds with a ratio
// calibrated once against steady_clock, so every consumer still sees ns.
#include <chrono>
#include <cstdint>
#include <thread>
#include <intrin.h>

struct Ps2xTscClock
{
    using duration = std::chrono::nanoseconds;
    using rep = duration::rep;
    using period = duration::period;
    using time_point = std::chrono::time_point<Ps2xTscClock, duration>;
    static constexpr bool is_steady = true;

    static double nsPerTick() noexcept
    {
        static const double s_ratio = []
        {
            const auto c0 = std::chrono::steady_clock::now();
            const uint64_t t0 = __rdtsc();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const uint64_t t1 = __rdtsc();
            const auto c1 = std::chrono::steady_clock::now();
            const double ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0).count());
            return (t1 > t0) ? ns / static_cast<double>(t1 - t0) : 1.0;
        }();
        return s_ratio;
    }

    static time_point now() noexcept
    {
        static const double s_ratio = nsPerTick();
        return time_point(duration(static_cast<rep>(static_cast<double>(__rdtsc()) * s_ratio)));
    }
};

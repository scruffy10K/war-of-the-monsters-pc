#include "module_factories.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ps2x::iop::detail
{
    namespace
    {
        // War of the Monsters (and other Incognito/989 titles) boot 989SND.IRX and
        // bind a private EE<->IOP RPC service by a hard-coded SID (0x123456 for WotM).
        // The EE side (snd_SendIOPCommandAndWait / snd_FlushSoundCommands) parks in a
        // spin loop inside snd_GotReturns until the IOP writes the sentinel word
        // 0xFFFFFFFF into the return buffer it published through snd_PrepareReturnBuffer:
        //
        //   *(u32*)0x435780            -> pointer to the return buffer (== rpc recv buffer)
        //   *(u32*)0x435784            -> entry count published for this batch
        //   returnBuffer[0]            must become 0xFFFFFFFF
        //   returnBuffer[count*4 + 4]  must become 0xFFFFFFFF
        //
        // We do not emulate the 989 command protocol; we just acknowledge every call by
        // filling the whole receive buffer with 0xFF so both sentinel slots are satisfied
        // regardless of the published count. Audio does not play, but the boot proceeds.
        class Snd989StubService final : public IopService
        {
        public:
            Snd989StubService(IopHost &host, Snd989Bindings bindings)
                : m_host(host), m_bindings(std::move(bindings)), m_sids{m_bindings.sid, m_bindings.loaderSid}
            {
            }

            [[nodiscard]] std::string_view name() const override
            {
                return m_bindings.serviceName;
            }

            [[nodiscard]] std::span<const uint32_t> sids() const override
            {
                // The loader server is optional; don't advertise SID 0.
                return std::span<const uint32_t>(m_sids.data(), m_bindings.loaderSid != 0u ? 2u : 1u);
            }

            void reset() override
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_callCount = 0u;
            }

            [[nodiscard]] RpcResult handleRpc(const RpcRequest &request) override
            {
                if (m_bindings.loaderSid != 0u && request.sid == m_bindings.loaderSid)
                {
                    return handleLoaderRpc(request);
                }
                if (request.sid != m_bindings.sid)
                {
                    return {};
                }
                // [snd989:census] which 989SND commands the game sends (diagnostic,
                // PS2X_SND_CENSUS=1): batch RPC 0x4D carries {u32 count; {u16 id,
                // u16 len, payload padded to 4}...}; other RPC numbers are direct.
                static const bool s_census = [] { const char *p = std::getenv("PS2X_SND_CENSUS"); return p && p[0] == '1'; }();
                if (s_census)
                    census(request);

                if (request.receive.address != 0u && request.receive.size != 0u)
                {
                    const std::vector<uint8_t> sentinel(request.receive.size, 0xFFu);
                    (void)m_host.writeGuest(request.receive.address,
                                            sentinel.data(),
                                            sentinel.size());
                }

                // snd_SendIOPCommandAndWait returns returnBuffer[4] (the slot
                // between the two sentinels). snd_PlayVAGStreamByLoc (command
                // 0x2C) must return a stream handle: the game's
                // startCinemaMusicFromId loops `while (handle == -1 || handle == 0)`
                // re-issuing the call, so the all-0xFF reply hung the end of every
                // Adventure level. Hand back a unique non-zero fake handle.
                if (request.function == 0x2Cu && request.receive.address != 0u && request.receive.size >= 8u)
                {
                    uint32_t handle = 0u;
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        handle = 0x00C00000u + (++m_vagStreams);
                    }
                    (void)m_host.writeGuest(request.receive.address + 4u, &handle, sizeof(handle));
                }

                // The runtime's 989SND player (ps2xRuntime/src/lib/ps2_snd989.cpp)
                // rewrites a batch reply with real per-command return values
                // (stream handles, still-playing state) and plays music streams.
                m_host.audioCommand(request.sid, request.function, request.send, request.receive);

                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    ++m_callCount;
                }

                RpcResult result;
                result.handled = true;
                result.resultAddress = request.receive.address;
                result.signalNowaitCompletion = true;
                result.signalCompletion = true;
                return result;
            }

            void appendDebugMetrics(std::vector<DebugMetric> &metrics) const override
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                metrics.push_back({"call_count", m_callCount, false});
            }

        private:
            // snd_BankLoadByLoc parks in `while (gLoadReturnValue == -1)`, and
            // gLoadReturnValue *is* the 4-byte receive buffer. Without a handler the
            // runtime's fallback echoed the send buffer back, so the game received
            // its own disc location (e.g. 0xB0EE) as the "bank handle". The EE only
            // stores the handle and hands it back to 989SND later (it never
            // dereferences it), so a unique non-zero fake handle is enough until
            // real audio exists.
            RpcResult handleLoaderRpc(const RpcRequest &request)
            {
                uint32_t handle = 0u;
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    handle = 0x00B00000u + (++m_bankLoads) * 0x100u;
                    ++m_callCount;
                }
                if (request.receive.address != 0u && request.receive.size >= sizeof(handle))
                    (void)m_host.writeGuest(request.receive.address, &handle, sizeof(handle));
                // The runtime's 989SND player loads the bank from the disc and
                // replaces the handle with its own.
                m_host.audioCommand(request.sid, request.function, request.send, request.receive);
                RpcResult result;
                result.handled = true;
                result.resultAddress = request.receive.address;
                result.signalNowaitCompletion = true;
                result.signalCompletion = true;
                return result;
            }

            void census(const RpcRequest &request)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto note = [this](uint32_t id, uint32_t len, const uint8_t *payload)
                {
                    auto &e = m_census[id];
                    if (e.count++ < 3u)
                    {
                        char line[256];
                        int n = std::snprintf(line, sizeof(line), "[snd989:cmd] id=0x%02x len=%u", id, len);
                        for (uint32_t k = 0; k < len && k < 24u && n < 230; k += 4u)
                        {
                            uint32_t w = 0u;
                            std::memcpy(&w, payload + k, std::min<uint32_t>(4u, len - k));
                            n += std::snprintf(line + n, sizeof(line) - static_cast<size_t>(n), " %08x", w);
                        }
                        std::fprintf(stderr, "%s\n", line);
                    }
                };
                if (request.function == 0x4Du && request.send.address != 0u && request.send.size >= 4u && request.send.size <= 0x1000u)
                {
                    std::vector<uint8_t> buf(request.send.size);
                    if (!m_host.readGuest(request.send.address, buf.data(), buf.size()))
                        return;
                    uint32_t count = 0u;
                    std::memcpy(&count, buf.data(), 4u);
                    size_t pos = 4u;
                    for (uint32_t i = 0; i < count && pos + 4u <= buf.size(); ++i)
                    {
                        uint16_t id = 0u, len = 0u;
                        std::memcpy(&id, buf.data() + pos, 2u);
                        std::memcpy(&len, buf.data() + pos + 2u, 2u);
                        if (pos + 4u + len > buf.size())
                            break;
                        note(id, len, buf.data() + pos + 4u);
                        pos += 4u + ((len + 3u) & ~3u);
                    }
                }
                else
                {
                    note(0x1000u | request.function, 0u, nullptr);
                }
                using clock = std::chrono::steady_clock;
                if (clock::now() - m_censusT0 > std::chrono::seconds(10))
                {
                    m_censusT0 = clock::now();
                    std::string line = "[snd989:census]";
                    char part[40];
                    for (const auto &[id, e] : m_census)
                    {
                        std::snprintf(part, sizeof(part), " %x:%llu", id, static_cast<unsigned long long>(e.count));
                        line += part;
                    }
                    std::fprintf(stderr, "%s\n", line.c_str());
                }
            }
            struct CensusEntry
            {
                uint64_t count = 0u;
            };
            std::map<uint32_t, CensusEntry> m_census;
            std::chrono::steady_clock::time_point m_censusT0 = std::chrono::steady_clock::now();
            IopHost &m_host;
            Snd989Bindings m_bindings;
            std::array<uint32_t, 2> m_sids;
            mutable std::mutex m_mutex;
            uint64_t m_callCount = 0u;
            uint32_t m_bankLoads = 0u;
            uint32_t m_vagStreams = 0u;
        };
    }

    std::unique_ptr<IopService> createSnd989StubService(IopHost &host, Snd989Bindings bindings)
    {
        if (bindings.serviceName.empty() || bindings.sid == 0u)
        {
            throw std::invalid_argument("invalid 989SND stub bindings");
        }
        return std::make_unique<Snd989StubService>(host, std::move(bindings));
    }
}

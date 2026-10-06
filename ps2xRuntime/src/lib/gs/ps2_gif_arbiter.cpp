#include "runtime/gs/ps2_gif_arbiter.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>

// Path of the packet currently being handed to the GS (-1 = not via arbiter).
// Read by gs_frontend.cpp's [gs:pkt] trace so a mis-parsed packet can be
// attributed to PATH1/2/3.
int g_gifArbiterCurrentPath = -1;

GifArbiter::GifArbiter(ProcessPacketFn processFn)
    : m_processFn(std::move(processFn))
{
}

bool GifArbiter::isImagePacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes < 16u)
        return false;

    uint64_t tagLo = 0;
    std::memcpy(&tagLo, data, sizeof(tagLo));
    const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
    return flg == 2u;
}

void GifArbiter::submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl)
{
    if (!data || sizeBytes < 16 || !m_processFn)
        return;

    GifArbiterPacket pkt;
    pkt.pathId = pathId;
    pkt.path2DirectHl = (pathId == GifPathId::Path2) && path2DirectHl;
    pkt.path3Image = (pathId == GifPathId::Path3) && isImagePacket(data, sizeBytes);
    pkt.data.resize(sizeBytes);
    std::memcpy(pkt.data.data(), data, sizeBytes);
    m_queue.push_back(std::move(pkt));
}

void GifArbiter::drain()
{
    if (!m_processFn)
        return;

    std::stable_sort(m_queue.begin(), m_queue.end(),
                     [](const GifArbiterPacket &a, const GifArbiterPacket &b)
                     {
                         // DIRECTHL cannot preempt PATH3 IMAGE transfers.
                         if (a.path2DirectHl != b.path2DirectHl || a.path3Image != b.path3Image)
                         {
                             if (a.path3Image && b.path2DirectHl)
                                 return true;
                             if (a.path2DirectHl && b.path3Image)
                                 return false;
                         }
                         return pathPriority(a.pathId) < pathPriority(b.pathId);
                     });

    for (size_t i = 0; i < m_queue.size(); ++i)
    {
        auto &pkt = m_queue[i];
        if (!pkt.data.empty())
        {
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
            {
                static std::atomic<uint32_t> s_arb{0u};
                const uint32_t n = s_arb.fetch_add(1u, std::memory_order_relaxed);
                if (n < 96u)
                {
                    uint64_t q0 = 0u, q0hi = 0u;
                    std::memcpy(&q0, pkt.data.data(), 8u);
                    std::memcpy(&q0hi, pkt.data.data() + 8u, 8u);
                    std::fprintf(stderr, "[gifarb] #%u drain=%zu/%zu path=%d img3=%d dhl=%d bytes=%zu q0=%016llx.%016llx\n",
                                 n, i, m_queue.size(), static_cast<int>(pkt.pathId),
                                 pkt.path3Image ? 1 : 0, pkt.path2DirectHl ? 1 : 0, pkt.data.size(),
                                 (unsigned long long)q0hi, (unsigned long long)q0);
                }
            }
#endif
            g_gifArbiterCurrentPath = static_cast<int>(pkt.pathId);
            m_processFn(pkt.data.data(), static_cast<uint32_t>(pkt.data.size()));
            g_gifArbiterCurrentPath = -1;
        }
    }
    m_queue.clear();
}

uint8_t GifArbiter::pathPriority(GifPathId id)
{
    return static_cast<uint8_t>(id);
}

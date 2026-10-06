// No observer runs during flag retirement. Keep the FIFO cursor and visible
// registers local while draining the ready prefix, then publish each once.
// Entry validity and all inactive metadata remain identical to the old loop.
namespace wotmVuFlags {
template<class Pipeline, class Order, class State>
inline void retireReady(Pipeline &pipeline, const Order &order, uint32_t &head,
                        uint32_t &count, uint32_t &mask, State &state, uint64_t cycle)
{
    if (!count || pipeline[order[head]].readyCycle>cycle) return;
    uint32_t nextHead=head, remaining=count, live=mask;
    uint32_t mac=state.mac, status=state.status, clip=state.clip;
    do {
        const uint32_t slot=order[nextHead];
        auto &entry=pipeline[slot];
        if (entry.readyCycle>cycle) break;
        nextHead=(nextHead+1u)%order.size();
        --remaining;
        if (entry.writesMac) mac=entry.mac;
        if (entry.writesStatus) {
            const uint32_t current=entry.status&0xFu;
            status=(status&0xFF0u)|current|((current|entry.extraSticky)<<6);
        }
        if (entry.writesSticky) status=(status&0x03Fu)|(entry.status&0xFC0u);
        if (entry.writesClip) clip=entry.clip;
        entry.valid=false;
        live&=~(1u<<slot);
    } while (remaining);
    head=nextHead;count=remaining;mask=live;
    state.mac=mac;state.status=status;state.clip=clip;
}
}

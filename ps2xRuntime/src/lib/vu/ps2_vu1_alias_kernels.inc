// Store/load ordering for the verified fixed VU1 vertex loops. Addresses are
// wrapped, aligned byte offsets in the 16 KiB VU1 data memory.
namespace wotmVuAlias
{
inline bool shader1028(uint32_t destination, uint32_t material)
{
    // 10f8..1200: stores are destination+0/+16/+32. Loads after the first
    // store are constants q8/q18/q19, material (vi9+181), and destination
    // q5/q4/q6. Only q18/q19 follow the second store; none follow the third.
    // Relative offsets stay distinct after the 1024-qword wrap, including
    // the signed VI8 increment. Thus the complete hazard set is:
    // d=q8/q18/q19/material, or d+1=q18/q19 (i.e. d=q17/q18).
    return destination == 8u * 16u || destination == 17u * 16u ||
           destination == 18u * 16u || destination == 19u * 16u ||
           destination == material;
}

// 128..200: the sole load after a store is vi8+4. Earlier stores are vi8+0
// and vi8+1; later stores are vi8+2, after VI8 advances by three. They cannot
// overlap that load, even when qword addressing wraps. Other loads precede
// every store. The native loop still stages writes until the vertex succeeds.
inline bool shader40() { return false; }
}

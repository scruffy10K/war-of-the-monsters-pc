// SCUS_971.97 srand (23f988) / rand (23f998): newlib's guest reent
// pointer is at 6e58a4, with its 32-bit seed at +58. The game consumes
// 23 random mantissa bits in mathfRandf; Windows' 15-bit rand collapses
// pedestrian scatter, and every other caller of that float helper.
static void wotmSrand(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
{
    const uint32_t reent = wotmRead32(rdram, 0x006e58a4u);
    wotmWrite32At(rdram, reent + 0x58u, getRegU32(ctx, 4));
    wotmSetReg(ctx, 2, 0x006e0000);
    wotmSetReg(ctx, 3, static_cast<int32_t>(reent));
    ctx->pc = getRegU32(ctx, 31);
}

static void wotmRand(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
{
    const uint32_t reent = wotmRead32(rdram, 0x006e58a4u);
    const uint32_t seed = wotmRead32(rdram, reent + 0x58u) * 1103515245u + 12345u;
    wotmWrite32At(rdram, reent + 0x58u, seed);
    wotmSetReg(ctx, 2, static_cast<int32_t>(seed & 0x7fffffffu));
    wotmSetReg(ctx, 3, static_cast<int32_t>(seed));
    wotmSetReg(ctx, 4, 1103515245);
    wotmSetReg(ctx, 5, static_cast<int32_t>(reent));
    ctx->pc = getRegU32(ctx, 31);
}

// s105: original control flow around native object matrix/clip kernels.
// DeadFlags removes only immediate MAC updates overwritten by the next pair;
// neither validated image has STATUS readers. Preserve every boundary MAC.
extern Vu1NativeStep (*g_ps2xVu1ObjectKernel[4])(VU1Interpreter &, Vu1NativeCtx &);
extern uint64_t g_ps2xVu1ObjectHits[4],g_ps2xVu1ObjectFallbacks[3];
namespace { struct ObjectKernelReport {
    ~ObjectKernelReport() {if(std::getenv("PS2X_VU1_OBJECT_PROFILE"))
        std::fprintf(stderr,"[vu1:object-total] fast=%llu/%llu/%llu clip=%llu fallback=%llu/%llu/%llu\n",g_ps2xVu1ObjectHits[0],g_ps2xVu1ObjectHits[1],g_ps2xVu1ObjectHits[2],g_ps2xVu1ObjectHits[3],g_ps2xVu1ObjectFallbacks[0],g_ps2xVu1ObjectFallbacks[1],g_ps2xVu1ObjectFallbacks[2]);}
} s_objectKernelReport; }
template<bool DeadFlags>
static Vu1NativeStep objectShaderBlock(VU1Interpreter &v, Vu1NativeCtx &c, Vu1NativeProgram reference) {
    static const bool profile=std::getenv("PS2X_VU1_OBJECT_PROFILE")!=nullptr;
    static uint64_t calls=0;
    if(profile && (++calls==1 || calls%10000==0))
        std::fprintf(stderr,"[vu1:object-native] calls=%llu fast=%llu/%llu/%llu clip=%llu fallback=%llu/%llu/%llu\n",calls,g_ps2xVu1ObjectHits[0],g_ps2xVu1ObjectHits[1],g_ps2xVu1ObjectHits[2],g_ps2xVu1ObjectHits[3],g_ps2xVu1ObjectFallbacks[0],g_ps2xVu1ObjectFallbacks[1],g_ps2xVu1ObjectFallbacks[2]);
    for (;;) {
        if (Vu1NativeAccess::stopRequested(v) || Vu1NativeAccess::cycle(v)>=c.budgetEnd) return Vu1NativeStep::Stop;
        Vu1NativeStep r;
        switch (Vu1NativeAccess::pc(v)) {
#include <objectShaderCases.inc>
        default: return reference(v,c);
        }
    }
}

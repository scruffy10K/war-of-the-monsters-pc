// Optional per-player orbit camera. Guest addresses are from SCUS_971.97:
// Cameras::Update calls normalUpdate before checkCollision (11c818).
// _cs is Z-up; position +10, RPH +60, matrix +20. Keep retail collision,
// lock-on, knockback, victory and cinematic cameras in charge of their modes.
static PS2Runtime::RecompiledFunction s_cameraNormal, s_cameraPad, s_cameraMatrix, s_cameraPinning;
static PS2Runtime::RecompiledFunction s_cameraInterpret, s_cameraToggle;
struct WotmOrbit {
    uint32_t monster = 0, cs = 0;
    uint64_t frame = 0;
    float yaw = 0, pitch = 0, distance = 0, height = 0;
    bool active = false;
};
static WotmOrbit s_orbits[2];
static float s_cameraXs[2]{}, s_cameraYs[2]{};
static float cameraFloat(const uint8_t *ram, uint32_t address)
{ float f; std::memcpy(&f, ram + address, 4); return f; }
static void cameraFloat(uint8_t *ram, uint32_t address, float value)
{ std::memcpy(ram + address, &value, 4); }
static bool cameraGameplay(uint8_t *ram, R5900Context *ctx)
{
    const uint32_t gp=getRegU32(ctx,28);
    return g_ps2xWotmPhase.load(std::memory_order_relaxed) == 2u &&
        gp>=0x7ff0u && gp<PS2_RAM_SIZE && (wotmRead32(ram,gp-0x6e10u)==1u || wotmRead32(ram,gp-0x6e10u)==2u) &&
        // doCinema sets this flag at 100978; endCinema clears it at 100bf4.
        wotmRead32(ram,gp-0x7ff0u)==0u;
}
// Camera objects 0/1 follow the corresponding players; object 2 is the
// game's shared/cinematic view and must not borrow either player's orbit.
static unsigned cameraViews(uint8_t *ram, R5900Context *ctx)
{ return cameraGameplay(ram,ctx) ? wotmRead32(ram,getRegU32(ctx,28)-0x6e10u) : 0u; }
static void cameraToggle(uint8_t *ram, R5900Context *ctx, PS2Runtime *runtime)
{
    const unsigned player=getRegU32(ctx,4);
    if (player<2u && player<cameraViews(ram,ctx)) {
        // Retail TogglePOV switches 0 <-> 3 (staticUpdate). That permanently
        // bypasses normalUpdate, so modern orbit never receives stick input.
        // In modern mode this action recenters using the next retail update.
        s_orbits[player]={};
        ctx->pc=getRegU32(ctx,31);
        std::fprintf(stderr,"[camera] Player %u camera recentered (modern mode).\n",player+1);
        return;
    }
    s_cameraToggle(ram,ctx,runtime);
}
static void cameraReadPad(uint8_t *ram, R5900Context *ctx, PS2Runtime *runtime)
{
    const uint32_t pad = getRegU32(ctx, 4), index = getRegU32(ctx, 5), caller = getRegU32(ctx,31);
    struct Scope { Scope(){++t_ps2xEeCheckpointDeferralDepth;} ~Scope(){--t_ps2xEeCheckpointDeferralDepth;} } scope;
    s_cameraPad(ram, ctx, runtime);
    if (ctx->pc != caller || t_ps2xEeUnwinding || index >= 2u || pad > PS2_RAM_SIZE - 0x18u) return;
    s_cameraXs[index] = (int(ram[pad+0x11]) - int(ram[pad+0x10])) / 254.0f;
    s_cameraYs[index] = (int(ram[pad+0x13]) - int(ram[pad+0x12])) / 254.0f;
    // The game already applies its analog dead zone in inputFixAnalogValue.
    // Reserve this stick for the camera in modern mode for each active player.
    if (index<cameraViews(ram,ctx)) std::memset(ram+pad+0x10, 0, 4);
    // Record mode transitions even without verbose tracing, so a user report
    // can distinguish a special camera from lost stick input or lock-on.
    static uint32_t lastMode[2]={~0u,~0u};
    const uint32_t camera=0x007c01f0u+index*0xeb0u, pov=wotmRead32(ram,camera+0xcf8u);
    const unsigned views=cameraViews(ram,ctx);
    const uint32_t mode=(views<<16)|(pov&0xffffu);
    if (mode!=lastMode[index]) {
        lastMode[index]=mode;
        std::fprintf(stderr,"[camera:mode] player=%u views=%u pov=%u stick=%.2f,%.2f\n",index+1,views,pov,s_cameraXs[index],s_cameraYs[index]);
    }
    static const bool moveTest=std::getenv("PS2X_CAMERA_MOVE_TEST")!=nullptr;
    if (moveTest && index==0 && cameraGameplay(ram,ctx)) {
        std::memset(ram+pad+0xc,0,4); ram[pad+0xe]=254;
        const uint32_t analog=1; std::memcpy(ram+pad+0x14,&analog,4);
    }
}
static void cameraInterpret(uint8_t *ram, R5900Context *ctx, PS2Runtime *runtime)
{
    const uint32_t flags=getRegU32(ctx,4), pad=getRegU32(ctx,5);
    if (!cameraGameplay(ram,ctx) || !flags || flags>PS2_RAM_SIZE-0x1814u ||
        !pad || pad>PS2_RAM_SIZE-0x1cu) { s_cameraInterpret(ram,ctx,runtime); return; }
    const uint32_t monster=wotmRead32(ram,flags+0x1810u);
    if (!monster || monster>PS2_RAM_SIZE-0x8000u) { s_cameraInterpret(ram,ctx,runtime); return; }
    const unsigned player=wotmRead32(ram,monster+0x6cd8u);
    if (player>=2u || player>=cameraViews(ram,ctx)) { s_cameraInterpret(ram,ctx,runtime); return; }
    const uint32_t camera=0x007c01f0u+player*0xeb0u, pov=wotmRead32(ram,camera+0xcf8u);
    // Only the player's ordinary camera. Cinematic/knockback/grapple views and
    // other monsters must retain their existing control interpretation.
    if (!monster || monster>PS2_RAM_SIZE-0x8000u || monster!=wotmRead32(ram,camera+0xcf4u) ||
        pov>=10u || wotmRead32(ram,0x006e8a2cu+pov*8u)!=0x0011f630u || ram[pad] || ram[pad+3]) {
        s_cameraInterpret(ram,ctx,runtime); return;
    }
    struct Scope { Scope(){++t_ps2xEeCheckpointDeferralDepth;} ~Scope(){--t_ps2xEeCheckpointDeferralDepth;} } scope;
    const R5900Context saved=*ctx;
    wotmSetReg(ctx,4,monster); ctx->pc=0x00163918u;
    s_cameraPinning(ram,ctx,runtime);
    const bool locked=getRegU32(ctx,2)!=0;
    *ctx=saved;
    const uint32_t target=wotmRead32(ram,monster+0x6c04u);
    if (locked || (target && target<PS2_RAM_SIZE-6u && (ram[target+4]&2u))) {
        s_cameraInterpret(ram,ctx,runtime); return;
    }
    // interpretInputs chooses CR for mapping0 and SR for mapping1. SR computes
    // movement from the current camera each call (1796b0..179760), retaining
    // retail analog speed, combat actions, acceleration and character turning.
    // Set both current/requested mapping to avoid the retail delayed switch
    // while a direction is held; never persist this override in user settings.
    struct MappingScope {
        uint8_t *ram; uint32_t address, values[2];
        MappingScope(uint8_t *r,uint32_t a):ram(r),address(a) {
            std::memcpy(values,ram+address,8);
            const uint32_t screenRelative[2]={1,1}; std::memcpy(ram+address,screenRelative,8);
        }
        ~MappingScope(){std::memcpy(ram+address,values,8);}
    } mapping(ram,flags+0x16e4u);
    s_cameraInterpret(ram,ctx,runtime);
    static const bool trace=std::getenv("PS2X_CAMERA_TRACE")!=nullptr;
    const auto frame=g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);
    if (trace && frame%60u==0) std::fprintf(stderr,"[camera:steering] frame=%llu mapping=%u/%u stick=%d,%d world=%.3f,%.3f\n",
        static_cast<unsigned long long>(frame),mapping.values[0],mapping.values[1],
        int(ram[pad+0xd])-int(ram[pad+0xc]),int(ram[pad+0xe])-int(ram[pad+0xf]),
        cameraFloat(ram,flags+0x17e0u),cameraFloat(ram,flags+0x17e4u));
}
static void cameraNormal(uint8_t *ram, R5900Context *ctx, PS2Runtime *runtime)
{
    const uint32_t camera = getRegU32(ctx,4), caller = getRegU32(ctx,31);
    static const bool trace=std::getenv("PS2X_CAMERA_TRACE") != nullptr;
    static unsigned calls=0;
    const bool report=trace && (++calls%120u==1u);
    struct Scope { Scope(){++t_ps2xEeCheckpointDeferralDepth;} ~Scope(){--t_ps2xEeCheckpointDeferralDepth;} } scope;
    s_cameraNormal(ram, ctx, runtime);
    if (ctx->pc != caller || t_ps2xEeUnwinding) return;
    if (report) std::fprintf(stderr,"[camera:state] camera=%08x phase=%u count=%u frame=%llu\n",camera,g_ps2xWotmPhase.load(),wotmRead32(ram,0x006f8e30),static_cast<unsigned long long>(g_ps2xWotmCompletedFrames.load()));
    if (camera<0x007c01f0u || (camera-0x007c01f0u)%0xeb0u) return;
    const unsigned player=(camera-0x007c01f0u)/0xeb0u;
    if (player>=2u) return;
    auto &s_orbit=s_orbits[player];
    if (player>=cameraViews(ram,ctx)) { s_orbit = {}; return; }
    const uint32_t monster = wotmRead32(ram,camera+0xcf4), cs = wotmRead32(ram,camera+0xcf0);
    if (!monster || monster > PS2_RAM_SIZE-0x8000u || !cs || cs > PS2_RAM_SIZE-0x70u) { s_orbit={}; return; }
    const uint32_t follow = wotmRead32(ram,monster+0xc);
    if (!follow || follow > PS2_RAM_SIZE-0x70u) { s_orbit={}; return; }
    if (report && std::getenv("PS2X_MOD_DIAG") && wotmRead32(ram,monster+0x14u)==0x1a0u)
        std::fprintf(stderr,"[mods:position] frame=%llu monster=%.2f,%.2f,%.2f camera=%.2f,%.2f,%.2f\n",
            static_cast<unsigned long long>(g_ps2xWotmCompletedFrames.load()),
            cameraFloat(ram,follow+0x10),cameraFloat(ram,follow+0x14),cameraFloat(ram,follow+0x18),
            cameraFloat(ram,cs+0x10),cameraFloat(ram,cs+0x14),cameraFloat(ram,cs+0x18));
    if (report && std::getenv("PS2X_MOD_DIAG") && wotmRead32(ram,monster+0x14u)==0x1a0u) {
        const uint32_t mesh=wotmRead32(ram,monster+0x1a3cu);
        if(mesh && mesh<PS2_RAM_SIZE-0x50u)
            std::fprintf(stderr,"[mods:hier-frame] node=%08x diag=%.3f,%.3f,%.3f\n",mesh,
                cameraFloat(ram,mesh+0x20),cameraFloat(ram,mesh+0x34),cameraFloat(ram,mesh+0x48));
    }
    // Query the actual held/toggled lock action, including lock with no target.
    const R5900Context saved = *ctx;
    wotmSetReg(ctx,4,monster); ctx->pc=0x00163918u;
    s_cameraPinning(ram,ctx,runtime);
    const bool locked = getRegU32(ctx,2) != 0;
    *ctx = saved;
    if (report) std::fprintf(stderr,"[camera:lock] monster=%08x cs=%08x held=%u target=%08x\n",monster,cs,locked,wotmRead32(ram,monster+0x6c04));
    if (locked) { s_orbit={}; return; }
    const uint32_t target=wotmRead32(ram,monster+0x6c04);
    if (target && target < PS2_RAM_SIZE-6u && (ram[target+4]&2u)) { s_orbit={}; return; }
    const uint64_t frame=g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);
    if (s_orbit.monster != monster || s_orbit.cs != cs || frame < s_orbit.frame || frame > s_orbit.frame+2u) s_orbit={};
    float x=std::clamp(s_cameraXs[player],-1.0f,1.0f), y=std::clamp(s_cameraYs[player],-1.0f,1.0f);
    static const bool test=std::getenv("PS2X_CAMERA_TEST") != nullptr;
    if (test) { const auto t=frame%900u; x=t>=90 && t<210 ? 0.7f : t>=330 && t<450 ? -0.7f : 0; y=t>=500 && t<550 ? 0.5f : t>=600 && t<650 ? -0.5f : 0; }
    const float px=cameraFloat(ram,follow+0x10), py=cameraFloat(ram,follow+0x14), pz=cameraFloat(ram,follow+0x18);
    if (!s_orbit.active) {
        if (std::fabs(x)<0.01f && std::fabs(y)<0.01f) return;
        const float dx=px-cameraFloat(ram,cs+0x10), dy=py-cameraFloat(ram,cs+0x14);
        const float horizontal=std::hypot(dx,dy), pitch=cameraFloat(ram,cs+0x60);
        if (!std::isfinite(horizontal) || horizontal<10 || horizontal>2000 || !std::isfinite(pitch) || std::fabs(pitch)>1.3f) return;
        s_orbit={monster,cs,frame,std::atan2(dx,dy),pitch,horizontal/std::cos(pitch),
            cameraFloat(ram,cs+0x18)+std::tan(pitch)*horizontal-pz,true};
    }
    // One camera update per simulation frame, independent of presentation rate.
    if (frame != s_orbit.frame) {
        s_orbit.yaw=std::remainder(s_orbit.yaw+x*0.035f,6.28318530718f);
        s_orbit.pitch=std::clamp(s_orbit.pitch-y*0.025f,-0.85f,0.65f);
    }
    s_orbit.frame=frame;
    const float horizontal=s_orbit.distance*std::cos(s_orbit.pitch);
    const float position[3]={px-std::sin(s_orbit.yaw)*horizontal,
        py-std::cos(s_orbit.yaw)*horizontal,pz+s_orbit.height-std::sin(s_orbit.pitch)*s_orbit.distance};
    for (unsigned i=0;i<3;++i) { cameraFloat(ram,cs+0x10+i*4,position[i]); cameraFloat(ram,camera+0xd20+i*4,position[i]); }
    cameraFloat(ram,cs+0x60,s_orbit.pitch); cameraFloat(ram,cs+0x64,0); cameraFloat(ram,cs+0x68,s_orbit.yaw);
    wotmSetReg(ctx,4,cs+0x20); wotmSetReg(ctx,5,cs+0x60); ctx->pc=0x0020d380u;
    s_cameraMatrix(ram,ctx,runtime);
    *ctx=saved;
    if (trace && frame%60u==0) std::fprintf(stderr,"[camera:orbit] frame=%llu stick=%.3f,%.3f yaw=%.3f pitch=%.3f distance=%.2f pos=%.2f,%.2f,%.2f\n",static_cast<unsigned long long>(frame),x,y,s_orbit.yaw,s_orbit.pitch,s_orbit.distance,position[0],position[1],position[2]);
}
static void wotmInstallModernCamera(PS2Runtime &runtime)
{
    const char *mode=std::getenv("PS2X_CAMERA");
    if (!mode || std::strcmp(mode,"modern")!=0) return;
    s_cameraNormal=runtime.lookupFunction(0x0011f630u); s_cameraPad=runtime.lookupFunction(0x0013a020u);
    s_cameraMatrix=runtime.lookupFunction(0x0020d380u); s_cameraPinning=runtime.lookupFunction(0x00163918u);
    s_cameraInterpret=runtime.lookupFunction(0x00177d20u);
    s_cameraToggle=runtime.lookupFunction(0x0011c070u);
    if (!s_cameraToggle || !s_cameraNormal || !s_cameraPad || !s_cameraMatrix || !s_cameraPinning || !s_cameraInterpret) return;
    for (unsigned i=0;i<2;++i) { s_orbits[i]={}; s_cameraXs[i]=s_cameraYs[i]=0; }
    runtime.replaceFunction(0x0011f630u,&cameraNormal); runtime.replaceFunction(0x0013a020u,&cameraReadPad);
    runtime.replaceFunction(0x00177d20u,&cameraInterpret);
    runtime.replaceFunction(0x0011c070u,&cameraToggle);
    std::fprintf(stderr,"[camera] Modern per-player orbit enabled; retail target lock and collision retained.\n");
}

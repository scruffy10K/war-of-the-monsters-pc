// Keep the extra roster selection outside the retail character table, and use
// it only while loading the selected character's assets and player object.
static PS2Runtime::RecompiledFunction s_modLoadTexture, s_modLoadResTexture, s_modLoadMonsters, s_modInitPlayers, s_modSetAIMonster, s_modScreenSelect, s_modScreenSelect2, s_modCsList, s_modHpCsList;
static const auto s_modBoot=std::chrono::steady_clock::now();
static unsigned s_modPlayer1=0;
static uint32_t s_modPlayerHierarchy=0;
static uint32_t s_modPlayerCs=0;
static uint32_t s_modPlayerObject=0;
#include "wotm_roster_native.inc"
static bool wotmModAllowed(unsigned id) {
    return id==1||id==2||id==3||id==4||id==5||id==7||id==8||id==9||id==10||id==11||id==13;
}
static void wotmModScreenSelect(uint8_t*r,R5900Context*c,PS2Runtime*t) {
    g_ps2xModSelectSeen.store(true,std::memory_order_relaxed);
    const auto now=std::chrono::steady_clock::now().time_since_epoch();
    const uint64_t ns=static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    const uint64_t previous=g_ps2xRosterHeartbeatNs.exchange(ns,std::memory_order_relaxed);
    if(g_ps2xRosterEnabled.load(std::memory_order_relaxed) && previous && ns>previous+1000000000ull) {
        g_ps2xRosterChosenType.store(0,std::memory_order_relaxed);
        g_ps2xRosterPage.store(false,std::memory_order_relaxed);
        s_nativeRosterPage=false;
    }
    static unsigned calls=0;
    const unsigned n=++calls;
    if(std::getenv("PS2X_MOD_DIAG") && (n<=12 || n%300u==0)) {
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-s_modBoot).count();
        const uint32_t shell=wotmRead32(r,0x6f8200u);
        const uint32_t gp=getRegU32(c,28);
        std::fprintf(stderr,"[mods:select] t=%.2f call=%u pc=%08x shell=%08x gp=%08x cursor=%08x row=%08x count=%u,%u,%u,%u\n",
            seconds,n,c->pc,shell,gp,
            gp>=0x9000u && gp<PS2_RAM_SIZE?wotmRead32(r,gp-0x79d8u):0u,
            gp>=0x9000u && gp<PS2_RAM_SIZE?wotmRead32(r,gp-0x79e0u):0u,
            wotmRead32(r,0x2b3410u),wotmRead32(r,0x2b3414u),
            wotmRead32(r,0x2b3418u),wotmRead32(r,0x2b341cu));
    }
    s_nativeRosterTwoPlayer=false;
    if(wotmNativeRosterEnabled())wotmNativeRosterScreen(r,c,t,s_modScreenSelect);
    else s_modScreenSelect(r,c,t);
}
static void wotmModScreenSelect2(uint8_t*r,R5900Context*c,PS2Runtime*t) {
    s_nativeRosterTwoPlayer=true;
    wotmNativeRosterScreen(r,c,t,s_modScreenSelect2);
}
static void wotmModCall(uint8_t *ram,R5900Context *ctx,PS2Runtime *runtime,PS2Runtime::RecompiledFunction original)
{
    if(g_ps2xRosterEnabled.load(std::memory_order_relaxed) && wotmNativeRosterEnabled()) {
        const uint32_t shell=getRegU32(ctx,4);
        if(std::getenv("PS2X_MOD_DIAG") && shell && shell<PS2_RAM_SIZE-0x2c20u)
            std::fprintf(stderr,"[mods:native-load] pc=%08x p1=%u p2=%u costume=%u/%u\n",ctx->pc,
                wotmRead32(ram,shell+0x2938u)>>5,wotmRead32(ram,shell+0x293cu)>>5,
                wotmRead32(ram,shell+0x2be0u),wotmRead32(ram,shell+0x2be4u));
        original(ram,ctx,runtime);return;
    }
    const uint32_t shell=getRegU32(ctx,4);
    const unsigned selected=g_ps2xRosterEnabled.load(std::memory_order_relaxed)
        ? static_cast<unsigned>(g_ps2xRosterChosenType.load(std::memory_order_relaxed)) : s_modPlayer1;
    const int slot=g_ps2xRosterSlot.load(std::memory_order_relaxed);
    const uint32_t selectedCostume=g_ps2xRosterEnabled.load(std::memory_order_relaxed) && slot>=0 && slot<10
        ? g_ps2xRosterCostumes[slot] : 0;
    if(std::getenv("PS2X_MOD_DIAG")) {
        static unsigned loadCalls=0;
        if(loadCalls++<24 && shell && shell<PS2_RAM_SIZE-0x2c20u)
            std::fprintf(stderr,"[mods:load] pc=%08x shell=%08x selected=%u costume=%u count=%u mode=%u original=%08x/%08x wardrobe=%u/%u filetype=%08x\n",
                ctx->pc,shell,selected,selectedCostume,wotmRead32(ram,shell+0x2bb4u),wotmRead32(ram,shell+0x2c10u),
                wotmRead32(ram,shell+0x2938u),wotmRead32(ram,shell+0x293cu),
                wotmRead32(ram,shell+0x2be0u),wotmRead32(ram,shell+0x2be4u),wotmRead32(ram,0x6f81d0u));
    }
    if (!selected || !shell || shell>PS2_RAM_SIZE-0x2c20u || wotmRead32(ram,shell+0x2bb4u)==0u) {
        original(ram,ctx,runtime);return;
    }
    struct SelectionScope {
        uint8_t *ram;uint32_t shell,selection,costume;
        SelectionScope(uint8_t *r,uint32_t s,unsigned character,uint32_t variant):ram(r),shell(s),selection(wotmRead32(r,s+0x2938)),costume(wotmRead32(r,s+0x2be0)) {
            ++t_ps2xEeCheckpointDeferralDepth;
            const uint32_t packed=character<<5;
            std::memcpy(ram+shell+0x2938,&packed,4);std::memcpy(ram+shell+0x2be0,&variant,4);
        }
        ~SelectionScope() {
            std::memcpy(ram+shell+0x2938,&selection,4);std::memcpy(ram+shell+0x2be0,&costume,4);
            --t_ps2xEeCheckpointDeferralDepth;
        }
    } scope(ram,shell,selected,selectedCostume);
    original(ram,ctx,runtime);
}
static void wotmModTexture(uint8_t*r,R5900Context*c,PS2Runtime*t){wotmModCall(r,c,t,s_modLoadTexture);}
static void wotmModResTexture(uint8_t*r,R5900Context*c,PS2Runtime*t){wotmModCall(r,c,t,s_modLoadResTexture);}
static void wotmModMonsters(uint8_t*r,R5900Context*c,PS2Runtime*t){wotmModCall(r,c,t,s_modLoadMonsters);}
static void wotmModPlayers(uint8_t*r,R5900Context*c,PS2Runtime*t){
    wotmModCall(r,c,t,s_modInitPlayers);
    const uint32_t game=wotmRead32(r,0x6f81f8);
    const uint32_t player=game && game<PS2_RAM_SIZE-0x120400u?wotmRead32(r,game+0x120380u):0;
    if (player && player<PS2_RAM_SIZE-0x8000u) {
        s_modPlayerObject=player;
        s_modPlayerCs=player+0x1a3cu;
        s_modPlayerHierarchy=wotmRead32(r,s_modPlayerCs);
        if (!std::getenv("PS2X_MOD_DIAG")) return;
        const uint32_t node=wotmRead32(r,player+0xcu);
        if (node && node<PS2_RAM_SIZE-0x40u) {
            float scale[3];std::memcpy(scale,r+node+0x20,sizeof(scale));
            std::fprintf(stderr,"[mods:scale] player=%08x cs=%08x values=%.3f,%.3f,%.3f\n",player,node,
                scale[0],scale[1],scale[2]);
        }
        const uint32_t mesh=wotmRead32(r,player+0x1a3cu);
        if (mesh && mesh<PS2_RAM_SIZE-0x80u) {
            std::fprintf(stderr,"[mods:hier] node=%08x ",mesh);
            for (unsigned i=0;i<0x50u;i+=4u) std::fprintf(stderr,"%02x:%08x ",i,wotmRead32(r,mesh+i));
            std::fprintf(stderr,"\n");
        }
    }
}
// Wrap the character lists, not the entire scene (whose BSP/packet work can
// return to the dispatcher before reaching the character lists). Every list
// continuation is registered below. Nodes outlive any in-flight traversal.
static uint32_t s_modScaleNodes=0;
static void wotmModScaledList(uint8_t*r,R5900Context*c,PS2Runtime*t,
                             PS2Runtime::RecompiledFunction original) {
    struct ScopedModels {
        uint8_t* ram;
        uint32_t addresses[2]{}, roots[2]{};
        explicit ScopedModels(uint8_t* r):ram(r) {++t_ps2xEeCheckpointDeferralDepth;}
        ~ScopedModels() {
            for(unsigned p=0;p<2;++p)if(addresses[p])std::memcpy(ram+addresses[p],&roots[p],4);
            --t_ps2xEeCheckpointDeferralDepth;
        }
    } models(r);
    const uint32_t game=wotmRead32(r,0x6f81f8u);
    if(s_modScaleNodes && game && game<PS2_RAM_SIZE-0x120400u)for(unsigned p=0;p<2;++p) {
        const uint32_t player=wotmRead32(r,game+0x120380u+p*4u);
        if(!player || player>=PS2_RAM_SIZE-0x8000u || wotmRead32(r,player+0x14u)!=0x1a0u ||
           wotmRead32(r,player+0x18u)!=1u || wotmRead32(r,player+0x24u)!=p)continue;
        const uint32_t cs=wotmRead32(r,player+0xcu);
        if(!cs || cs>=PS2_RAM_SIZE-0xb0u || (p && cs==models.addresses[0]))continue;
        const uint32_t root=wotmRead32(r,cs);
        const uint32_t node=s_modScaleNodes+p*64u;
        if(!root || root>=PS2_RAM_SIZE-48u || root==node)continue;
        std::memset(r+node,0,64);
        wotmWrite32At(r,node,0x1eu);
        wotmWrite32At(r,node+8u,1u);
        const float scale=0.30f;
        for(unsigned axis=0;axis<3;++axis)std::memcpy(r+node+0x20u+axis*4u,&scale,4);
        wotmWrite32At(r,node+0x2cu,root);
        models.addresses[p]=cs;models.roots[p]=root;
        wotmWrite32At(r,cs,node);
    }
    const uint32_t expectedReturn=getRegU32(c,31);
    original(r,c,t);
    static unsigned early=0;
    if(models.addresses[0] && c->pc!=expectedReturn && std::getenv("PS2X_MOD_DIAG") && early++<12)
        std::fprintf(stderr,"[mods:scale-list-resume] pc=%08x expected=%08x\n",c->pc,expectedReturn);
}
static void wotmModCsList(uint8_t*r,R5900Context*c,PS2Runtime*t) {
    wotmModScaledList(r,c,t,s_modCsList);
}
static void wotmModHpCsList(uint8_t*r,R5900Context*c,PS2Runtime*t) {
    wotmModScaledList(r,c,t,s_modHpCsList);
}
static void wotmModSetAIMonster(uint8_t*r,R5900Context*c,PS2Runtime*t) {
    const uint32_t game=getRegU32(c,4);
    const uint32_t player=game && game<PS2_RAM_SIZE-0x120400u?wotmRead32(r,game+0x120380u):0;
    const bool protect=player && player<PS2_RAM_SIZE-0x20u && wotmRead32(r,player+0x14)==0x1a0u && wotmRead32(r,player+0x18)==1u;
    if (protect) {
        // Retail SetAIMonster unconditionally deactivates every type-0x1a0 object.
        // Hide only the player's type from that AI-only sweep while it runs.
        struct TypeScope {
            uint8_t *ram;uint32_t address,type;
            TypeScope(uint8_t *r,uint32_t a):ram(r),address(a),type(wotmRead32(r,a)) {
                const uint32_t masked=0x1a1u;
                ++t_ps2xEeCheckpointDeferralDepth;
                std::memcpy(ram+address,&masked,4);
            }
            ~TypeScope(){std::memcpy(ram+address,&type,4);--t_ps2xEeCheckpointDeferralDepth;}
        } scope(r,player+0x14);
        s_modSetAIMonster(r,c,t);
    } else s_modSetAIMonster(r,c,t);
}
static void wotmInstallMods(PS2Runtime &runtime)
{
    const char *diagnostic=std::getenv("PS2X_MOD_DIAG");
    const bool diag=diagnostic && std::strcmp(diagnostic,"1")==0;
    const char *roster=std::getenv("PS2X_MOD_ROSTER_IDS");
    const bool rosterMode=roster && *roster;
    if (!rosterMode && !diag) return;
    unsigned long id=0;
    if(rosterMode) {
        const char *p=roster;
        for(unsigned slot=0;slot<10;++slot) {
            char *end=nullptr;const unsigned long value=std::strtoul(p,&end,10);
            if(end==p || value>13 || (value && !wotmModAllowed(static_cast<unsigned>(value))) ||
               (slot<9 && *end!=',') || (slot==9 && *end!='\0')) {
                std::fprintf(stderr,"[mods] Invalid roster slot list; roster disabled.\n");return;
            }
            g_ps2xRosterIds[slot]=static_cast<uint32_t>(value);
            p=slot<9?end+1:end;
        }
        const char *costumes=std::getenv("PS2X_MOD_ROSTER_COSTUMES");
        if(costumes && *costumes) {
            const char *q=costumes;
            for(unsigned slot=0;slot<10;++slot) {
                char *end=nullptr;const unsigned long value=std::strtoul(q,&end,10);
                if(end==q || value>3 || (slot<9 && *end!=',') || (slot==9 && *end!='\0')) {
                    std::fprintf(stderr,"[mods] Invalid roster costumes; roster disabled.\n");return;
                }
                g_ps2xRosterCostumes[slot]=static_cast<uint32_t>(value);
                q=slot<9?end+1:end;
            }
        }
        const char *names=std::getenv("PS2X_MOD_ROSTER_NAMES");
        if(names)for(unsigned slot=0;slot<10;++slot) {
            unsigned length=0;
            while(*names && *names!='|' && length<32) {
                const unsigned char ch=static_cast<unsigned char>(*names++);
                if((ch>='A'&&ch<='Z')||(ch>='0'&&ch<='9')||ch==' '||ch=='-')
                    g_ps2xRosterNames[slot][length++]=static_cast<char>(ch);
            }
            while(*names && *names!='|')++names;
            if(*names=='|')++names;
        }
    } else {
        const char *v=std::getenv("PS2X_MOD_PLAYER1");if(!v||!*v)return;
        char *end=nullptr;id=std::strtoul(v,&end,10);
        if (*end || !wotmModAllowed(static_cast<unsigned>(id))) {
            std::fprintf(stderr,"[mods] Invalid diagnostic character; mod disabled.\n");return;
        }
    }
    s_modLoadTexture=runtime.lookupFunction(0x1a9690);s_modLoadResTexture=runtime.lookupFunction(0x1a9230);s_modLoadMonsters=runtime.lookupFunction(0x1a9bf0);s_modInitPlayers=runtime.lookupFunction(0x1a6d60);
    s_modSetAIMonster=runtime.lookupFunction(0x137b10);
    s_modScreenSelect=runtime.lookupFunction(0x18f628);
    s_modScreenSelect2=runtime.lookupFunction(0x1964d0);
    if(!s_modLoadTexture||!s_modLoadResTexture||!s_modLoadMonsters||!s_modInitPlayers||!s_modSetAIMonster||!s_modScreenSelect)return;
    s_modPlayer1=static_cast<unsigned>(id);
    if(wotmNativeRosterEnabled()) {
        s_nativeRosterInput=runtime.lookupFunction(0x1a58a0u);
        if(s_nativeRosterInput)runtime.replaceFunction(0x1a58a0u,wotmNativeRosterGetInput);
    }
    runtime.replaceFunction(0x1a9690,wotmModTexture);runtime.replaceFunction(0x1a9230,wotmModResTexture);runtime.replaceFunction(0x1a9bf0,wotmModMonsters);runtime.replaceFunction(0x1a6d60,wotmModPlayers);
    runtime.replaceFunction(0x137b10,wotmModSetAIMonster);
    runtime.replaceFunction(0x18f628,wotmModScreenSelect);
    if(s_modScreenSelect2)runtime.replaceFunction(0x1964d0,wotmModScreenSelect2);
    if(rosterMode) {
        const uint32_t reserved=runtime.reserveAsyncCallbackStack(128u,16u);
        s_modScaleNodes=reserved ? reserved+16u-128u : 0u;
        s_modCsList=runtime.lookupFunction(0x202fd8u);
        s_modHpCsList=runtime.lookupFunction(0x2030c0u);
        // Cover the registered continuation PCs too; original code can yield
        // during packet processing and resume without entering the list again.
        for(uint32_t pc=0x202fd8u;pc<0x2031a8u;pc+=4u)if(runtime.hasFunction(pc)) {
            const auto fn=runtime.lookupFunction(pc);
            if(s_modCsList && fn==s_modCsList)runtime.replaceFunction(pc,wotmModCsList);
            else if(s_modHpCsList && fn==s_modHpCsList)runtime.replaceFunction(pc,wotmModHpCsList);
        }
    }
    if(rosterMode) {
        g_ps2xRosterEnabled.store(true,std::memory_order_release);
        std::fprintf(stderr,"[mods] Expanded roster page enabled with ten slots.\n");
    } else std::fprintf(stderr,"[mods] Diagnostic player-one character: type %u.\n",s_modPlayer1);
}




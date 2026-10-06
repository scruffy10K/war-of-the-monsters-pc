// Native character-strip pagination. All layout and selection borders are retail.
static bool wotmNativeRosterEnabled() { return true; }
static PS2Runtime::RecompiledFunction s_nativeRosterInput=nullptr;
static bool s_nativeRosterInScreen=false;
static bool s_nativeRosterPage=false;
static bool s_nativeRosterTwoPlayer=false;
static unsigned s_nativeRosterFocusPlayer=0;
static unsigned s_nativeRosterCostumeChoice[2][10]{};
static uint32_t s_nativeRosterFlags[10][4][2]{};
static uint32_t s_nativeRosterMapping[10]{};
static uint32_t s_nativeRosterSwitches[10]{};
static uint32_t s_nativeRosterPortraits[10]{};
static uint32_t s_nativeRosterPortraitTextures[10]{};
static uint32_t s_nativeRosterLockedTexture=0;
static uint8_t s_nativeRosterBindings[10][48]{};
static uint8_t s_nativeRosterLockedBinding[48]{};
static uint32_t s_nativeRosterShell=0;

static void wotmNativeRosterTable(uint8_t* r, bool modPage) {
    for(unsigned row=0;row<10;++row) {
        wotmWrite32At(r,0x2a1a40u+row*4u,modPage ? g_ps2xRosterIds[row]<<5 : s_nativeRosterMapping[row]);
        for(unsigned skin=0;skin<4;++skin) {
            const uint32_t a=0x2a1eb0u+row*0xb0u+skin*0x2cu;
            // Browsing a placeholder must terminate retail costume-search loops.
            // Confirm is separately blocked for empty rows in the input hook.
            wotmWrite32At(r,a,modPage ? 1u : s_nativeRosterFlags[row][skin][0]);
            wotmWrite32At(r,a+4,modPage ? 1u : s_nativeRosterFlags[row][skin][1]);
        }
    }
}
static void wotmNativeRosterArt(uint8_t* r, uint32_t shell, bool modPage) {
    const uint32_t last=wotmRead32(r,shell+0x18cu);
    // The strip group follows the final nine-way switch. Verify every child
    // against Shell's named switches before treating its first child as art.
    const uint32_t group=last+48u;
    if(!last || group>=PS2_RAM_SIZE-80u || (wotmRead32(r,group)&63u)!=1u ||
       (wotmRead32(r,group+8)&65535u)!=11u)return;
    if(shell!=s_nativeRosterShell || s_nativeRosterSwitches[9]!=last) {
        for(unsigned row=0;row<10;++row)
            if(wotmRead32(r,group+36u+row*4u)!=wotmRead32(r,shell+0x168u+row*4u))return;
        const uint32_t base=wotmRead32(r,group+32u);
        if(!base || base>=PS2_RAM_SIZE-80u || (wotmRead32(r,base)&63u)!=1u ||
           (wotmRead32(r,base+8)&65535u)!=12u)return;
        const uint32_t locked=wotmRead32(r,last+28u);
        if(!locked || locked>=PS2_RAM_SIZE-0xb0u || (wotmRead32(r,locked)&63u)!=0u)return;
        for(unsigned row=0;row<10;++row) {
            const uint32_t object=wotmRead32(r,base+36u+row*4u);
            if(!object || object>=PS2_RAM_SIZE-0xb0u || (wotmRead32(r,object)&63u)!=0u ||
               (wotmRead32(r,object+0x58u)>>16)!=1u)return;
        }
        s_nativeRosterShell=shell;
        s_nativeRosterLockedTexture=999u; // texmGetTextureAddrs preserves port-owned bindings.
        std::memcpy(s_nativeRosterLockedBinding,r+locked+0x70u,48);
        const uint64_t marker=0x3fffull | (1ull<<14) | (0x13ull<<20) | (6ull<<26) |
            (6ull<<30) | (1ull<<34) | (0x3fffull<<37) | (29ull<<56);
        std::memcpy(s_nativeRosterLockedBinding,&marker,8);
        std::memset(s_nativeRosterLockedBinding+32,0,16);
        for(unsigned row=0;row<10;++row) {
            s_nativeRosterSwitches[row]=wotmRead32(r,shell+0x168u+row*4u);
            s_nativeRosterPortraits[row]=wotmRead32(r,base+36u+row*4u);
            s_nativeRosterPortraitTextures[row]=wotmRead32(r,s_nativeRosterPortraits[row]+0x50u);
            std::memcpy(s_nativeRosterBindings[row],r+s_nativeRosterPortraits[row]+0x70u,48);
        }
        std::fprintf(stderr,"[mods:native-art] verified strip=%08x lockedTexture=%u\n",group,s_nativeRosterLockedTexture);
    }
    for(unsigned row=0;row<10;++row) {
        std::memcpy(r+s_nativeRosterPortraits[row]+0x70u,modPage?s_nativeRosterLockedBinding:s_nativeRosterBindings[row],48);
        wotmWrite32At(r,s_nativeRosterPortraits[row]+0x50u,
            modPage?s_nativeRosterLockedTexture:s_nativeRosterPortraitTextures[row]);
    }
    if(!modPage)for(unsigned row : {0u,9u})
        if(!s_nativeRosterFlags[row][0][1])r[s_nativeRosterSwitches[row]+8u]=4;
    if(modPage) {
        for(unsigned offset : {0x1dcu,0x1e0u}) {
            const uint32_t node=wotmRead32(r,shell+offset);
            if(node && node<PS2_RAM_SIZE-12u && (wotmRead32(r,node)&63u)==6u)r[node+8u]=0;
        }
        const uint32_t portrait=wotmRead32(r,shell+0x5ccu);
        if(portrait && portrait<PS2_RAM_SIZE-12u && (wotmRead32(r,portrait)&63u)==6u)
            r[portrait+8u]=0; // No unrelated retail monster portrait/name on page two.
    }
}
// Let a locked endpoint receive focus without changing its saved unlock flag.
// Its dummy costume exists only during this bounded menu call; confirm is blocked.
static void wotmNativeRosterBrowseTable(uint8_t* r) {
    wotmNativeRosterTable(r,s_nativeRosterPage);
    if(!s_nativeRosterPage)for(unsigned row : {0u,9u}) {
        if(!s_nativeRosterFlags[row][0][1]) {
            wotmWrite32At(r,0x2a1eb0u+row*0xb0u,1);
            wotmWrite32At(r,0x2a1eb4u+row*0xb0u,1);
        }
    }
}
static void wotmNativeRosterGetInput(uint8_t* r,R5900Context* c,PS2Runtime* t) {
    const uint32_t caller=getRegU32(c,31);
    s_nativeRosterInput(r,c,t);
    const bool playerTwo=caller==0x198a00u;
    if(!s_nativeRosterInScreen || (caller!=0x18fdc0u && caller!=0x19726cu && !playerTwo) ||
       c->pc!=caller || t_ps2xEeUnwinding)return;
    unsigned action=getRegU32(c,2)&0xffffu;
    const uint32_t screen=wotmRead32(r,0x6f8464u);
    if(screen>=22u)return;
    const uint32_t cursorAddress=0x2b3410u+screen*4u+(playerTwo?0x58u:0u);
    const uint32_t cursor=wotmRead32(r,cursorAddress);
    const uint32_t gp=getRegU32(c,28);
    if(gp<0x8000u || gp>=PS2_RAM_SIZE)return;
    const uint32_t costumeAddress=gp-(s_nativeRosterTwoPlayer?(playerTwo?0x7924u:0x7928u):0x79d8u);
    if(std::getenv("PS2X_NATIVE_ROSTER_TEST") && !playerTwo) {
        static unsigned ticks=0,pageTicks=0;
        static bool returned=false;
        ++ticks;action=0;
        if(!s_nativeRosterPage && !returned && ticks%15u==0)action=4;
        if(s_nativeRosterPage) {
            ++pageTicks;
            if(pageTicks==60u && std::getenv("PS2X_ROSTER_EMPTY_TEST"))action=4;
            if(pageTicks==90u && std::getenv("PS2X_ROSTER_EMPTY_TEST"))action=6;
            if(pageTicks==600u && std::getenv("PS2X_ROSTER_RETURN_TEST")) {action=5;returned=true;}
            if(pageTicks==120u && std::getenv("PS2X_MOD_PAGE_TEST_CONFIRM"))action=6;
        }
        wotmSetReg(c,2,action);
    }
    if(action)s_nativeRosterFocusPlayer=playerTwo?1u:0u;
    // Ready players retain the retail confirm/back behavior; their choice is
    // already stored in Shell independently of the visible page.
    if(s_nativeRosterTwoPlayer && wotmRead32(r,gp-(playerTwo?0x791cu:0x7920u)))return;
    const bool backToRetail=s_nativeRosterPage && action==5;
    if((action==3 && cursor==1u)||(action==4 && cursor==10u)||backToRetail) {
        s_nativeRosterPage=backToRetail?false:!s_nativeRosterPage;
        g_ps2xRosterPage.store(s_nativeRosterPage,std::memory_order_relaxed);
        wotmNativeRosterBrowseTable(r);
        const uint32_t firstAddress=gp-(s_nativeRosterTwoPlayer?0x7940u:0x79e0u);
        wotmWrite32At(r,firstAddress,1);wotmWrite32At(r,firstAddress+4u,10);
        wotmWrite32At(r,cursorAddress,action==4?1u:10u);
        wotmWrite32At(r,costumeAddress,0);
        // We performed the move; don't let retail wrap again or exit the menu.
        wotmSetReg(c,2,0);
        std::fprintf(stderr,"[mods:native-page] page=%u player=%u direction=%u\n",s_nativeRosterPage?2:1,playerTwo?2:1,action);
        return;
    }
    if(cursor<1u || cursor>10u)return;
    const unsigned slot=cursor-1u;
    if(s_nativeRosterPage && (action==1 || action==2)) {
        if(g_ps2xRosterIds[slot]==13u) {
            unsigned& choice=s_nativeRosterCostumeChoice[playerTwo?1u:0u][slot];
            choice=(choice+1u)%2u; // two complete ASSBOSS asset sets exist
            wotmWrite32At(r,costumeAddress,choice);
        }
        wotmSetReg(c,2,0);return;
    }
    if(action==6) {
        if((s_nativeRosterPage && !g_ps2xRosterIds[slot]) ||
           (!s_nativeRosterPage && !s_nativeRosterFlags[slot][wotmRead32(r,costumeAddress)&3u][1])) {
            wotmSetReg(c,2,0);
            if(std::getenv("PS2X_MOD_DIAG"))std::fprintf(stderr,"[mods:native-locked] page=%u slot=%u\n",s_nativeRosterPage?2:1,slot);
            return;
        }
        if(s_nativeRosterPage) {
            const unsigned costume=g_ps2xRosterIds[slot]==13u ? s_nativeRosterCostumeChoice[playerTwo?1u:0u][slot] : g_ps2xRosterCostumes[slot];
            wotmWrite32At(r,costumeAddress,costume);
            if(!playerTwo) {
                g_ps2xRosterSlot.store(static_cast<int>(slot),std::memory_order_relaxed);
                g_ps2xRosterChosenType.store(static_cast<int>(g_ps2xRosterIds[slot]),std::memory_order_relaxed);
            }
            std::fprintf(stderr,"[mods:native-choice] player=%u slot=%u guest=%u costume=%u\n",playerTwo?2:1,slot,g_ps2xRosterIds[slot],costume);
        } else if(!playerTwo)g_ps2xRosterChosenType.store(0,std::memory_order_relaxed);
    }
}
static void wotmNativeRosterScreen(uint8_t* r,R5900Context* c,PS2Runtime* t,
                                  PS2Runtime::RecompiledFunction original) {
    const uint32_t shell=wotmRead32(r,0x6f8200u);
    if(!shell || shell>PS2_RAM_SIZE-0x2c20u) {original(r,c,t);return;}
    const uint32_t gp=getRegU32(c,28);
    if(gp>=0x8000u && gp<PS2_RAM_SIZE && wotmRead32(r,gp-0x7a70u)) {
        for(unsigned player=0;player<2;++player)for(unsigned slot=0;slot<10;++slot)
            s_nativeRosterCostumeChoice[player][slot]=g_ps2xRosterCostumes[slot];
        s_nativeRosterPage=false;
        g_ps2xRosterPage.store(false,std::memory_order_relaxed);
        g_ps2xRosterChosenType.store(0,std::memory_order_relaxed);
    }
    // This scope is bounded to one menu update. Never leave modified unlock
    // records behind for saves, other menus, or gameplay.
    struct Scope { Scope(){++t_ps2xEeCheckpointDeferralDepth;} ~Scope(){--t_ps2xEeCheckpointDeferralDepth;} } scope;
    for(unsigned row=0;row<10;++row) {
        s_nativeRosterMapping[row]=wotmRead32(r,0x2a1a40u+row*4u);
        for(unsigned skin=0;skin<4;++skin)for(unsigned flag=0;flag<2;++flag)
            s_nativeRosterFlags[row][skin][flag]=wotmRead32(r,0x2a1eb0u+row*0xb0u+skin*0x2cu+flag*4u);
    }
    wotmNativeRosterBrowseTable(r);
    const uint32_t screenBefore=wotmRead32(r,0x6f8464u);
    const bool pageBefore=s_nativeRosterPage;
    if(pageBefore && gp>=0x8000u && gp<PS2_RAM_SIZE && screenBefore<22u) {
        for(unsigned player=0;player<(s_nativeRosterTwoPlayer?2u:1u);++player) {
            const uint32_t cursor=wotmRead32(r,0x2b3410u+screenBefore*4u+player*0x58u);
            if(cursor>=1u && cursor<=10u && g_ps2xRosterIds[cursor-1u]==13u)
                wotmWrite32At(r,gp-(s_nativeRosterTwoPlayer?(player?0x7924u:0x7928u):0x79d8u),
                    s_nativeRosterCostumeChoice[player][cursor-1u]);
        }
    }
    s_nativeRosterInScreen=true;
    original(r,c,t);
    s_nativeRosterInScreen=false;
    // Preserve retail costume occupancy changes, excluding temporary locked slots.
    if(!pageBefore && !s_nativeRosterPage)for(unsigned row=0;row<10;++row)
        for(unsigned skin=0;skin<4;++skin)if(s_nativeRosterFlags[row][skin][1])
            s_nativeRosterFlags[row][skin][0]=wotmRead32(r,0x2a1eb0u+row*0xb0u+skin*0x2cu);
    if(wotmRead32(r,0x6f8464u)!=screenBefore)s_nativeRosterPage=false;
    wotmNativeRosterTable(r,false);
    wotmNativeRosterArt(r,shell,s_nativeRosterPage);
    if(s_nativeRosterPage && s_wotmLabelBase && s_wotmOriginalFontPrint) {
        const uint32_t cursor=wotmRead32(r,0x2b3410u+screenBefore*4u+(s_nativeRosterTwoPlayer?s_nativeRosterFocusPlayer*0x58u:0u));
        const unsigned slot=cursor>=1u && cursor<=10u?cursor-1u:0u;
        wotmWriteLabel(r,kWotmTitleSlot,"MOD CHARACTERS");
        wotmSetRowStyle(r,c,t,false);
        wotmPrintRow(r,c,t,68,wotmLabelAddr(kWotmTitleSlot));
        char caption[96];
        if(g_ps2xRosterIds[slot]==13u)
            std::snprintf(caption,sizeof(caption),"%s  -  VARIANT %u OF 2",g_ps2xRosterNames[slot],
                s_nativeRosterCostumeChoice[s_nativeRosterFocusPlayer][slot]+1u);
        else std::snprintf(caption,sizeof(caption),"%s",g_ps2xRosterIds[slot]?g_ps2xRosterNames[slot]:"LOCKED");
        wotmWriteLabel(r,kWotmHintSlot,caption);
        wotmSetRowStyle(r,c,t,true);
        wotmPrintRow(r,c,t,90,wotmLabelAddr(kWotmHintSlot));
    }
    if(s_nativeRosterPage) {
        static bool dumped=false;
        if(!dumped)if(const char* path=std::getenv("PS2X_NATIVE_ROSTER_DUMP")) {
            if(FILE* f=std::fopen(path,"wb")) {
                const size_t written=std::fwrite(r,1,PS2_RAM_SIZE,f);
                std::fclose(f);dumped=true;
                std::fprintf(stderr,"[mods:native-dump] bytes=%zu shell=%08x\n",written,shell);
            }
        }
    }
}

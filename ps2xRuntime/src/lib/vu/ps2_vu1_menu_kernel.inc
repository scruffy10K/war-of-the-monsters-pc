// Shared-image shader 19f8, straight pairs 1a48..1aa8. No MAC/STATUS
// readers observe these FMAC writes (the enclosing block already suppresses
// them). There are no stores or branches. Keep both DIV issue times: the
// first Q becomes visible at pair 1a88, the second remains pending on return.
if(!g_ps2xVu1MenuVertex)g_ps2xVu1MenuVertex=+[](VU1Interpreter &v,Vu1NativeCtx &c)->bool {
    static const bool enabled=[] {const char *p=std::getenv("PS2X_VU1_MENU_VERTEX");return !p || p[0]!='0';}();
    if(!enabled || !v.m_immediateWrites || v.m_cycle+13u>c.budgetEnd || c.dataSize!=16384u ||
       v.m_pipelineNextReady!=~0ull || s_flagCheck || v.m_xgkick.active ||
       v.m_state.branchPending || v.m_state.ebit || v.m_state.haltAfterDelaySlot || v.m_stopRequested)
        return false;
    const auto norm=[](__m128 x) {return _mm_castsi128_ps(vu1n::vuNormalizeBits(_mm_castps_si128(x)));};
    const auto load=[&](unsigned reg) {return _mm_loadu_ps(v.m_state.vf[reg]);};
    const auto bc=[&](__m128 x,unsigned lane) {
        float words[4];_mm_storeu_ps(words,x);
        return _mm_castsi128_ps(vu1n::vuBroadcastNormalized(words[lane]));
    };
    const auto product=[](__m128 result,__m128 a,__m128 b,unsigned lanes) {
        const auto ordinary=vu1n::vuOrdinaryLanes(result) |
            ((vu1n::vuZeroLanes(a)|vu1n::vuZeroLanes(b))&vu1n::vuZeroLanes(result));
        return (lanes&~ordinary)==0;
    };
    const auto sum=[](__m128 result,unsigned lanes) {
        return (lanes&~(vu1n::vuOrdinaryLanes(result)|vu1n::vuZeroLanes(result)))==0;
    };
    const auto madd=[&](__m128 acc,__m128 a,__m128 b,__m128 &result) {
        const auto p=_mm_mul_ps(a,b);result=_mm_add_ps(acc,p);
        const auto zero=vu1n::vuZeroLanes(a)|vu1n::vuZeroLanes(b);
        return ((vu1n::vuOrdinaryLanes(p)|zero)&
            (vu1n::vuOrdinaryLanes(result)|(zero&vu1n::vuZeroLanes(result))))==15u;
    };
    const auto address=[](int32_t vi,int imm) {return (uint32_t(vi+imm)*16u)&16383u;};
    const int32_t vi8=v.m_state.vi[8];
    const auto vertex=_mm_loadu_ps(reinterpret_cast<const float*>(c.vuData+address(v.m_state.vi[6],181)));
    // vf0.w is hardwired to +1; multiplication by it only normalises vf4.
    const auto acc0=norm(load(4));
    const float den0=Vu1NativeAccess::normOp(v.m_state.vf[25][3]);
    if(den0==0.0f)return false;
    const float q0=1.0f/den0;
    if(vu1n::vuOrdinaryLanes(_mm_set1_ps(q0))!=15u)return false;
    const auto r13=norm(load(13));
    const auto addZ=_mm_add_ps(r13,bc(load(25),3));
    if(!sum(addZ,4u))return false;
    auto r5=_mm_blend_ps(load(5),addZ,4);
    const auto r7=_mm_loadu_ps(reinterpret_cast<const float*>(c.vuData+address(vi8,5)));
    __m128 acc1,acc2,r26;
    if(!madd(acc0,norm(load(1)),bc(vertex,0),acc1) ||
       !madd(acc1,norm(load(2)),bc(vertex,1),acc2) ||
       !madd(acc2,norm(load(3)),bc(vertex,2),r26))return false;
    uint32_t iw;std::memcpy(&iw,c.vuData+address(vi8,6)+12u,4u);
    const int32_t vi9=static_cast<int16_t>(iw);
    const auto color=_mm_mul_ps(_mm_cvtepi32_ps(_mm_castps_si128(r7)),_mm_set1_ps(1.0f/4096.0f));
    const auto r11=norm(load(9)); // MULbc by vf0.w
    const auto q=_mm_set1_ps(q0);
    const auto r25=norm(load(25));
    const auto p23=_mm_mul_ps(r25,q),p8=_mm_mul_ps(r11,q);
    if(!product(p23,r25,q,7u) || !product(p8,r11,q,7u))return false;
    const auto r23=_mm_blend_ps(load(23),p23,7);
    const auto addXY=_mm_add_ps(norm(r23),r13);
    if(!sum(addXY,3u))return false;
    r5=_mm_blend_ps(r5,addXY,3);
    float projected[4];_mm_storeu_ps(projected,r26);
    const float den1=Vu1NativeAccess::normOp(projected[3]);
    if(den1==0.0f)return false;
    const float q1=1.0f/den1;
    if(vu1n::vuOrdinaryLanes(_mm_set1_ps(q1))!=15u)return false;
    const auto nextVertex=_mm_loadu_ps(reinterpret_cast<const float*>(c.vuData+address(vi9,181)));
    const int32_t masked=v.m_state.vi[10]&v.m_state.vi[4];
    const int32_t twice=static_cast<int16_t>(masked+masked);
    const int32_t thrice=static_cast<int16_t>(twice+masked);
    const int32_t next=static_cast<int16_t>(thrice+vi8);
    static const bool verify=std::getenv("PS2X_VU1_MENU_VERTEX_VERIFY")!=nullptr;
    static uint64_t candidates=0,checked=0,hits=0;
    const bool check=verify && (++candidates<=64u || candidates%128u==0);
    auto reference=check?std::make_unique<VU1Interpreter>(v):nullptr;
    const uint64_t startCycle=v.m_cycle;
    const uint64_t startCount=c.instrCount;
    // Preserve even inactive branch-read metadata from the last VI write.
    const auto note=[&]<uint32_t U,uint32_t L>(int32_t oldValue) {
        constexpr auto d=Vu1NativeAccess::PairDecode<U,L>::d;
        constexpr uint32_t bits=uint32_t(d.lowerUsage.viWrite)&0xfffeu;
        if constexpr(bits && d.lowerUsage.delaysNextBranchRead)
            v.recordViWriteForBranch(std::countr_zero(bits),oldValue);
    };
    note.operator()<locallyGeneratedVu::upper<0x0060u>,locallyGeneratedVu::lower<0x1a60u>>(v.m_state.vi[5]);
    note.operator()<locallyGeneratedVu::upper<0x0068u>,locallyGeneratedVu::lower<0x00c0u>>(v.m_state.vi[10]);
    note.operator()<locallyGeneratedVu::upper<0x00d0u>,locallyGeneratedVu::lower<0x0130u>>(v.m_state.vi[9]);
    note.operator()<locallyGeneratedVu::upper<0x1a80u>,locallyGeneratedVu::lower<0x00c8u>>(v.m_state.vi[3]);
    note.operator()<locallyGeneratedVu::upper<0x1a88u>,locallyGeneratedVu::lower<0x00d0u>>(twice);
    note.operator()<locallyGeneratedVu::upper<0x0100u>,locallyGeneratedVu::lower<0x00e0u>>(masked);
    note.operator()<locallyGeneratedVu::upper<0x0000u>,locallyGeneratedVu::lower<0x0108u>>(v.m_state.vi[12]);
    _mm_storeu_ps(v.m_state.vf[5],r5);
    _mm_storeu_ps(v.m_state.vf[7],color);
    _mm_storeu_ps(v.m_state.vf[8],_mm_blend_ps(load(8),p8,7));
    _mm_storeu_ps(v.m_state.vf[11],r11);
    _mm_storeu_ps(v.m_state.vf[20],nextVertex);
    _mm_storeu_ps(v.m_state.vf[23],r23);
    _mm_storeu_ps(v.m_state.vf[26],r26);
    _mm_storeu_ps(v.m_state.acc,acc2);
    v.m_state.vi[5]=864;v.m_state.vi[9]=vi9;v.m_state.vi[3]=thrice;
    v.m_state.vi[10]=next;v.m_state.vi[12]=static_cast<int16_t>(vi8+4);
    v.m_state.q=q0;v.m_state.i=255.0f;
    v.m_state.status &= 0xfcfu; // first DIV completed with neither D nor I
    if(s_flagCheck)v.m_shadowStatus &= 0xfcfu;
    v.m_cycle=startCycle+12u;
    v.queueQ(q1,7u,0u);
    Vu1NativeAccess::advanceOneCycleFast(v);
    v.m_currentUpperInstruction=locallyGeneratedVu::upper<0x1aa8u>;
    v.m_viBranchBackupValid=false;c.instrCount+=13u;
    if(check) {
        Vu1NativeCtx rc=c;rc.instrCount=startCount;
        const auto step=[&]<uint32_t U,uint32_t L,bool Imm>() {
            if(Vu1NativeAccess::stepPair<U,L,Imm,true,true,true,true>(*reference,rc)!=Vu1NativeStep::Continue)std::abort();
        };
        step.operator()<locallyGeneratedVu::upper<0x0040u>,locallyGeneratedVu::lower<0x00a0u>,true>();
        step.operator()<locallyGeneratedVu::upper<0x0000u>,locallyGeneratedVu::lower<0x00a8u>,true>();
        step.operator()<locallyGeneratedVu::upper<0x0138u>,locallyGeneratedVu::lower<0x00b8u>,true>();
        step.operator()<locallyGeneratedVu::upper<0x0060u>,locallyGeneratedVu::lower<0x1a60u>,true>();
        step.operator()<locallyGeneratedVu::upper<0x0068u>,locallyGeneratedVu::lower<0x00c0u>,true>();
        step.operator()<locallyGeneratedVu::upper<0x00d0u>,locallyGeneratedVu::lower<0x0130u>,true>();
        step.operator()<locallyGeneratedVu::upper<0x1908u>,locallyGeneratedVu::lower<0x0128u>,true>();
        step.operator()<locallyGeneratedVu::upper<0x1a80u>,locallyGeneratedVu::lower<0x00c8u>,true>();
        step.operator()<locallyGeneratedVu::upper<0x1a88u>,locallyGeneratedVu::lower<0x00d0u>,false>();
        step.operator()<locallyGeneratedVu::upper<0x0000u>,locallyGeneratedVu::lower<0x0058u>,false>();
        step.operator()<locallyGeneratedVu::upper<0x0100u>,locallyGeneratedVu::lower<0x00e0u>,false>();
        step.operator()<locallyGeneratedVu::upper<0x0000u>,locallyGeneratedVu::lower<0x0108u>,false>();
        step.operator()<locallyGeneratedVu::upper<0x1aa8u>,locallyGeneratedVu::lower<0x0140u>,false>();
        const char *field=nullptr;
        bool same=Vu1NativeAccess::hleSameState(v,*reference,&field) && rc.instrCount==c.instrCount &&
            v.m_fdiv.readyCycle==reference->m_fdiv.readyCycle &&
            std::memcmp(&v.m_fdiv.value,&reference->m_fdiv.value,4u)==0 &&
            v.m_fdiv.statusDi==reference->m_fdiv.statusDi;
        if(!same) {std::fprintf(stderr,"[vu1:menu-vertex] MISMATCH %s\n",field?field:"Q/count");std::abort();}
        if(++checked==1 || checked%10000u==0)std::fprintf(stderr,"[vu1:menu-vertex] verified=%llu mismatches=0\n",checked);
    }
    if(++hits==1 || hits%1000000u==0)std::fprintf(stderr,"[vu1:menu-vertex] vertices=%llu\n",hits);
    return true;
};

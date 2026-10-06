// Complete shared 19f8 loop, 1ab8..1b20. Keep matrices/live values in SSE
// registers across vertices; preserve Q, CLIP, MAC and store sequencing.
if(!g_ps2xVu1MenuLoop)g_ps2xVu1MenuLoop=+[](VU1Interpreter &v,Vu1NativeCtx &c)->bool {
    static const bool enabled=[] {const char *p=std::getenv("PS2X_VU1_MENU_LOOP");return !p || p[0]!='0';}();
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    return false;
#endif
    const bool quietQ=v.m_fdiv.valid
        ? v.m_pipelineNextReady==v.m_fdiv.readyCycle &&
          (v.m_fdiv.readyCycle-v.m_cycle==4u || v.m_fdiv.readyCycle-v.m_cycle==5u)
        : v.m_pipelineNextReady==~0ull;
    const uint32_t vf0[4]={0u,0u,0u,0x3f800000u};
    if(!enabled || !v.m_immediateWrites || !s_nativeStores || t_nativeStoreReference || s_flagCheck ||
       v.m_unit!=Unit::VU1 || c.codeSize!=16384u || c.dataSize!=16384u ||
       v.m_activeVuData!=c.vuData || v.m_activeVuDataSize!=c.dataSize || !quietQ ||
       v.m_storeMask || v.m_vfWriteMask || v.m_viWriteMask || v.m_accWriteMask ||
       v.m_efu[0].valid || v.m_efu[1].valid || v.m_xgkick.active ||
       v.m_state.branchPending || v.m_state.ebit || v.m_state.haltAfterDelaySlot || v.m_stopRequested ||
       v.m_cycle>=c.budgetEnd || c.budgetEnd-v.m_cycle<14u ||
       std::memcmp(v.m_state.vf[0],vf0,16u) || v.m_state.vi[0]!=0)
        return false;
    for(uint32_t live=v.m_flagMask;live;live&=live-1u)
        if(v.m_flagPipeline[std::countr_zero(live)].readyCycle>v.m_cycle+6u)return false;
    if(v.m_flagOrderCount>VU1Interpreter::kMaxFlagEntries-2u)return false;
    static const bool verify=std::getenv("PS2X_VU1_MENU_LOOP_VERIFY")!=nullptr;
    static uint64_t candidates=0,checked=0,chunks=0,vertices=0;
    const bool check=verify && (++candidates<=64u || candidates%128u==0u);
    auto reference=check?std::make_unique<VU1Interpreter>(v):nullptr;
    std::vector<uint8_t> referenceData;
    const uint64_t startCount=c.instrCount;
    if(check) {
        referenceData.assign(c.vuData,c.vuData+c.dataSize);
        reference->m_activeVuData=referenceData.data();
    }
    const auto norm=[](__m128 x) {return _mm_castsi128_ps(vu1n::vuNormalizeBits(_mm_castps_si128(x)));};
    const auto bc=[]<unsigned Lane>(__m128 x) {
        x=_mm_castsi128_ps(vu1n::vuNormalizeBits(_mm_castps_si128(x)));
        return _mm_shuffle_ps(x,x,Lane*0x55u);
    };
    const auto product=[](__m128 r,__m128 a,__m128 b,unsigned lanes) {
        return (lanes&~(vu1n::vuOrdinaryLanes(r)|
            ((vu1n::vuZeroLanes(a)|vu1n::vuZeroLanes(b))&vu1n::vuZeroLanes(r))))==0u;
    };
    const auto sum=[](__m128 r,unsigned lanes) {
        return (lanes&~(vu1n::vuOrdinaryLanes(r)|vu1n::vuZeroLanes(r)))==0u;
    };
    const auto madd=[](__m128 p,__m128 r,__m128 a,__m128 b) {
        const auto zero=vu1n::vuZeroLanes(a)|vu1n::vuZeroLanes(b);
        return ((vu1n::vuOrdinaryLanes(p)|zero)&
            (vu1n::vuOrdinaryLanes(r)|(zero&vu1n::vuZeroLanes(r))))==15u;
    };
    const auto flags=[&]<uint8_t Dest,bool Imm>(__m128 result,uint32_t sticky=0u) {
        const uint32_t neg=_mm_movemask_ps(result),zero=vu1n::vuZeroLanes(result);
        uint8_t lanes[4];
        for(unsigned i=0;i<4u;++i)lanes[i]=uint8_t(((neg>>i)&1u)*2u+((zero>>i)&1u));
        Vu1NativeAccess::updateFmacFlagsC<Dest,Imm,false>(v,lanes,sticky);
    };
    const auto address=[](int32_t vi,int imm) {return (uint32_t(vi+imm)*16u)&16383u;};
    __m128 R[32];
    constexpr unsigned needed[]={1,2,3,4,5,7,8,9,11,13,14,20,23,24,25,26,28,29,30};
    for(unsigned i:needed)R[i]=_mm_loadu_ps(v.m_state.vf[i]);
    const auto matrix1=norm(R[1]),matrix2=norm(R[2]),matrix3=norm(R[3]),translation=norm(R[4]);
    const auto screen=norm(R[13]),w14=bc.operator()<3>(R[14]);
    unsigned iterations=0;
    __m128 ACC;
    do {
        const uint64_t at=v.m_cycle;
        const int32_t vi8=v.m_state.vi[8];
        // CLIP uses raw bit ordering (not a host float comparison).
        uint32_t raw5[4],raw13[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(raw5),_mm_castps_si128(R[5]));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(raw13),_mm_castps_si128(R[13]));
        const int32_t limit=(raw13[3]&0x7f800000u)?int32_t(raw13[3]&0x7fffffffu):0x007fffff;
        uint32_t clip=0;
        for(unsigned i=0;i<3u;++i) {
            if(int32_t(raw5[i])>limit)clip|=1u<<(i*2u);
            if(int32_t(raw5[i]^0x80000000u)>limit)clip|=2u<<(i*2u);
        }
        const uint32_t workingClip=((v.m_workingClip<<6u)|clip)&0xffffffu;
        if(workingClip&0x02fbefu)break; // Original clipped route handles this vertex.
        const auto next9=_mm_loadu_ps(reinterpret_cast<const float*>(c.vuData+address(vi8,8)));
        uint32_t iw;std::memcpy(&iw,c.vuData+address(vi8,9)+12u,4u);
        const int32_t vi9=static_cast<int16_t>(iw);
        const uint32_t firstStore=address(vi8,0),vertexLoad=address(vi9,181);
        if(firstStore==vertexLoad)break; // Never move a load ahead of its aliased store.
        const auto next20=_mm_loadu_ps(reinterpret_cast<const float*>(c.vuData+vertexLoad));
        const auto x=bc.operator()<0>(R[20]),y=bc.operator()<1>(R[20]),z=bc.operator()<2>(R[20]);
        const auto px=_mm_mul_ps(matrix1,x),ax=_mm_add_ps(translation,px);
        const auto py=_mm_mul_ps(matrix2,y),ay=_mm_add_ps(ax,py);
        const auto pz=_mm_mul_ps(matrix3,z),next26=_mm_add_ps(ay,pz);
        if(!madd(px,ax,matrix1,x) || !madd(py,ay,matrix2,y) || !madd(pz,next26,matrix3,z))break;
        const auto addW=_mm_add_ps(bc.operator()<3>(R[11]),w14);
        if(!sum(addW,8u))break;
        const auto next24=_mm_blend_ps(R[23],addW,8u);
        const float previousQ=v.m_fdiv.valid?v.m_fdiv.value:v.m_state.q;
        const auto q=_mm_set1_ps(Vu1NativeAccess::normOp(previousQ));
        const auto a26=norm(R[26]),a7=norm(R[7]);
        const auto p23=_mm_mul_ps(a26,q),p8=_mm_mul_ps(a7,q);
        if(!product(p23,a26,q,7u) || !product(p8,a7,q,7u))break;
        const auto next23=_mm_blend_ps(R[23],p23,7u),next8=_mm_blend_ps(R[8],p8,7u);
        const auto addZ=_mm_add_ps(screen,bc.operator()<3>(R[26]));
        const auto addXY=_mm_add_ps(norm(next23),screen);
        if(!sum(addZ,4u) || !sum(addXY,3u))break;
        const auto next5=_mm_blend_ps(_mm_blend_ps(R[5],addZ,4u),addXY,3u);
        float convertedSource[4];_mm_storeu_ps(convertedSource,next24);
        const auto next30=_mm_castsi128_ps(vu1ConvertVector<16>(convertedSource));
        const auto next7=_mm_mul_ps(_mm_cvtepi32_ps(_mm_castps_si128(next9)),_mm_set1_ps(1.0f/4096.0f));
        float projected[4];_mm_storeu_ps(projected,next26);
        const float den=Vu1NativeAccess::normOp(projected[3]);
        if(den==0.0f)break;
        const float nextQ=1.0f/den;
        if(vu1n::vuOrdinaryLanes(_mm_set1_ps(nextQ))!=15u)break;
        const auto next29=norm(R[5]); // MULw VF29 = VF28 * hardwired +1; ACC stays MADDAy.
        const int32_t nextVi8=static_cast<int16_t>(vi8+3);
        const bool loop=static_cast<int16_t>(nextVi8)!=static_cast<int16_t>(v.m_state.vi[10]);
        // All guards are above the first mutation. Functions below retain the
        // original FIFO allocation/retirement and scalar pipeline metadata.
        v.m_cycle=at+2u;v.queueClip(clip);
        if(v.m_fdiv.valid) {
            v.m_cycle=v.m_fdiv.readyCycle;v.m_state.cycles=v.m_cycle;v.commitReadyPipelines();
        }
        std::memcpy(c.vuData+firstStore,&R[8],12u);
        v.m_cycle=at+6u;flags.operator()<14u,false>(p23);v.retireFlags();
        v.m_state.vi[1]=0;
        v.m_cycle=at+7u;
        const uint32_t sticky=((_mm_movemask_ps(pz)!=0)?2u:0u)|(vu1n::vuZeroLanes(pz)?1u:0u);
        flags.operator()<15u,true>(next26,sticky);
        v.m_cycle=at+8u;flags.operator()<14u,true>(p8);
        v.m_cycle=at+10u;flags.operator()<2u,true>(addZ);
        v.recordViWriteForBranch(8u,vi8);
        v.m_state.vi[8]=nextVi8;v.m_state.vi[9]=vi9;
        v.m_cycle=at+11u;flags.operator()<12u,true>(addXY);v.queueQ(nextQ,7u,0u);
        v.m_cycle=at+13u;flags.operator()<15u,true>(next29);
        std::memcpy(c.vuData+address(nextVi8,-1),&next30,16u);
        R[9]=next9;R[24]=next24;R[25]=R[26];R[28]=R[5];R[11]=R[7];
        R[23]=next23;R[26]=next26;R[8]=next8;R[20]=next20;
        R[30]=next30;R[5]=next5;R[7]=next7;R[29]=next29;ACC=ay;
        v.m_viBranchBackupValid=false;
        v.m_currentUpperInstruction=locallyGeneratedVu::upper<0x1b20u>;
        v.m_cycle=at+14u;v.m_state.cycles=v.m_cycle;c.instrCount+=14u;
        v.m_state.branchPending=false;
        if(loop) {v.m_state.branchTarget=0x1ab8u;v.m_state.branchDelay=0u;}
        v.m_state.pc=loop?0x1ab8u:0x1b28u;
        ++iterations;
        if(!loop)break;
    } while(iterations<64u && v.m_cycle<c.budgetEnd && c.budgetEnd-v.m_cycle>=14u);
    if(!iterations)return false;
    constexpr unsigned written[]={5,7,8,9,11,20,23,24,25,26,28,29,30};
    for(unsigned i:written)_mm_storeu_ps(v.m_state.vf[i],R[i]);
    _mm_storeu_ps(v.m_state.acc,ACC);
    if(check) {
        Vu1NativeCtx rc=c;rc.vuData=referenceData.data();rc.instrCount=startCount;
        const auto step=[&]<uint32_t U,uint32_t L,bool Imm,bool NoFlags,bool Mid>() {
            if(Vu1NativeAccess::stepPair<U,L,Imm,true,true,NoFlags,Mid>(*reference,rc)!=Vu1NativeStep::Continue)std::abort();
        };
        for(unsigned i=0;i<iterations;++i) {
            step.operator()<locallyGeneratedVu::upper<0x0040u>,locallyGeneratedVu::lower<0x0168u>,true,true,true>();
            step.operator()<locallyGeneratedVu::upper<0x0060u>,locallyGeneratedVu::lower<0x02f0u>,false,true,true>();
            step.operator()<locallyGeneratedVu::upper<0x0170u>,locallyGeneratedVu::lower<0x01d8u>,false,true,true>();
            step.operator()<locallyGeneratedVu::upper<0x0068u>,locallyGeneratedVu::lower<0x0180u>,false,true,true>();
            step.operator()<locallyGeneratedVu::upper<0x1ad8u>,locallyGeneratedVu::lower<0x0150u>,false,true,true>();
            step.operator()<locallyGeneratedVu::upper<0x0000u>,locallyGeneratedVu::lower<0x01a0u>,false,true,true>();
            step.operator()<locallyGeneratedVu::upper<0x1ae8u>,locallyGeneratedVu::lower<0x0190u>,false,false,true>();
            reference->m_state.pc=0x1af0u;
            step.operator()<locallyGeneratedVu::upper<0x00d0u>,locallyGeneratedVu::lower<0x1af0u>,true,false,false>();
            step.operator()<locallyGeneratedVu::upper<0x0198u>,locallyGeneratedVu::lower<0x0058u>,true,false,false>();
            step.operator()<locallyGeneratedVu::upper<0x19b0u>,locallyGeneratedVu::lower<0x01c0u>,true,false,true>();
            step.operator()<locallyGeneratedVu::upper<0x0138u>,locallyGeneratedVu::lower<0x01d0u>,true,false,true>();
            step.operator()<locallyGeneratedVu::upper<0x1aa8u>,locallyGeneratedVu::lower<0x0140u>,true,false,true>();
            reference->m_state.pc=0x1b18u;
            step.operator()<locallyGeneratedVu::upper<0x01c8u>,locallyGeneratedVu::lower<0x0bd8u>,true,false,false>();
            step.operator()<locallyGeneratedVu::upper<0x1b20u>,locallyGeneratedVu::lower<0x13a8u>,true,false,false>();
        }
        const char *field=nullptr;
        const bool same=Vu1NativeAccess::hleSameState(v,*reference,&field) && rc.instrCount==c.instrCount &&
            std::memcmp(c.vuData,referenceData.data(),c.dataSize)==0 && v.m_flagOrder==reference->m_flagOrder &&
            v.m_maxReadyCycle==reference->m_maxReadyCycle && v.m_fdiv.readyCycle==reference->m_fdiv.readyCycle &&
            std::memcmp(&v.m_fdiv.value,&reference->m_fdiv.value,4u)==0 && v.m_fdiv.statusDi==reference->m_fdiv.statusDi;
        if(!same) {
            std::fprintf(stderr,"[vu1:menu-loop] MISMATCH %s count=%u\n",field?field:"count/flags/Q/data",iterations);
            for(unsigned reg=0;reg<32u;++reg)for(unsigned lane=0;lane<4u;++lane) {
                uint32_t a,b;std::memcpy(&a,&v.m_state.vf[reg][lane],4u);std::memcpy(&b,&reference->m_state.vf[reg][lane],4u);
                if(a!=b)std::fprintf(stderr,"[vu1:menu-loop] vf%u.%u actual=%08x expected=%08x\n",reg,lane,a,b);
            }
            std::abort();
        }
        if(++checked==1u || checked%10000u==0u)std::fprintf(stderr,"[vu1:menu-loop] verified=%llu mismatches=0\n",checked);
    }
    vertices+=iterations;
    if(++chunks==1u || chunks%1000000u==0u)
        std::fprintf(stderr,"[vu1:menu-loop] chunks=%llu vertices=%llu\n",chunks,vertices);
    return true;
};

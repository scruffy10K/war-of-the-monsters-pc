// Initialized from the VU member context to keep private pipeline access local.
// No public/header changes; only shader15e8 in the two validated code images.
if (!g_ps2xVu1ObjectKernel[0]) {
    const auto matrix=[]<uint32_t U0,uint32_t L0,uint32_t U1,uint32_t L1,uint32_t U2,uint32_t L2,bool ToAcc>(VU1Interpreter &v,Vu1NativeCtx &c) {
        constexpr unsigned kind=U0==locallyGeneratedVu::upper<0x1790u>?2u:L0==locallyGeneratedVu::lower<0x1750u>?1u:0u;
        static const bool profile=std::getenv("PS2X_VU1_OBJECT_PROFILE") || std::getenv("PS2X_VU1_OBJECT_TEST");
        const auto reference=[&] {
            if(profile)++g_ps2xVu1ObjectFallbacks[kind];
            auto r=Vu1NativeAccess::stepPair<U0,L0,true,true,true,false,true>(v,c);
            if(r!=Vu1NativeStep::Continue)return r;
            r=Vu1NativeAccess::stepPair<U1,L1,true,true,true,false,true>(v,c);
            if(r!=Vu1NativeStep::Continue)return r;
            return Vu1NativeAccess::stepPair<U2,L2,true,true,true,false,true>(v,c);
        };
        if(v.m_cycle>=v.m_pipelineNextReady)v.commitReadyPipelines();
        // Scalar Q/P completions are safe: the matrix only reads VF registers.
        // Keep their issue/retirement cycles, but decline pending register or
        // memory writes and queued flags that could change the captured inputs.
        if(!v.m_immediateWrites || v.m_flagOrderCount || v.m_xgkick.active ||
           v.m_vfWriteMask || v.m_viWriteMask || v.m_accWriteMask || v.m_storeMask)
            return reference();
        constexpr unsigned dest=(U0>>21)&15u;
        static_assert(dest==((U1>>21)&15u) && dest==((U2>>21)&15u));
        constexpr unsigned lanes=((dest&8)>>3)|((dest&4)>>1)|((dest&2)<<1)|((dest&1)<<3);
        static_assert(((U0>>16)&31u)==((U1>>16)&31u) && ((U0>>16)&31u)==((U2>>16)&31u));
        const __m128 xyz=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[(U0>>16)&31u]));
        const __m128 x=_mm_shuffle_ps(xyz,xyz,0),y=_mm_shuffle_ps(xyz,xyz,0x55),z=_mm_shuffle_ps(xyz,xyz,0xaa);
        const __m128 a=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[(U0>>11)&31u]));
        const __m128 b=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[(U1>>11)&31u]));
        const __m128 d=_mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.vf[(U2>>11)&31u]));
        const __m128 p0=_mm_mul_ps(a,x),p1=_mm_mul_ps(b,y),p2=_mm_mul_ps(d,z);
        const __m128 r1=_mm_add_ps(p0,p1),r2=_mm_add_ps(r1,p2);
        const auto product=[](__m128 p,__m128 l,__m128 r) {return vu1n::vuOrdinaryLanes(p)|((vu1n::vuZeroLanes(l)|vu1n::vuZeroLanes(r))&vu1n::vuZeroLanes(p));};
        const auto sum=[](__m128 p,__m128 res,__m128 l,__m128 r) {const unsigned zero=vu1n::vuZeroLanes(l)|vu1n::vuZeroLanes(r);return (vu1n::vuOrdinaryLanes(p)|zero)&(vu1n::vuOrdinaryLanes(res)|(zero&vu1n::vuZeroLanes(res)));};
        if((lanes & ~(product(p0,a,x)&sum(p1,r1,b,y)&sum(p2,r2,d,z)))!=0) return reference();
        if(profile)++g_ps2xVu1ObjectHits[kind];
        // No lower half reads MAC/STATUS. Combine the ordinary S/Z sticky
        // contributions and publish the final flags once, at the same cycle.
        const unsigned neg=unsigned(_mm_movemask_ps(r2)),zero=vu1n::vuZeroLanes(r2);
        uint8_t f[4];for(unsigned i=0;i<4;++i)f[i]=uint8_t(((neg>>i)&1u)*2u+((zero>>i)&1u));
        const unsigned signs=unsigned(_mm_movemask_ps(p0)|_mm_movemask_ps(p1)|_mm_movemask_ps(r1)|_mm_movemask_ps(p2));
        const unsigned zeros=vu1n::vuZeroLanes(p0)|vu1n::vuZeroLanes(p1)|vu1n::vuZeroLanes(r1)|vu1n::vuZeroLanes(p2);
        const unsigned sticky=((signs&lanes)?2u:0u)|((zeros&lanes)?1u:0u);
        Vu1NativeAccess::execLowerN<U0,L0>(v,c);Vu1NativeAccess::advanceOneCycleFast(v);
        Vu1NativeAccess::execLowerN<U1,L1>(v,c);Vu1NativeAccess::advanceOneCycleFast(v);
        Vu1NativeAccess::updateFmacFlagsC<dest,true,false>(v,f,sticky);
        if constexpr((U2&0x80000000u)!=0) {float imm;constexpr uint32_t bits=L2;std::memcpy(&imm,&bits,4);v.m_state.i=v.normalizeOperand(imm);}
        else Vu1NativeAccess::execLowerN<U2,L2>(v,c);
        const __m128 oldAcc=_mm_loadu_ps(v.m_state.acc);
        _mm_storeu_ps(v.m_state.acc,_mm_blend_ps(oldAcc,ToAcc?r2:r1,lanes));
        if constexpr(!ToAcc) {
            auto *dst=v.m_state.vf[(U2>>6)&31u];_mm_storeu_ps(dst,_mm_blend_ps(_mm_loadu_ps(dst),r2,lanes));
        }
        v.m_currentUpperInstruction=U2;v.m_viBranchBackupValid=false;
        v.m_state.vf[0][0]=v.m_state.vf[0][1]=v.m_state.vf[0][2]=0;v.m_state.vf[0][3]=1;v.m_state.vi[0]=0;
        c.instrCount+=3;Vu1NativeAccess::advanceOneCycleFast(v);
        return Vu1NativeStep::Continue;
    };
    g_ps2xVu1ObjectKernel[0]=+[](VU1Interpreter &v,Vu1NativeCtx &c){return decltype(matrix){}.operator()<locallyGeneratedVu::upper<0x1698u>,locallyGeneratedVu::lower<0x0008u>,locallyGeneratedVu::upper<0x16a0u>,locallyGeneratedVu::lower<0x16a0u>,locallyGeneratedVu::upper<0x16a8u>,locallyGeneratedVu::lower<0x16a8u>,false>(v,c);};
    g_ps2xVu1ObjectKernel[1]=+[](VU1Interpreter &v,Vu1NativeCtx &c){return decltype(matrix){}.operator()<locallyGeneratedVu::upper<0x1698u>,locallyGeneratedVu::lower<0x1750u>,locallyGeneratedVu::upper<0x16a0u>,locallyGeneratedVu::lower<0x0008u>,locallyGeneratedVu::upper<0x1760u>,locallyGeneratedVu::lower<0x1760u>,false>(v,c);};
    g_ps2xVu1ObjectKernel[2]=+[](VU1Interpreter &v,Vu1NativeCtx &c){return decltype(matrix){}.operator()<locallyGeneratedVu::upper<0x1790u>,locallyGeneratedVu::lower<0x0008u>,locallyGeneratedVu::upper<0x1798u>,locallyGeneratedVu::lower<0x0008u>,locallyGeneratedVu::upper<0x17a0u>,locallyGeneratedVu::lower<0x16a8u>,true>(v,c);};
    g_ps2xVu1ObjectKernel[3]=+[](VU1Interpreter &v,Vu1NativeCtx &c) {
        static const bool profile=std::getenv("PS2X_VU1_OBJECT_PROFILE") || std::getenv("PS2X_VU1_OBJECT_TEST");
        if(profile)++g_ps2xVu1ObjectHits[3];
        if(v.m_cycle>=v.m_pipelineNextReady)v.commitReadyPipelines();
        // CLIPw vf5,vf13: use signed integer comparisons like the VU, including
        // denormal W and exponent255; ordinary floating comparisons differ.
        const uint32_t w=std::bit_cast<uint32_t>(v.m_state.vf[13][3]);
        const __m128i limit=_mm_set1_epi32((w&0x7f800000u)?int32_t(w&0x7fffffffu):0x007fffff);
        const __m128i xyz=_mm_loadu_si128(reinterpret_cast<const __m128i*>(v.m_state.vf[5]));
        const unsigned pos=_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(xyz,limit)));
        const unsigned neg=_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(_mm_xor_si128(xyz,_mm_set1_epi32(int32_t(0x80000000u))),limit)));
        v.m_currentUpperInstruction=locallyGeneratedVu::upper<0x0170u>;
        v.queueClip((pos&1u)|((neg&1u)<<1)|((pos&2u)<<1)|((neg&2u)<<2)|((pos&4u)<<2)|((neg&4u)<<3));
        v.m_viBranchBackupValid=false;v.m_state.vi[0]=0;
        v.m_state.vf[0][0]=v.m_state.vf[0][1]=v.m_state.vf[0][2]=0;v.m_state.vf[0][3]=1;
        ++c.instrCount;Vu1NativeAccess::advanceOneCycleFast(v);return Vu1NativeStep::Continue;
    };
}

if(t_vuBenchReplay && std::getenv("PS2X_VU1_OBJECT_TEST")) {
    unsigned failures=0;uint32_t random=0x58291adu;
    const uint32_t special[]={0,0x80000000u,1,0x80000001u,0x007fffffu,0x00800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,0x7fc00000u,0x3f800000u};
    for(unsigned test=0;test<8192;++test) {
        *clone=*this;data=input;
        clone->m_activeVuData=data.data();clone->m_activeMemory=nullptr;clone->m_activeGs=nullptr;
        clone->m_gifCapture=&gif;clone->m_gifCaptureOnly=true;
        clone->m_pipelineNextReady=~0ull;clone->m_xgkick.active=false;
        clone->m_flagOrderCount=0;clone->m_flagMask=0;clone->m_storeMask=0;
        clone->m_fdiv.valid=false;clone->m_efu[0].valid=clone->m_efu[1].valid=false;
        for(unsigned reg=1;reg<32;++reg)for(unsigned lane=0;lane<4;++lane) {
            random^=random<<13;random^=random>>17;random^=random<<5;
            const uint32_t bits=test%3==0?special[random%12]:test%3==1?random:((random&0x807fffffu)|((120u+random%14u)<<23));
            std::memcpy(&clone->m_state.vf[reg][lane],&bits,4);
        }
        if(test%7==0) {clone->m_fdiv.valid=true;clone->m_fdiv.readyCycle=clone->m_cycle+1;clone->m_fdiv.value=1.5f;clone->m_fdiv.statusDi=0;clone->m_pipelineNextReady=clone->m_cycle+1;}
        auto seed=std::make_unique<VU1Interpreter>(*clone);
        Vu1NativeCtx q{data.data(),dataSize,codeSize,&gs,nullptr,budgetEnd,0};
        const auto reference=[&]<uint32_t U0,uint32_t L0,uint32_t U1,uint32_t L1,uint32_t U2,uint32_t L2>() {
            auto r=Vu1NativeAccess::stepPair<U0,L0,true,true,true,false,true>(*clone,q);
            if(r==Vu1NativeStep::Continue)r=Vu1NativeAccess::stepPair<U1,L1,true,true,true,false,true>(*clone,q);
            if(r==Vu1NativeStep::Continue)r=Vu1NativeAccess::stepPair<U2,L2,true,true,true,false,true>(*clone,q);
            return r;
        };
        Vu1NativeStep r;
        switch(test%4) {
        case 0:r=reference.operator()<locallyGeneratedVu::upper<0x1698u>,locallyGeneratedVu::lower<0x0008u>,locallyGeneratedVu::upper<0x16a0u>,locallyGeneratedVu::lower<0x16a0u>,locallyGeneratedVu::upper<0x16a8u>,locallyGeneratedVu::lower<0x16a8u>>();break;
        case 1:r=reference.operator()<locallyGeneratedVu::upper<0x1698u>,locallyGeneratedVu::lower<0x1750u>,locallyGeneratedVu::upper<0x16a0u>,locallyGeneratedVu::lower<0x0008u>,locallyGeneratedVu::upper<0x1760u>,locallyGeneratedVu::lower<0x1760u>>();break;
        case 2:r=reference.operator()<locallyGeneratedVu::upper<0x1790u>,locallyGeneratedVu::lower<0x0008u>,locallyGeneratedVu::upper<0x1798u>,locallyGeneratedVu::lower<0x0008u>,locallyGeneratedVu::upper<0x17a0u>,locallyGeneratedVu::lower<0x16a8u>>();break;
        default:r=Vu1NativeAccess::stepPair<locallyGeneratedVu::upper<0x0170u>,locallyGeneratedVu::lower<0x0008u>,false,true,true,false,true>(*clone,q);break;
        }
        *expected=*clone;*clone=*seed;
        Vu1NativeCtx z{data.data(),dataSize,codeSize,&gs,nullptr,budgetEnd,0};
        const auto got=g_ps2xVu1ObjectKernel[test%4](*clone,z);
        const char *what="";
        if(got!=r || q.instrCount!=z.instrCount || !Vu1NativeAccess::hleSameState(*clone,*expected,&what)) {
            if(++failures<=4)std::fprintf(stderr,"[vu1:object-test] mismatch case=%u kernel=%u field=%s\n",test,test%4,what);
        }
    }
    std::fprintf(stderr,"[vu1:object-test] cases=8192 failures=%u\n",failures);
    std::fprintf(stderr,"[vu1:object-test] fast=%llu/%llu/%llu clip=%llu fallback=%llu/%llu/%llu\n",g_ps2xVu1ObjectHits[0],g_ps2xVu1ObjectHits[1],g_ps2xVu1ObjectHits[2],g_ps2xVu1ObjectHits[3],g_ps2xVu1ObjectFallbacks[0],g_ps2xVu1ObjectFallbacks[1],g_ps2xVu1ObjectFallbacks[2]);
    if(failures)std::abort();
}

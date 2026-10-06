// Presentation-thread owned GPU history. One simulation-frame delay permits
// interpolation without prediction, CPU readback, or faster physics.
bool interpolationEnabled=false, interpolationValid=false;
Texture interpolationCurrent,interpolationPrevious,interpolationFlow;
uint64_t interpolationFrame=0,interpolationEpoch=0,interpolationPairs=0,interpolationPresents=0;
uint32_t interpolationVisibleWidth=0,interpolationVisibleHeight=0;
std::chrono::steady_clock::time_point interpolationArrival{};
double interpolationInterval=1.0/60.0;
void copyInterpolationTexture(Texture& destination,Texture& source,UINT slot) {
  if(!destination.resource || destination.w!=source.w || destination.h!=source.h ||
     destination.resource->GetDesc().Format!=source.resource->GetDesc().Format) {
    destination={};destination.w=source.w;destination.h=source.h;destination.srv=slot;
    destination.state=D3D12_RESOURCE_STATE_COPY_DEST;
    auto d=source.resource->GetDesc();d.Flags=D3D12_RESOURCE_FLAG_NONE;
    D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    check(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&d,destination.state,nullptr,IID_PPV_ARGS(&destination.resource)),"Interpolation history");
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};view.Format=d.Format;view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;view.Texture2D.MipLevels=1;
    device->CreateShaderResourceView(destination.resource.Get(),&view,cpu(srvs.Get(),slot,srvStep));
  }
  barrier(source,D3D12_RESOURCE_STATE_COPY_SOURCE);barrier(destination,D3D12_RESOURCE_STATE_COPY_DEST);
  commands->CopyResource(destination.resource.Get(),source.resource.Get());
  barrier(destination,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}
void captureInterpolation(wotm_capture::Frame& frame,uint32_t vw,uint32_t vh) {
#ifndef WOTM_D3D12_REPLAY
  if(!interpolationEnabled)return;
  const uint64_t id=g_ps2xGsPresentedGameFrame.load(std::memory_order_acquire);
  auto color=colors.find(frame.display),flow=motionTargets.find(frame.display);
  const bool gameplay=g_ps2xWotmPhase.load(std::memory_order_relaxed)==2u && !g_ps2xWotmPaused.load(std::memory_order_acquire);
  if(!id || !gameplay || color==colors.end() || flow==motionTargets.end() || !frame.motionResetSerial) {
    interpolationValid=false;interpolationFrame=0;return;
  }
  if(id==interpolationFrame)return;
  const auto now=std::chrono::steady_clock::now();
  const double elapsed=std::chrono::duration<double>(now-interpolationArrival).count();
  interpolationValid=interpolationCurrent.resource && interpolationFrame+1==id &&
    interpolationEpoch==frame.motionResetSerial && interpolationVisibleWidth==vw && interpolationVisibleHeight==vh &&
    elapsed>=1.0/240.0 && elapsed<=0.080;
  std::swap(interpolationCurrent,interpolationPrevious);
  const UINT colorSlot=interpolationCurrent.resource?interpolationCurrent.srv:
      (interpolationPrevious.resource && interpolationPrevious.srv==40u?41u:40u);
  copyInterpolationTexture(interpolationCurrent,color->second,colorSlot);
  copyInterpolationTexture(interpolationFlow,flow->second,42u);
  if(interpolationValid) {interpolationInterval=interpolationPairs?interpolationInterval*0.875+elapsed*0.125:elapsed;++interpolationPairs;}
  interpolationFrame=id;interpolationEpoch=frame.motionResetSerial;interpolationArrival=now;
  interpolationVisibleWidth=vw;interpolationVisibleHeight=vh;
  if(id%300u==0)std::fprintf(stderr,"[present:interpolation] completed=%llu pairs=%llu interpolated-presents=%llu valid=%u interval=%.3fms\n",
      id,interpolationPairs,interpolationPresents,unsigned(interpolationValid),interpolationInterval*1000.0);
#else
  (void)frame;(void)vw;(void)vh;
#endif
}

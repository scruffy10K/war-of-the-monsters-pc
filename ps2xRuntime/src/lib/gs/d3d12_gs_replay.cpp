// Live Direct3D12 GS renderer and differential replay tool. No game header
// changes.
#include "d3d12_presenter.cpp"
#include "gs_draw_capture.inc"
#include "gs_vertex_upload.inc"
#include <algorithm>
#include <array>
#include <cmath>
#include <d3dcompiler.h>
#include <fstream>
#include <map>
#include <string>
#include <atomic>
#include <chrono>
#ifndef WOTM_D3D12_REPLAY
extern std::atomic<bool> g_ps2xWotmWideActive,g_ps2xWotmPaused;
extern std::atomic<uint32_t> g_ps2xWotmPhase;
extern std::atomic<uint64_t> g_ps2xGsPresentedGameFrame;
#endif
#pragma comment(lib, "d3dcompiler.lib")
namespace wotm::d3d12 {
struct Constants {
  float target[2], texSize[2];
  uint32_t flags[4], tests[4];
  float alphaFog[4], fogColor[4];
  int32_t wrap[4];
  float region[4], dimx[16],motionInfo[4];
};
static_assert(sizeof(Constants) == 192);
// Retail cinematic fades are untextured, constant-color, near-plane rectangles.
// They change scene exposure, not its geometry; preserve underlying motion.
static bool screenFade(const wotm_capture::Frame& f,const wotm_capture::Draw& d,float fieldHeight) {
  if(d.clear || d.count!=6 || !d.abe || (d.alpha&255u)!=0x44 || d.texture!=~0u || d.textureFbp!=~0u ||
     d.sx!=0 || d.sy!=0 || d.sw!=640 || d.sh!=int(fieldHeight))return false;
  const auto& a=f.vertexData()[d.first];unsigned corners=0;
  for(size_t i=d.first;i<size_t(d.first)+6;++i){const auto& v=f.vertexData()[i];
    if(v.z<0.9999f || v.r!=a.r || v.g!=a.g || v.b!=a.b || v.a!=a.a ||
       (v.x!=0 && v.x!=640) || (v.y!=0 && v.y!=fieldHeight))return false;
    corners|=1u<<((v.x==640?1:0)+(v.y==fieldHeight?2:0));
  }
  return corners==15;
}
class GsReplay {
  struct Texture {
    ComPtr<ID3D12Resource> resource;
    UINT srv = 0, w = 0, h = 0, mips = 1;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST;
  };
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> commands;
  ComPtr<ID3D12Fence> fence;
  HANDLE event = nullptr;
  UINT64 serial = 0;
  ComPtr<ID3D12DescriptorHeap> srvs, rtvs, dsvs, samplers;
  UINT srvStep = 0, rtvStep = 0, dsvStep = 0, samplerStep = 0, nextSrv = 0;
  ComPtr<ID3D12RootSignature> root;
  ComPtr<ID3DBlob> vs, ps, motionVs, motionPs, fxaaPs, interpolationPs;
  bool motionEnabled=false;
  bool fxaaEnabled=false;
  ComPtr<ID3D12DescriptorHeap> motionRtvs;
  std::map<uint32_t,Texture> motionTargets;
  uint64_t motionFrames=0;
  std::vector<ComPtr<ID3D12Resource>> uploads;
  // Recycled only after finish() has completed the preceding submission.
  struct FreeUpload { ComPtr<ID3D12Resource> resource; uint64_t retired; };
  std::multimap<size_t, FreeUpload> freeUploads;
  uint64_t uploadRetirement=0, uploadEvictions=0;
  bool recycleRecentUploads = [] {const char* p=std::getenv("PS2X_D3D12_UPLOAD_RECENCY");return !p || p[0]!='0';}();
  size_t freeUploadBytes = 0;
  uint64_t uploadAllocations = 0, uploadReuses = 0;
  bool reuseUploads = [] { const char* v=std::getenv("PS2X_D3D12_UPLOAD_REUSE"); return !v || v[0]!='0'; }();
  std::vector<Texture> textures, feedbackCopies;
  std::map<uint32_t, Texture> colors, depths;
  Texture white;
  std::map<uint32_t, UINT> colorSlots, depthSlots;
  std::map<std::string, ComPtr<ID3D12PipelineState>> pipelines;
  std::map<std::string, UINT> samplerSlots;
  std::vector<uint8_t> constants;
  UINT width = 0, height = 0, scale = 1;
  bool recording = true;
  // Keep a scene and its immediately following display draw in one recording.
  // Every allocator/descriptor/upload reuse still follows a completed fence.
  // Batching is opt-in: verified equivalent, without a proven live FPS gain.
  bool batchLivePresent = [] {const char* p=std::getenv("PS2X_D3D12_BATCH_PRESENT");return p && p[0]=='1';}();
  bool pendingLiveScene=false;
  uint64_t liveSceneCount=0,batchedPresentCount=0,flushedSceneCount=0;
  ComPtr<IDXGISwapChain3> liveSwap;
  UINT swapWidth=0, swapHeight=0;
  bool liveTearing=false, presentationSuspended=false, liveOccluded=false;
  Texture movie;
  std::map<uint32_t,uint32_t> livePassHeights;
  uint32_t liveDisplay=0, visibleWidth=640, visibleHeight=448;
  bool movieVisible=false;
  struct CachedTexture { Texture texture; uint64_t used=0; size_t bytes=0; };
  std::map<std::shared_ptr<const void>, CachedTexture, std::owner_less<std::shared_ptr<const void>>> liveTextures;
  std::vector<UINT> freeTextureSlots;
  UINT nextTextureSlot=64;
  size_t cacheBytes=0;
  std::map<uint32_t,uint64_t> motionGenerations;
  uint64_t motionResetSerial=0;
  Texture& motionTarget(uint32_t id) {
    auto found=motionTargets.find(id);if(found!=motionTargets.end())return found->second;
    Texture t;t.w=width;t.h=height;t.state=D3D12_RESOURCE_STATE_RENDER_TARGET;
    D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=width;d.Height=height;
    d.DepthOrArraySize=1;d.MipLevels=1;d.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;d.SampleDesc.Count=1;d.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    D3D12_CLEAR_VALUE clear{};clear.Format=d.Format;
    check(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&d,t.state,&clear,IID_PPV_ARGS(&t.resource)),"Create motion target");
    device->CreateRenderTargetView(t.resource.Get(),nullptr,cpu(motionRtvs.Get(),colorSlots.at(id),rtvStep));
    const float zero[4]={};
    commands->ClearRenderTargetView(cpu(motionRtvs.Get(),colorSlots.at(id),rtvStep),zero,0,nullptr);
    return motionTargets.emplace(id,std::move(t)).first->second;
  }
  ComPtr<ID3D12Resource> motionReadback;
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT motionFootprint{};
  std::string motionCapturePath;
  uint64_t motionLastCapture=0;
  unsigned motionCaptureCount=0;
  void captureMotion(const wotm_capture::Frame& frame) {
    if(!motionEnabled)return;
    ++motionFrames;
    size_t validVertices=0,onscreen=0;for(size_t i=0;i<frame.motion.size();++i)if(frame.motion[i][3]>0){++validVertices;const auto& v=frame.vertexData()[i];onscreen+=v.x>=0 && v.x<=640 && v.y>=0 && v.y<=224;}
    if(validVertices && motionFrames%120==0)std::fprintf(stderr,"[motion:screen] candidates=%zu onscreen=%zu display=%u\n",validVertices,onscreen,frame.display);
    if(motionFrames%120==0)std::fprintf(stderr,"[motion:gpu] frame=%llu vertices=%zu candidates=%zu\n",motionFrames,frame.vertexCount(),validVertices);
    const char* path=std::getenv("PS2X_D3D12_MOTION_CAPTURE");
    static const uint64_t firstCapture=[] {const char* p=std::getenv("PS2X_D3D12_MOTION_CAPTURE_START");return p?std::strtoull(p,nullptr,10):0ull;}();
    static const uint64_t captureInterval=[] {const char* p=std::getenv("PS2X_D3D12_MOTION_CAPTURE_INTERVAL");return p?std::max(1ull,std::strtoull(p,nullptr,10)):120ull;}();
    if(motionFrames<firstCapture)return;
    if(!path || !*path || !validVertices || motionCaptureCount>=8 || (motionLastCapture && motionFrames<motionLastCapture+captureInterval))return;
    if(!motionLastCapture){
      auto snapshot=frame;
      for(auto& texture:snapshot.textures)for(auto& level:texture.levels)if(level.borrowed){level.bytes.assign(level.borrowed,level.borrowed+size_t(level.width)*level.height*4);level.borrowed=nullptr;}
      wotm_capture::File file((std::string(path)+".snapshot.wdr").c_str(),false);file.frame(snapshot);
    }
    auto found=motionTargets.find(frame.display);if(found==motionTargets.end())return;
    auto& t=found->second;barrier(t,D3D12_RESOURCE_STATE_COPY_SOURCE);
    auto desc=t.resource->GetDesc();UINT64 bytes=0;
    device->GetCopyableFootprints(&desc,0,1,0,&motionFootprint,nullptr,nullptr,&bytes);
    motionReadback=buffer(size_t(bytes),D3D12_HEAP_TYPE_READBACK);
    D3D12_TEXTURE_COPY_LOCATION src{},dst{};src.pResource=t.resource.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource=motionReadback.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=motionFootprint;
    commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    motionCapturePath=std::string(path)+"."+std::to_string(motionFrames);
    motionLastCapture=motionFrames;++motionCaptureCount;
  }
  void saveMotion() {
    if(!motionReadback)return;
    uint8_t* bytes=nullptr;D3D12_RANGE read{0,size_t(motionFootprint.Footprint.RowPitch)*height};
    check(motionReadback->Map(0,&read,reinterpret_cast<void**>(&bytes)),"Map motion capture");
    std::ofstream raw(motionCapturePath+".f32",std::ios::binary),image(motionCapturePath+".ppm",std::ios::binary);
    image<<"P6\n"<<width<<" "<<height<<"\n255\n";
    size_t valid=0,moving=0,nonfinite=0;
    for(UINT y=0;y<height;++y){
      const float* row=reinterpret_cast<const float*>(bytes+motionFootprint.Offset+size_t(y)*motionFootprint.Footprint.RowPitch);
      raw.write(reinterpret_cast<const char*>(row),size_t(width)*16);
      for(UINT x=0;x<width;++x){const float* p=row+x*4;uint8_t rgb[3]{};
        for(int k=0;k<4;++k)nonfinite+=!std::isfinite(p[k]);
        if(p[2]>0.5f && std::isfinite(p[0]) && std::isfinite(p[1])){++valid;moving+=std::abs(p[0])+std::abs(p[1])>0.05f;
          rgb[0]=uint8_t(std::clamp(128.f+p[0]*8,0.f,255.f));rgb[1]=uint8_t(std::clamp(128.f+p[1]*8,0.f,255.f));rgb[2]=96;}
        image.write(reinterpret_cast<const char*>(rgb),3);
      }
    }
    D3D12_RANGE none{0,0};motionReadback->Unmap(0,&none);motionReadback.Reset();
    if(!raw||!image||nonfinite)throw std::runtime_error("Invalid motion capture or output failure");
    std::fprintf(stderr,"[motion:gpu-capture] %s valid=%zu moving=%zu nonfinite=%zu\n",motionCapturePath.c_str(),valid,moving,nonfinite);
  }
  void beginLive() {
    // On the first paired scene/display, serial is still zero. Do not rewind
    // live descriptors that the unsubmitted scene already references.
    if(recording) { if(!serial && !nextSrv) nextSrv=8192; return; }
    check(allocator->Reset(), "Live allocator reset");
    check(commands->Reset(allocator.Get(), nullptr), "Live command reset");
    if (reuseUploads) for (auto& resource : uploads) {
      constexpr size_t limit=128u*1024u*1024u;
      const size_t bytes = size_t(resource->GetDesc().Width);
      if(bytes>limit)continue;
      // Texture-loading staging buffers must not permanently occupy the pool.
      // All entries and this recording are complete before this retirement.
      if(recycleRecentUploads)while(freeUploads.size()>=64 || freeUploadBytes>limit-bytes) {
        auto oldest=std::min_element(freeUploads.begin(),freeUploads.end(),
          [](const auto& a,const auto& b){return a.second.retired<b.second.retired;});
        freeUploadBytes-=oldest->first;freeUploads.erase(oldest);++uploadEvictions;
      }
      if (freeUploads.size() < 64 && bytes <= limit - freeUploadBytes) {
        freeUploads.emplace(bytes, FreeUpload{std::move(resource),++uploadRetirement});
        freeUploadBytes += bytes;
      }
    }
    uploads.clear(); textures.clear(); feedbackCopies.clear();
    nextSrv=8192; recording=true;
    // Bound retained immutable decoded textures. All previous commands are complete here.
    if (cacheBytes>256u*1024u*1024u || nextTextureSlot>=4096)
    for(auto it=liveTextures.begin();it!=liveTextures.end();) {
      if ((cacheBytes>256u*1024u*1024u || nextTextureSlot>=4096) && it->second.used+2<serial) {
        cacheBytes-=it->second.bytes;freeTextureSlots.push_back(it->second.texture.srv);it=liveTextures.erase(it);
      } else ++it;
    }
  }
  void liveTarget(uint32_t id, bool depth) {
    auto& map=depth?depths:colors;auto& slots=depth?depthSlots:colorSlots;
    if(map.count(id))return;
    if(map.size()>=30)throw std::runtime_error("Live target limit reached");
    UINT slot=UINT(map.size());UINT saved=nextSrv;nextSrv=slot;
    wotm_capture::Image blank{id,width,height,{}};blank.bytes.resize(size_t(width)*height*4);
    Texture t=texture({blank},!depth,depth);nextSrv=saved;
    if(depth)device->CreateDepthStencilView(t.resource.Get(),nullptr,cpu(dsvs.Get(),slot,dsvStep));
    else device->CreateRenderTargetView(t.resource.Get(),nullptr,cpu(rtvs.Get(),slot,rtvStep));
    slots[id]=slot;map.emplace(id,std::move(t));
  }
  Texture liveTexture(const wotm_capture::Texture& input) {
    if(input.owner) {
      auto found=liveTextures.find(input.owner);
      if(found!=liveTextures.end()){found->second.used=serial;return found->second.texture;}
      UINT slot;
      if(!freeTextureSlots.empty()){slot=freeTextureSlots.back();freeTextureSlots.pop_back();}
      else {if(nextTextureSlot>=8192)throw std::runtime_error("Live texture cache full");slot=nextTextureSlot++;}
      UINT saved=nextSrv;nextSrv=slot;Texture t=texture(input.levels);nextSrv=saved;
      size_t bytes=0;for(const auto& l:input.levels)bytes+=size_t(l.width)*l.height*4;
      cacheBytes+=bytes;liveTextures.emplace(input.owner,CachedTexture{t,serial,bytes});return t;
    }
    if(nextSrv>=65536)throw std::runtime_error("Live transient texture limit");
    return texture(input.levels);
  }
  void waitQueue() {
    // Present queues work after the render submission. A separate fence also
    // drains that work before ResizeBuffers releases swap-chain resources.
    check(queue->Signal(fence.Get(), ++serial), "GS GPU Signal");
    const UINT64 completed=fence->GetCompletedValue();
    if(completed==UINT64_MAX)check(device->GetDeviceRemovedReason(), "GS device removed");
    if(completed<serial) {
      check(fence->SetEventOnCompletion(serial, event), "GS GPU fence event");
      if(WaitForSingleObject(event, 10000)!=WAIT_OBJECT_0) {
        check(device->GetDeviceRemovedReason(), "GS device removed during wait");
        throw std::runtime_error("GS GPU wait failed or timed out");
      }
    }
    check(device->GetDeviceRemovedReason(), "GS device");
  }
  void finish() {
    check(commands->Close(), "Close GS commands");
    ID3D12CommandList *lists[] = {commands.Get()};
    queue->ExecuteCommandLists(1, lists);
    waitQueue();
    recording=false;
  }
  void completeLiveScene() {
    if(!pendingLiveScene)return;
    finish();pendingLiveScene=false;++flushedSceneCount;
    saveMotion();debugMessages();
  }
  D3D12_CPU_DESCRIPTOR_HANDLE cpu(ID3D12DescriptorHeap *heap, UINT index,
                                  UINT step) {
    auto h = heap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += SIZE_T(index) * step;
    return h;
  }
  D3D12_GPU_DESCRIPTOR_HANDLE gpu(ID3D12DescriptorHeap *heap, UINT index,
                                  UINT step) {
    auto h = heap->GetGPUDescriptorHandleForHeapStart();
    h.ptr += UINT64(index) * step;
    return h;
  }
  void barrier(Texture &t, D3D12_RESOURCE_STATES state) {
    if (t.state == state)
      return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {t.resource.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    t.state, state};
    commands->ResourceBarrier(1, &b);
    t.state = state;
  }
  ComPtr<ID3D12Resource> buffer(size_t bytes, D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES h{};
    h.Type = type;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = std::max<size_t>(bytes, 256);
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> r;
    check(device->CreateCommittedResource(
              &h, D3D12_HEAP_FLAG_NONE, &d,
              type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ
                                             : D3D12_RESOURCE_STATE_COPY_DEST,
              nullptr, IID_PPV_ARGS(&r)),
          "Create buffer");
    return r;
  }
  ComPtr<ID3D12Resource> acquireUpload(size_t bytes) {
    const size_t capacity = std::max<size_t>(bytes, 256);
    ComPtr<ID3D12Resource> r;
    auto available = freeUploads.lower_bound(capacity);
    if (available != freeUploads.end() && available->first - capacity <= capacity) {
      freeUploadBytes -= available->first;
      r = std::move(available->second.resource);
      freeUploads.erase(available);
      ++uploadReuses;
    } else {
      r = buffer(bytes, D3D12_HEAP_TYPE_UPLOAD);
      ++uploadAllocations;
    }
    return r;
  }
  ComPtr<ID3D12Resource> upload(const void *data, size_t bytes) {
    auto r=acquireUpload(bytes);
    void *p = nullptr;
    D3D12_RANGE none{0, 0};
    check(r->Map(0, &none, &p), "Map upload");
    if (bytes)
      std::memcpy(p, data, bytes);
    r->Unmap(0, nullptr);
    uploads.push_back(r);
    return r;
  }
  template<class T> ComPtr<ID3D12Resource> uploadTriangles(const T* input,size_t count) {
    const size_t bytes=count*sizeof(T);
    auto r=acquireUpload(bytes);
    void* mapped=nullptr;D3D12_RANGE none{0,0};
    check(r->Map(0,&none,&mapped),"Map triangle upload");
    wotm_capture::copyProvokingVertices(mapped,input,count);
    r->Unmap(0,nullptr);uploads.push_back(r);return r;
  }
  Texture texture(const std::vector<wotm_capture::Image> &levels,
                  bool target = false, bool depth = false) {
    Texture t;
    t.w = levels[0].width;
    t.h = levels[0].height;
    t.mips = UINT(levels.size());
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = t.w;
    d.Height = t.h;
    d.DepthOrArraySize = 1;
    d.MipLevels = UINT16(t.mips);
    d.SampleDesc.Count = 1;
    d.Format = depth ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
    d.Flags = depth    ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
              : target ? D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
                       : D3D12_RESOURCE_FLAG_NONE;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    // Zero is the common GS clear. Arbitrary colours use a shader fill below
    // so they neither mismatch this optimized value nor incur debug warnings.
    D3D12_CLEAR_VALUE optimized{};optimized.Format=d.Format;
    check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d,
                                          D3D12_RESOURCE_STATE_COPY_DEST,
                                          target || depth ? &optimized : nullptr, IID_PPV_ARGS(&t.resource)),
          "Create texture");
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(t.mips);
    UINT64 size = 0;
    device->GetCopyableFootprints(&d, 0, t.mips, 0, footprints.data(), nullptr,
                                  nullptr, &size);
    std::vector<uint8_t> bytes(size);
    for (UINT mip = 0; mip < t.mips; ++mip) {
      const auto &l = levels[mip];
      auto &f = footprints[mip];
      if (l.width != f.Footprint.Width || l.height != f.Footprint.Height)
        throw std::runtime_error("Invalid mip dimensions");
      for (UINT y = 0; y < l.height; ++y)
        std::memcpy(bytes.data() + f.Offset + size_t(y) * f.Footprint.RowPitch,
                    l.data() + size_t(y) * l.width * 4,
                    size_t(l.width) * 4);
    }
    auto staging = upload(bytes.data(), bytes.size());
    for (UINT mip = 0; mip < t.mips; ++mip) {
      D3D12_TEXTURE_COPY_LOCATION src{};
      src.pResource = staging.Get();
      src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      src.PlacedFootprint = footprints[mip];
      D3D12_TEXTURE_COPY_LOCATION dst{};
      dst.pResource = t.resource.Get();
      dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      dst.SubresourceIndex = mip;
      commands->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (!depth) {
      t.srv = nextSrv++;
      D3D12_SHADER_RESOURCE_VIEW_DESC view{};
      view.Format = d.Format;
      view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      view.Texture2D.MipLevels = t.mips;
      device->CreateShaderResourceView(t.resource.Get(), &view,
                                       cpu(srvs.Get(), t.srv, srvStep));
    }
    barrier(t, depth    ? D3D12_RESOURCE_STATE_DEPTH_WRITE
               : target ? D3D12_RESOURCE_STATE_RENDER_TARGET
                        : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    return t;
  }
  UINT sampler(const wotm_capture::Draw &c, const Texture &t,
               bool framebuffer) {
    D3D12_SAMPLER_DESC d{};
    bool minLinear = c.mmin == 1 || c.mmin >= 4, magLinear = c.mmag != 0,
         mipLinear = c.mmin == 3 || c.mmin == 5;
    if (c.transient || framebuffer) {
      minLinear = magLinear = true;
      mipLinear = false;
    }
    d.Filter = D3D12_ENCODE_BASIC_FILTER(
        minLinear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
        magLinear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
        mipLinear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
        D3D12_FILTER_REDUCTION_TYPE_STANDARD);
    d.AddressU = c.repeatU && !c.transient && !framebuffer
                     ? D3D12_TEXTURE_ADDRESS_MODE_WRAP
                     : D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    d.AddressV = c.repeatV && !c.transient && !framebuffer
                     ? D3D12_TEXTURE_ADDRESS_MODE_WRAP
                     : D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    d.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    d.MipLODBias = c.lodBias;
    d.MaxAnisotropy = 1;
    d.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    d.MaxLOD = c.mmin >= 2 ? float(t.mips - 1) : 0;
    std::string key(reinterpret_cast<const char *>(&d), sizeof(d));
    auto found = samplerSlots.find(key);
    if (found != samplerSlots.end())
      return found->second;
    UINT index = UINT(samplerSlots.size());
    if (index >= 2048)
      throw std::runtime_error("Sampler heap exhausted");
    device->CreateSampler(&d, cpu(samplers.Get(), index, samplerStep));
    samplerSlots.emplace(key, index);
    return index;
  }
  ID3D12PipelineState *pipeline(const wotm_capture::Draw &c, bool failPass,
                               bool present = false, bool fxaa = false, bool interpolate=false) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC p{};
    p.pRootSignature = root.Get();
    p.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    p.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    static const D3D12_INPUT_ELEMENT_DESC layout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 28,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"FOG", 0, DXGI_FORMAT_R32_FLOAT, 0, 40,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
    static const D3D12_INPUT_ELEMENT_DESC motionLayout[] = {
        layout[0],layout[1],layout[2],layout[3],
        {"MOTION",0,DXGI_FORMAT_R32G32B32A32_FLOAT,1,0,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
    const bool temporal=motionEnabled && !present;
    p.InputLayout = temporal ? D3D12_INPUT_LAYOUT_DESC{motionLayout,5}:D3D12_INPUT_LAYOUT_DESC{layout,4};
    if(temporal){p.VS={motionVs->GetBufferPointer(),motionVs->GetBufferSize()};p.PS={motionPs->GetBufferPointer(),motionPs->GetBufferSize()};}
    if (present && fxaa) p.PS={fxaaPs->GetBufferPointer(),fxaaPs->GetBufferSize()};
    if(present && interpolate)p.PS={interpolationPs->GetBufferPointer(),interpolationPs->GetBufferSize()};
    p.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    p.NumRenderTargets = temporal?2:1;
    if(temporal)p.RTVFormats[1]=DXGI_FORMAT_R32G32B32A32_FLOAT;
    p.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    p.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    p.SampleDesc.Count = 1;
    p.SampleMask = ~0u;
    p.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    p.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    p.RasterizerState.DepthClipEnable = TRUE;
    auto &ds = p.DepthStencilState;
    ds.DepthEnable = TRUE;
    ds.DepthWriteMask = (!c.zmsk && (!failPass || c.afail == 2))
                            ? D3D12_DEPTH_WRITE_MASK_ALL
                            : D3D12_DEPTH_WRITE_MASK_ZERO;
    uint32_t z = (c.test >> 17) & 3;
    ds.DepthFunc = !(c.test & (1ull << 16)) ? D3D12_COMPARISON_FUNC_ALWAYS
                   : z == 0                 ? D3D12_COMPARISON_FUNC_NEVER
                   : z == 1                 ? D3D12_COMPARISON_FUNC_ALWAYS
                   : z == 2 ? D3D12_COMPARISON_FUNC_GREATER_EQUAL
                            : D3D12_COMPARISON_FUNC_GREATER;
    auto &b = p.BlendState.RenderTarget[0];
    b.BlendEnable = c.abe != 0;
    b.SrcBlend = D3D12_BLEND_ONE;
    b.DestBlend = D3D12_BLEND_ZERO;
    b.BlendOp = D3D12_BLEND_OP_ADD;
    b.SrcBlendAlpha = D3D12_BLEND_ONE;
    b.DestBlendAlpha = D3D12_BLEND_ZERO;
    b.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    b.LogicOp = D3D12_LOGIC_OP_NOOP;
    for (UINT i = 0; i < 4; ++i)
      if (((c.fbmsk >> (i * 8)) & 255) != 255)
        b.RenderTargetWriteMask |= UINT8(1 << i);
    if (failPass)
      b.RenderTargetWriteMask &= c.afail == 2 ? 0 : c.afail == 1 ? 15 : 7;
    if (c.abe) {
      uint32_t a = c.alpha & 3, bop = (c.alpha >> 2) & 3,
               coef = (c.alpha >> 4) & 3, d = (c.alpha >> 6) & 3;
      int cs = 0, os = 0, cd = 0, od = 0;
      auto bump = [&](uint32_t op, int cc, int one) {
        if (op == 0) {
          cs += cc;
          os += one;
        } else if (op == 1) {
          cd += cc;
          od += one;
        }
      };
      bump(a, 1, 0);
      bump(bop, -1, 0);
      bump(d, 0, 1);
      D3D12_BLEND cf = coef == 0   ? D3D12_BLEND_SRC_ALPHA
                       : coef == 1 ? D3D12_BLEND_DEST_ALPHA
                                   : D3D12_BLEND_BLEND_FACTOR;
      D3D12_BLEND inv = coef == 0   ? D3D12_BLEND_INV_SRC_ALPHA
                        : coef == 1 ? D3D12_BLEND_INV_DEST_ALPHA
                                    : D3D12_BLEND_INV_BLEND_FACTOR;
      auto term = [&](int cc, int oc, D3D12_BLEND &out) {
        if (!cc && !oc) {
          out = D3D12_BLEND_ZERO;
          return 1;
        }
        if (!cc) {
          out = D3D12_BLEND_ONE;
          return oc > 0 ? 1 : -1;
        }
        if (!oc) {
          out = cf;
          return cc > 0 ? 1 : -1;
        }
        out = cc < 0 ? inv : D3D12_BLEND_ONE;
        return 1;
      };
      int s = term(cs, os, b.SrcBlend), t = term(cd, od, b.DestBlend);
      if (s < 0 && t > 0)
        b.BlendOp = D3D12_BLEND_OP_REV_SUBTRACT;
      else if (t < 0 && s > 0)
        b.BlendOp = D3D12_BLEND_OP_SUBTRACT;
      else if (t < 0 && s < 0)
        b.SrcBlend = b.DestBlend = D3D12_BLEND_ZERO;
    }
    if(temporal) {
      p.BlendState.IndependentBlendEnable=TRUE;
      auto& m=p.BlendState.RenderTarget[1];m.BlendEnable=TRUE;m.LogicOpEnable=FALSE;
      m.SrcBlend=m.SrcBlendAlpha=D3D12_BLEND_ONE;m.DestBlend=m.DestBlendAlpha=D3D12_BLEND_INV_SRC_ALPHA;
      m.BlendOp=m.BlendOpAlpha=D3D12_BLEND_OP_ADD;m.LogicOp=D3D12_LOGIC_OP_NOOP;
      m.RenderTargetWriteMask=((b.RenderTargetWriteMask&7)==0)?0:7;
    }
    if (present) { p.DSVFormat = DXGI_FORMAT_UNKNOWN; ds.DepthEnable = FALSE; ds.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO; }
    // Pointer fields are identical within one device; the initialized
    // descriptor is a stable local cache key.
    std::string key(reinterpret_cast<const char *>(&p), sizeof(p));
    auto it = pipelines.find(key);
    if (it != pipelines.end())
      return it->second.Get();
    ComPtr<ID3D12PipelineState> state;
    check(device->CreateGraphicsPipelineState(&p, IID_PPV_ARGS(&state)),
          "Create GS pipeline");
    auto *result = state.Get();
    pipelines.emplace(std::move(key), std::move(state));
    return result;
  }
  void debugMessages() {
    ComPtr<ID3D12InfoQueue> info;
    if (FAILED(device.As(&info)))
      return;
    bool bad = false;
    for (UINT64 i = 0; i < info->GetNumStoredMessagesAllowedByRetrievalFilter();
         ++i) {
      SIZE_T n = 0;
      check(info->GetMessage(i, nullptr, &n), "Message size");
      std::vector<uint8_t> b(n);
      auto *m = reinterpret_cast<D3D12_MESSAGE *>(b.data());
      check(info->GetMessage(i, m, &n), "Message");
      if (m->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
        std::fprintf(stderr, "D3D12: %s\n", m->pDescription);
        bad = true;
      }
    }
    info->ClearStoredMessages();
    if (bad)
      throw std::runtime_error("D3D12 validation warnings/errors");
  }

public:
  ~GsReplay() {
    if(std::getenv("PS2X_D3D12_BATCH_STATS"))
      std::fprintf(stderr,"[d3d12:batch] enabled=%u scenes=%llu paired=%llu separately-flushed=%llu\n",
        unsigned(batchLivePresent),liveSceneCount,batchedPresentCount,flushedSceneCount);
    if (std::getenv("PS2X_D3D12_UPLOAD_STATS"))
      std::fprintf(stderr,"[d3d12:uploads] allocations=%llu reuses=%llu retained=%zu bytes=%zu evictions=%llu recent=%u\n",
        static_cast<unsigned long long>(uploadAllocations),
        static_cast<unsigned long long>(uploadReuses),freeUploads.size(),freeUploadBytes,
        static_cast<unsigned long long>(uploadEvictions),unsigned(recycleRecentUploads));
    // DXGI Present queues work after the draw fence. Drain that work too before
    // releasing the swap chain or resources on shutdown.
    if(queue && fence && event && SUCCEEDED(queue->Signal(fence.Get(),++serial)) &&
       SUCCEEDED(fence->SetEventOnCompletion(serial,event))) WaitForSingleObject(event,10000);
    // Discard any unsubmitted recording before releasing resources it references.
    commands.Reset();
    if (event)
      CloseHandle(event);
  }
  void initialize(const wchar_t *shader, bool validation = true) {
    ComPtr<ID3D12Debug> debug;
    if (validation) { check(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)), "Get debug layer"); debug->EnableDebugLayer(); }
    ComPtr<IDXGIFactory6> factory;
    check(CreateDXGIFactory2(validation ? DXGI_CREATE_FACTORY_DEBUG : 0, IID_PPV_ARGS(&factory)),
          "Factory");
    for (UINT i = 0;; ++i) {
      ComPtr<IDXGIAdapter1> a;
      HRESULT hr = factory->EnumAdapterByGpuPreference(
          i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&a));
      if (hr == DXGI_ERROR_NOT_FOUND)
        break;
      check(hr, "Adapter");
      DXGI_ADAPTER_DESC1 d{};
      check(a->GetDesc1(&d), "Adapter description");
      if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
          SUCCEEDED(D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_11_0,
                                      IID_PPV_ARGS(&device)))) {
        std::printf("GS replay adapter: %ls\n", d.Description);
        break;
      }
    }
    if (!device)
      throw std::runtime_error("No D3D12 hardware device");
    if(validation) {
      ComPtr<ID3D12InfoQueue1> info;
      if(SUCCEEDED(device.As(&info))) {DWORD cookie=0;
        check(info->RegisterMessageCallback([](D3D12_MESSAGE_CATEGORY,D3D12_MESSAGE_SEVERITY severity,D3D12_MESSAGE_ID, LPCSTR text,void*) {
          if(severity<=D3D12_MESSAGE_SEVERITY_WARNING)std::fprintf(stderr,"[d3d12:validation] %s\n",text);
        },D3D12_MESSAGE_CALLBACK_FLAG_NONE,nullptr,&cookie),"Debug callback");
      }
    }
    D3D12_COMMAND_QUEUE_DESC q{};
    check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)), "Queue");
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                         IID_PPV_ARGS(&allocator)),
          "Allocator");
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                    allocator.Get(), nullptr,
                                    IID_PPV_ARGS(&commands)),
          "Commands");
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)),
          "Fence");
    event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event)
      throw std::runtime_error("Fence event failed");
    auto heap = [&](D3D12_DESCRIPTOR_HEAP_TYPE type, UINT n, bool visible,
                    ComPtr<ID3D12DescriptorHeap> &out) {
      D3D12_DESCRIPTOR_HEAP_DESC h{};
      h.Type = type;
      h.NumDescriptors = n;
      h.Flags = visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
                        : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
      check(device->CreateDescriptorHeap(&h, IID_PPV_ARGS(&out)), "Heap");
      return device->GetDescriptorHandleIncrementSize(type);
    };
    srvStep = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 65536, true, srvs);
    rtvStep = heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 32, false, rtvs);
    dsvStep = heap(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 32, false, dsvs);
    samplerStep =
        heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2048, true, samplers);
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0};
    ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].Descriptor = {0, 0};
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (UINT i = 1; i < 3; ++i) {
      params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      params[i].DescriptorTable = {1, &ranges[i - 1]};
      params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    }
    D3D12_ROOT_SIGNATURE_DESC desc{
        3, params, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
    ComPtr<ID3DBlob> serialized, errors;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                      &serialized, &errors),
          "Root serialize");
    check(device->CreateRootSignature(0, serialized->GetBufferPointer(),
                                      serialized->GetBufferSize(),
                                      IID_PPV_ARGS(&root)),
          "Root signature");
    auto compile = [&](const char *entry, const char *target,
                       ComPtr<ID3DBlob> &out, bool motion=false) {
      const D3D_SHADER_MACRO defines[]={{"MOTION_OUTPUT","1"},{nullptr,nullptr}};
      HRESULT hr = D3DCompileFromFile(shader, motion?defines:nullptr, nullptr, entry, target,
                                      D3DCOMPILE_ENABLE_STRICTNESS |
                                          D3DCOMPILE_OPTIMIZATION_LEVEL3,
                                      0, &out, &errors);
      if (FAILED(hr) && errors)
        std::fprintf(stderr, "%s\n",
                     static_cast<char *>(errors->GetBufferPointer()));
      check(hr, "Compile HLSL");
    };
    compile("VSMain", "vs_5_1", vs);
    compile("PSMain", "ps_5_1", ps);
    const char* aaEnv=std::getenv("PS2X_AA");
    fxaaEnabled=aaEnv && std::strcmp(aaEnv,"fxaa")==0;
    if(fxaaEnabled)compile("PSFxAA","ps_5_1",fxaaPs);
    const char* interpolationEnv=std::getenv("PS2X_FRAME_INTERPOLATION");interpolationEnabled=interpolationEnv && interpolationEnv[0]=='1';
    if(interpolationEnabled)compile("PSInterpolation","ps_5_1",interpolationPs);
    const char* motionEnv=std::getenv("PS2X_D3D12_MOTION");motionEnabled=interpolationEnabled || (motionEnv && motionEnv[0]=='1');
    if(motionEnabled){
      compile("VSMain","vs_5_1",motionVs,true);compile("PSMain","ps_5_1",motionPs,true);
      D3D12_DESCRIPTOR_HEAP_DESC h{};h.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV;h.NumDescriptors=32;
      check(device->CreateDescriptorHeap(&h,IID_PPV_ARGS(&motionRtvs)),"Motion RTV heap");
    }
  }
  void clearColorTarget(D3D12_CPU_DESCRIPTOR_HANDLE rtv,const D3D12_RECT& area,const float* rgba) {
    if(rgba[0]==0 && rgba[1]==0 && rgba[2]==0 && rgba[3]==0) {
      commands->ClearRenderTargetView(rtv,rgba,1,&area);return;
    }
    // Draw a constant-colour rectangle for nonzero clears. This also works on
    // drivers that cannot optimize that colour. No depth or blending writes.
    wotm_capture::Vertex quad[6]{};
    const float xy[6][2]={{0,0},{float(width),0},{float(width),float(height)},
      {0,0},{float(width),float(height)},{0,float(height)}};
    for(unsigned i=0;i<6;++i) {
      quad[i].x=xy[i][0];quad[i].y=xy[i][1];quad[i].q=1;
      quad[i].r=rgba[0];quad[i].g=rgba[1];quad[i].b=rgba[2];quad[i].a=rgba[3];
    }
    auto vertices=upload(quad,sizeof(quad));
    D3D12_VERTEX_BUFFER_VIEW view{vertices->GetGPUVirtualAddress(),sizeof(quad),sizeof(quad[0])};
    Constants k{};k.target[0]=float(width);k.target[1]=float(height);k.flags[3]=1;
    auto cb=upload(&k,sizeof(k));wotm_capture::Draw d{};d.zmsk=1;
    commands->OMSetRenderTargets(1,&rtv,FALSE,nullptr);
    commands->SetGraphicsRootConstantBufferView(0,cb->GetGPUVirtualAddress());
    commands->SetGraphicsRootDescriptorTable(1,gpu(srvs.Get(),white.srv,srvStep));
    commands->SetGraphicsRootDescriptorTable(2,gpu(samplers.Get(),sampler(d,white,false),samplerStep));
    commands->SetPipelineState(pipeline(d,false,true));
    commands->IASetVertexBuffers(0,1,&view);commands->DrawInstanced(6,1,0,0);
  }
  void draw(wotm_capture::Frame& frame) {
    const size_t vertexCount=frame.vertexCount();
    if(vertexCount>UINT_MAX/sizeof(wotm_capture::Vertex))throw std::runtime_error("Vertex upload exceeds buffer view size");
    for(const auto& d:frame.draws)if(!d.clear && (size_t(d.first)+d.count>vertexCount || d.count%3))
      throw std::runtime_error("Draw packet exceeds its vertex range or is not triangular");
    // D3D flat shading uses the first vertex, GL uses the last; rotate each
    // triangle.
    static const bool directUpload=[] {const char* p=std::getenv("PS2X_D3D12_DIRECT_UPLOAD");return !p || p[0]!='0';}();
    ComPtr<ID3D12Resource> vb;
    if(directUpload)vb=uploadTriangles(frame.vertexData(),vertexCount);
    else {
      std::vector<wotm_capture::Vertex> vertices;
      if(vertexCount)vertices.assign(frame.vertexData(),frame.vertexData()+vertexCount);
      for(size_t i=0;i+2<vertices.size();i+=3)std::rotate(vertices.begin()+i,vertices.begin()+i+2,vertices.begin()+i+3);
      vb=upload(vertices.data(),vertices.size()*sizeof(vertices[0]));
    }
    D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(),
                                 UINT(vertexCount * sizeof(wotm_capture::Vertex)),
                                 sizeof(wotm_capture::Vertex)};
    if(motionEnabled){
      if(!frame.motion.empty() && frame.motion.size()!=vertexCount)throw std::runtime_error("Motion/vertex count mismatch");
      ComPtr<ID3D12Resource> mb;
      if(directUpload)mb=uploadTriangles(frame.motion.empty()?nullptr:frame.motion.data(),vertexCount);
      else {
        auto motion=frame.motion;
        if(motion.empty())motion.resize(vertexCount);
        for(size_t i=0;i+2<motion.size();i+=3)std::rotate(motion.begin()+i,motion.begin()+i+2,motion.begin()+i+3);
        mb=upload(motion.data(),motion.size()*sizeof(motion[0]));
      }
      D3D12_VERTEX_BUFFER_VIEW mv{mb->GetGPUVirtualAddress(),UINT(vertexCount*sizeof(frame.motion[0])),sizeof(frame.motion[0])};
      commands->IASetVertexBuffers(1,1,&mv);
    }
    constants.resize(frame.draws.size() * 2 * 256);
    auto cb = upload(constants.data(), constants.size());
    uint8_t *mapped = nullptr;
    D3D12_RANGE none{0, 0};
    check(cb->Map(0, &none, reinterpret_cast<void **>(&mapped)),
          "Map draw constants");
    std::map<uint32_t, uint32_t> heights;
    for (auto &c : frame.draws)
      heights[c.fbp] =
          std::max(heights[c.fbp], uint32_t(std::max(0, c.sy + c.sh)));
    // v3/live packets retain the target extent across cropped-only updates.
    // v1/v2 captures keep their historical inference for compatibility.
    for (const auto& h : frame.heights) {
      if (!h.height || h.height > 2048u)
        throw std::runtime_error("Invalid recorded target height");
      heights[h.fbp] = h.height;
    }
    ID3D12DescriptorHeap *heaps[] = {srvs.Get(), samplers.Get()};
    commands->SetDescriptorHeaps(2, heaps);
    commands->SetGraphicsRootSignature(root.Get());
    commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    commands->IASetVertexBuffers(0, 1, &vbv);
    D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
    commands->RSSetViewports(1, &viewport);
    if(!frame.motionGenerations.empty() && frame.motionGenerations.size()!=frame.draws.size())throw std::runtime_error("Motion generation/draw count mismatch");
    if(motionEnabled && frame.motionGenerations.empty()){
      const float zero[4]={};
      std::map<uint32_t,bool> touched;
      for(const auto& d:frame.draws)if(touched.emplace(d.fbp,true).second){auto& motion=motionTarget(d.fbp);barrier(motion,D3D12_RESOURCE_STATE_RENDER_TARGET);commands->ClearRenderTargetView(cpu(motionRtvs.Get(),colorSlots.at(d.fbp),rtvStep),zero,0,nullptr);}
    }
    if(!frame.motionKinds.empty() && frame.motionKinds.size()!=frame.draws.size())throw std::runtime_error("Motion kind/draw count mismatch");
    if(motionEnabled && frame.motionResetSerial && frame.motionResetSerial!=motionResetSerial){
      const float zero[4]={};
      for(auto& entry:motionTargets){barrier(entry.second,D3D12_RESOURCE_STATE_RENDER_TARGET);commands->ClearRenderTargetView(cpu(motionRtvs.Get(),colorSlots.at(entry.first),rtvStep),zero,0,nullptr);}
      motionGenerations.clear();motionResetSerial=frame.motionResetSerial;
    }
    size_t drawIndex = 0;
    for (auto &c : frame.draws) {
      if(motionEnabled && !c.clear && !frame.motion.empty()){
        static unsigned logged=0;bool candidate=false;
        for(size_t k=c.first;k<size_t(c.first)+c.count;++k)candidate|=frame.motion[k][3]>0;
        if(candidate && logged++<12)std::fprintf(stderr,"[motion:material] target=%u display=%u zmsk=%u abe=%u fbmsk=%08x test=%llx count=%u\n",c.fbp,frame.display,c.zmsk,c.abe,c.fbmsk,c.test,c.count);
      }
      auto color = colors.find(c.fbp), depth = depths.find(c.zbp);
      if (color == colors.end() || depth == depths.end())
        throw std::runtime_error("Missing target");
      barrier(color->second, D3D12_RESOURCE_STATE_RENDER_TARGET);
      barrier(depth->second, D3D12_RESOURCE_STATE_DEPTH_WRITE);
      auto rtv = cpu(rtvs.Get(), colorSlots.at(c.fbp), rtvStep),
           dsv = cpu(dsvs.Get(), depthSlots.at(c.zbp), dsvStep);
      D3D12_CPU_DESCRIPTOR_HANDLE motionRtv{};
      if(motionEnabled){
        auto& motion=motionTarget(c.fbp);
        barrier(motion,D3D12_RESOURCE_STATE_RENDER_TARGET);
        const uint64_t generation=frame.motionGenerations.empty()?0:frame.motionGenerations[drawIndex];
        if(generation && motionGenerations[c.fbp]!=generation){
          const float zero[4]={};commands->ClearRenderTargetView(cpu(motionRtvs.Get(),colorSlots.at(c.fbp),rtvStep),zero,0,nullptr);
          motionGenerations[c.fbp]=generation;
        }
        motionRtv=cpu(motionRtvs.Get(),colorSlots.at(c.fbp),rtvStep);
        D3D12_CPU_DESCRIPTOR_HANDLE targets[]={rtv,motionRtv};commands->OMSetRenderTargets(2,targets,FALSE,&dsv);
      } else commands->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
      float sy = float(height) / std::max(heights[c.fbp], 1u);
      LONG y = LONG(c.sy * sy), h = LONG(c.sh * sy + 0.5f);
      D3D12_RECT scissor{LONG(c.sx * scale), y, LONG((c.sx + c.sw) * scale),
                         y + h};
      commands->RSSetScissorRects(1, &scissor);
      if (c.clear) {
        clearColorTarget(rtv,scissor,c.clearColor);
        commands->IASetVertexBuffers(0,1,&vbv);
        if(motionEnabled){const float zero[4]={};commands->ClearRenderTargetView(motionRtv,zero,1,&scissor);}
        ++drawIndex;
        continue;
      }
      Texture *t = &white;
      bool textured = false, feedback = false;
      if (c.textureFbp != ~0u) {
        auto found = colors.find(c.textureFbp);
        if (found != colors.end()) {
          t = &found->second;
          textured = feedback = true;
          if (c.textureFbp == c.fbp) {
            Texture copy;
            copy.w = t->w;
            copy.h = t->h;
            copy.srv = nextSrv++;
            if (copy.srv >= 65536)
              throw std::runtime_error("Feedback descriptor heap exhausted");
            auto desc = t->resource->GetDesc();
            desc.Flags = D3D12_RESOURCE_FLAG_NONE;
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            check(device->CreateCommittedResource(
                      &heap, D3D12_HEAP_FLAG_NONE, &desc,
                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                      IID_PPV_ARGS(&copy.resource)),
                  "Feedback snapshot");
            barrier(*t, D3D12_RESOURCE_STATE_COPY_SOURCE);
            commands->CopyResource(copy.resource.Get(), t->resource.Get());
            barrier(*t, D3D12_RESOURCE_STATE_RENDER_TARGET);
            barrier(copy, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format = desc.Format;
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Shader4ComponentMapping =
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            view.Texture2D.MipLevels = 1;
            device->CreateShaderResourceView(
                copy.resource.Get(), &view, cpu(srvs.Get(), copy.srv, srvStep));
            feedbackCopies.push_back(std::move(copy));
            t = &feedbackCopies.back();
          } else
            barrier(*t, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }
      } else if (c.texture != ~0u) {
        t = &textures.at(c.texture);
        textured = true;
      }
      commands->SetGraphicsRootDescriptorTable(
          1, gpu(srvs.Get(), t->srv, srvStep));
      commands->SetGraphicsRootDescriptorTable(
          2, gpu(samplers.Get(), sampler(c, *t, feedback), samplerStep));
      float factor = std::min(float((c.alpha >> 32) & 255) / 128.0f, 1.0f);
      float blend[] = {factor, factor, factor, factor};
      commands->OMSetBlendFactor(blend);
      for (UINT pass = 0; pass < (c.afail ? 2u : 1u); ++pass) {
        Constants k{};
        k.motionInfo[0]=float(width);k.motionInfo[1]=float(height);k.motionInfo[2]=(c.fbmsk&0xffffffu)==0 && !pass ? (!c.abe ? 1.f : ((c.alpha&255u)==0x44 ? 2.f:0.f)) : 0.f;
        k.motionInfo[3]=c.abe && (c.alpha&255u)==0x44u;
        k.target[0] = 640;
        k.target[1] = float(std::max(heights[c.fbp], 1u));
        if(screenFade(frame,c,k.target[1]))k.motionInfo[3]=2;
        // (0-Cs)*As+Cd changes lighting on the existing receiver; it must not
        // replace its surface motion. Require traced blocker ancestry, not blend alone.
        if(!frame.motionKinds.empty() && frame.motionKinds[drawIndex]==1 && c.abe &&
           (c.alpha&255u)==0x42u && c.zmsk && !pass && !c.afail && !c.pabe &&
           (c.fbmsk&0xffffffu)==0 && c.texture!=~0u){
          k.motionInfo[2]=0;k.motionInfo[3]=2;
          static unsigned shadowLogs=0;if(shadowLogs++<4)std::fprintf(stderr,"[motion:shadow] preserving receiver target=%u draw=%zu\n",c.fbp,drawIndex);
        }
        const bool sceneRefraction = feedback && c.tcc == 0u && c.tfx == 0u &&
            c.wrapU == 2 && c.wrapV == 2 && c.region[0] == 1.f && c.region[1] == 638.f &&
            c.region[2] == 1.f && c.region[3] == 222.f;
        k.texSize[0] = sceneRefraction ? 1024.f : float(t->w);
        k.texSize[1] = sceneRefraction ? 256.f : float(t->h);
        k.alphaFog[2] = sceneRefraction ? 1.f : 0.f;
        k.flags[0] = textured;
        k.flags[1] = c.tcc;
        k.flags[2] = c.tfx;
        k.flags[3] = c.iip;
        k.tests[0] = c.alphaTest;
        k.tests[1] = c.atst;
        k.tests[2] = pass;
        k.tests[3] = c.pabe;
        k.alphaFog[0] = c.aref;
        k.alphaFog[1] = float(c.fog);
        k.alphaFog[3] = float(height);
        std::copy_n(c.fogColor, 3, k.fogColor);
        k.wrap[0] = c.wrapU;
        k.wrap[1] = c.wrapV;
        k.wrap[2] = feedback;
        k.wrap[3] = c.dither;
        std::copy_n(c.region, 4, k.region);
        std::copy_n(c.dimx, 16, k.dimx);
        size_t offset = (drawIndex * 2 + pass) * 256;
        std::memcpy(mapped + offset, &k, sizeof(k));
        commands->SetGraphicsRootConstantBufferView(
            0, cb->GetGPUVirtualAddress() + offset);
        commands->SetPipelineState(pipeline(c, pass != 0));
        commands->DrawInstanced(c.count, 1, c.first, 0);
      }
      ++drawIndex;
    }
    cb->Unmap(0, nullptr);
  }
  #include "d3d12_interpolation.inc"
  void liveFrame(wotm_capture::Frame& frame, uint32_t vw, uint32_t vh) {
    // A scene with no display must complete before its resources can be reused.
    completeLiveScene();
    for(const auto& h:frame.heights)livePassHeights[h.fbp]=h.height;
    beginLive();
    wotm_capture::packTransferTiles(frame);
    if(width && (width!=frame.width||height!=frame.height))throw std::runtime_error("Live render scale changed; restart required");
    width=frame.width;height=frame.height;scale=frame.scale;
    for(const auto& c:frame.draws){liveTarget(c.fbp,false);liveTarget(c.zbp,true);}
    if(!white.resource){UINT saved=nextSrv;nextSrv=32;white=texture({{0,1,1,{255,255,255,255}}});nextSrv=saved;}
    for(const auto& t:frame.textures)textures.push_back(liveTexture(t));
    draw(frame);captureMotion(frame);captureInterpolation(frame,vw,vh);
    pendingLiveScene=true;++liveSceneCount;
    if(!batchLivePresent)completeLiveScene();
    liveDisplay=frame.display;visibleWidth=vw;visibleHeight=vh;movieVisible=false;
  }
  void liveMovie(const uint8_t* pixels, uint32_t w, uint32_t h) {
    if(!pixels||!w||!h)return;
    completeLiveScene();
    beginLive();UINT saved=nextSrv;nextSrv=33;
    wotm_capture::Image image{0,w,h,{}};image.borrowed=pixels;
    movie=texture({image});nextSrv=saved;interpolationValid=false;interpolationFrame=0;finish();debugMessages();
    visibleWidth=w;visibleHeight=h;movieVisible=true;
  }
  void presentLive(HWND window, bool vsync) {
    const char* shotBase=std::getenv("PS2X_D3D12_SHOT");
    RECT rect{};if(!GetClientRect(window,&rect))throw std::runtime_error("Get client rectangle failed");
    UINT w=UINT(std::max<LONG>(0,rect.right)),h=UINT(std::max<LONG>(0,rect.bottom));
    if(!w||!h||IsIconic(window)) {
      completeLiveScene();
      presentationSuspended=true;interpolationValid=false;interpolationFrame=0;
      return;
    }
    if(presentationSuspended) {
      presentationSuspended=false;interpolationValid=false;interpolationFrame=0;
    }
    if(!liveSwap) {
      ComPtr<IDXGIFactory4> factory;check(CreateDXGIFactory2(0,IID_PPV_ARGS(&factory)),"Live factory");
      DXGI_SWAP_CHAIN_DESC1 desc{};desc.Width=w;desc.Height=h;desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
      desc.SampleDesc.Count=1;desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;desc.BufferCount=2;
      desc.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
      ComPtr<IDXGIFactory5> factory5;BOOL allowTearing=FALSE;
      if(SUCCEEDED(factory.As(&factory5)) && SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING,&allowTearing,sizeof(allowTearing))))
        liveTearing=allowTearing!=FALSE;
      if(liveTearing)desc.Flags=DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
      std::fprintf(stderr,"[present:swapchain] unsynchronized presentation support=%u\n",unsigned(liveTearing));
      ComPtr<IDXGISwapChain1> first;
      check(factory->CreateSwapChainForHwnd(queue.Get(),window,&desc,nullptr,nullptr,&first),"Live swap chain");
      check(factory->MakeWindowAssociation(window,DXGI_MWA_NO_ALT_ENTER),"Live window association");
      check(first.As(&liveSwap),"Live swap interface");swapWidth=w;swapHeight=h;
    } else if(w!=swapWidth||h!=swapHeight) {
      completeLiveScene();
      waitQueue();
      check(liveSwap->ResizeBuffers(2,w,h,DXGI_FORMAT_R8G8B8A8_UNORM,liveTearing?DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING:0),"Live resize");
      swapWidth=w;swapHeight=h;interpolationValid=false;interpolationFrame=0;
    }
    // Explicit screenshot diagnostics still render in a hidden test window.
    if(liveOccluded && (!shotBase || !*shotBase)) {
      const HRESULT visibility=liveSwap->Present(0,DXGI_PRESENT_TEST);
      check(visibility,"Live visibility test");
      if(visibility==DXGI_STATUS_OCCLUDED){completeLiveScene();return;}
      liveOccluded=false;interpolationValid=false;interpolationFrame=0;
    }
    if(pendingLiveScene)++batchedPresentCount;
    beginLive();
    ComPtr<ID3D12Resource> back;check(liveSwap->GetBuffer(liveSwap->GetCurrentBackBufferIndex(),IID_PPV_ARGS(&back)),"Live back buffer");
    D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition={back.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_PRESENT,D3D12_RESOURCE_STATE_RENDER_TARGET};
    commands->ResourceBarrier(1,&b);
    auto rtv=cpu(rtvs.Get(),31,rtvStep);device->CreateRenderTargetView(back.Get(),nullptr,rtv);
    commands->OMSetRenderTargets(1,&rtv,FALSE,nullptr);const float black[4]={0,0,0,1};commands->ClearRenderTargetView(rtv,black,0,nullptr);
    Texture* source=nullptr;
    if(movieVisible)source=&movie;else {auto found=colors.find(liveDisplay);if(found!=colors.end())source=&found->second;}
    bool interpolate=interpolationEnabled && interpolationValid && !movieVisible;
#ifndef WOTM_D3D12_REPLAY
    interpolate=interpolate && !g_ps2xWotmPaused.load(std::memory_order_acquire) && g_ps2xWotmPhase.load(std::memory_order_relaxed)==2u;
#endif
    if(interpolate)source=&interpolationCurrent;
    if(source&&source->resource) {
      barrier(*source,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
      bool wide=false;
#ifndef WOTM_D3D12_REPLAY
      wide=!movieVisible && visibleWidth==640u && visibleHeight==448u && g_ps2xWotmWideActive.load(std::memory_order_relaxed);
#endif
      const float displayWidth=wide?float(visibleHeight)*(16.f/9.f):float(visibleWidth);
      const float fit=std::min(float(w)/displayWidth,float(h)/visibleHeight);
      const float dw=displayWidth*fit,dh=visibleHeight*fit,x=(w-dw)*0.5f,y=(h-dh)*0.5f;
      const float u=movieVisible?1.f:float(visibleWidth*scale)/source->w;
      const auto pass=livePassHeights.find(liveDisplay);
      const float v=movieVisible?1.f:float(wotm_capture::presentationSourceHeight(visibleHeight,
        pass!=livePassHeights.end()?pass->second:448u,height))/source->h;
      wotm_capture::Vertex quad[6]={{x,y,0,1,1,1,1,0,0,1,255},{x+dw,y,0,1,1,1,1,u,0,1,255},{x+dw,y+dh,0,1,1,1,1,u,v,1,255},
        {x,y,0,1,1,1,1,0,0,1,255},{x+dw,y+dh,0,1,1,1,1,u,v,1,255},{x,y+dh,0,1,1,1,1,0,v,1,255}};
      auto vb=upload(quad,sizeof(quad));D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(),sizeof(quad),sizeof(quad[0])};
      Constants k{};k.target[0]=float(w);k.target[1]=float(h);k.texSize[0]=float(source->w);k.texSize[1]=float(source->h);k.flags[0]=1;k.flags[2]=1;
      k.region[0]=u;k.region[1]=v;
      UINT presentSrv=source->srv;
      if(interpolate) {
        const double age=std::chrono::duration<double>(std::chrono::steady_clock::now()-interpolationArrival).count();
        k.motionInfo[0]=float(std::clamp(age/interpolationInterval,0.0,1.0));k.motionInfo[1]=float(fxaaEnabled);
        k.motionInfo[2]=64.0f*scale;
        if(nextSrv>65533u)throw std::runtime_error("Interpolation descriptor limit");
        presentSrv=nextSrv;nextSrv+=3;
        // Shader-visible descriptor heaps are not valid descriptor copy sources.
        // Create the three views directly in this command list's transient table.
        Texture* inputs[]={source,&interpolationPrevious,&interpolationFlow};
        for(UINT i=0;i<3;++i) {
          D3D12_SHADER_RESOURCE_VIEW_DESC view{};view.Format=inputs[i]->resource->GetDesc().Format;
          view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;view.Texture2D.MipLevels=1;
          view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
          device->CreateShaderResourceView(inputs[i]->resource.Get(),&view,cpu(srvs.Get(),presentSrv+i,srvStep));
        }
        if(k.motionInfo[0]>0.001f && k.motionInfo[0]<0.999f)++interpolationPresents;
      }
      auto cb=upload(&k,sizeof(k));wotm_capture::Draw d{};d.zmsk=1;d.mmin=d.mmag=1;
      ID3D12DescriptorHeap* heaps[]={srvs.Get(),samplers.Get()};commands->SetDescriptorHeaps(2,heaps);commands->SetGraphicsRootSignature(root.Get());
      commands->SetGraphicsRootConstantBufferView(0,cb->GetGPUVirtualAddress());commands->SetGraphicsRootDescriptorTable(1,gpu(srvs.Get(),presentSrv,srvStep));
      commands->SetGraphicsRootDescriptorTable(2,gpu(samplers.Get(),sampler(d,*source,false),samplerStep));
      commands->SetPipelineState(pipeline(d,false,true,fxaaEnabled && !movieVisible,interpolate));commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);commands->IASetVertexBuffers(0,1,&vbv);
      D3D12_VIEWPORT vp{0,0,float(w),float(h),0,1};D3D12_RECT sc{0,0,LONG(w),LONG(h)};commands->RSSetViewports(1,&vp);commands->RSSetScissorRects(1,&sc);commands->DrawInstanced(6,1,0,0);
    }
    static uint64_t presents=0;++presents;
    if(interpolationEnabled) {
      static auto started=std::chrono::steady_clock::now();static uint64_t lastCount=0;
      const auto now=std::chrono::steady_clock::now();const double elapsed=std::chrono::duration<double>(now-started).count();
      if(elapsed>=5.0) {
        std::fprintf(stderr,"[present:rate] %.2f/s presented=%llu interpolated=%llu completed=%llu\n",
          double(presents-lastCount)/elapsed,presents,interpolationPresents,interpolationFrame);
        started=now;lastCount=presents;
      }
    }
    const char* shotAt=std::getenv("PS2X_D3D12_SHOT_FRAME");
    static const std::vector<uint64_t> shotFrames=[] {
      std::vector<uint64_t> frames;const char* p=std::getenv("PS2X_D3D12_SHOT_FRAMES");
      while(p && *p && frames.size()<32) {char* end=nullptr;auto frame=std::strtoull(p,&end,10);
        if(end==p)break;frames.push_back(frame);p=*end==','?end+1:end;}
      return frames;
    }();
    const bool shot=shotBase && (shotFrames.empty()?presents==(shotAt?std::strtoull(shotAt,nullptr,10):1800ull):
      std::find(shotFrames.begin(),shotFrames.end(),presents)!=shotFrames.end());
    const std::string shotPath=shot?std::string(shotBase)+(shotFrames.empty()?"":"."+std::to_string(presents)+".ppm"):"";
    ComPtr<ID3D12Resource> readback;D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    if(shot){
      b.Transition.StateBefore=D3D12_RESOURCE_STATE_RENDER_TARGET;b.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;commands->ResourceBarrier(1,&b);
      auto desc=back->GetDesc();UINT64 bytes;device->GetCopyableFootprints(&desc,0,1,0,&footprint,nullptr,nullptr,&bytes);readback=buffer(bytes,D3D12_HEAP_TYPE_READBACK);
      D3D12_TEXTURE_COPY_LOCATION dst{};dst.pResource=readback.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=footprint;
      D3D12_TEXTURE_COPY_LOCATION src{};src.pResource=back.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    }
    b.Transition.StateBefore=shot?D3D12_RESOURCE_STATE_COPY_SOURCE:D3D12_RESOURCE_STATE_RENDER_TARGET;b.Transition.StateAfter=D3D12_RESOURCE_STATE_PRESENT;commands->ResourceBarrier(1,&b);
    const auto gpuStart=std::chrono::steady_clock::now();finish();pendingLiveScene=false;
    const auto gpuEnd=std::chrono::steady_clock::now();saveMotion();
    const HRESULT presented=liveSwap->Present(vsync?1:0,!vsync && liveTearing?DXGI_PRESENT_ALLOW_TEARING:0);
    check(presented,"Live Present");
    if(presented==DXGI_STATUS_OCCLUDED) {
      liveOccluded=true;
      // Explicit hidden-window captures still exercise temporal rendering.
      // Ordinary occlusion continues to discard history and suspend display.
      if(!shotBase || !*shotBase){interpolationValid=false;interpolationFrame=0;}
    }
    static const bool trace=std::getenv("PS2X_PRESENT_TIMING")!=nullptr;
    if(trace) {
      static auto start=gpuStart;static double gpu=0,present=0;static unsigned count=0;
      const auto now=std::chrono::steady_clock::now();gpu+=std::chrono::duration<double>(gpuEnd-gpuStart).count();
      present+=std::chrono::duration<double>(now-gpuEnd).count();++count;
      if(std::chrono::duration<double>(now-start).count()>=5.0) {
        std::fprintf(stderr,"[present:gpu-timing] finish=%.3fms present=%.3fms vsync=%u\n",gpu*1000/count,present*1000/count,unsigned(vsync));
        start=now;gpu=present=0;count=0;
      }
    }
    debugMessages();
    if(shot){uint8_t* data=nullptr;D3D12_RANGE range{0,size_t(footprint.Footprint.RowPitch)*(h-1)+w*4};check(readback->Map(0,&range,reinterpret_cast<void**>(&data)),"Live shot map");
      std::ofstream out(shotPath,std::ios::binary);out<<"P6\n"<<w<<" "<<h<<"\n255\n";
      for(UINT y=0;y<h;++y)for(UINT x=0;x<w;++x)out.write(reinterpret_cast<char*>(data+size_t(y)*footprint.Footprint.RowPitch+x*4),3);
      D3D12_RANGE none{0,0};readback->Unmap(0,&none);if(!out)throw std::runtime_error("Live screenshot write failed");
      std::fprintf(stderr,"[gs:d3d12] Live presentation screenshot saved: %s\n",shotPath.c_str());
    }
  }
  // Static captured frame shown directly from a D3D12 resource, not live
  // gameplay.
  void preview(uint32_t display) {
    auto found = colors.find(display);
    if (found == colors.end())
      throw std::runtime_error("Preview display target missing");
    WNDCLASSW cls{};
    cls.hInstance = GetModuleHandleW(nullptr);
    cls.lpszClassName = L"WotmD3D12ReplayPreview";
    cls.lpfnWndProc = [](HWND hwnd, UINT msg, WPARAM w, LPARAM l) -> LRESULT {
      if (msg == WM_KEYDOWN && w == VK_ESCAPE) {
        DestroyWindow(hwnd);
        return 0;
      }
      return DefWindowProcW(hwnd, msg, w, l);
    };
    if (!RegisterClassW(&cls))
      throw std::runtime_error("Register preview window failed");
    const DWORD style =
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    RECT rect{0, 0, 1280, 896};
    AdjustWindowRect(&rect, style, FALSE);
    HWND window = CreateWindowW(cls.lpszClassName,
                                L"War of the Monsters - D3D12 captured-frame "
                                L"preview (not live gameplay)",
                                style, 100, 80, rect.right - rect.left,
                                rect.bottom - rect.top, nullptr, nullptr,
                                cls.hInstance, nullptr);
    if (!window)
      throw std::runtime_error("Create preview window failed");
    struct WindowScope {
      HWND h;
      ~WindowScope() {
        if (IsWindow(h))
          DestroyWindow(h);
      }
    } scope{window};
    ComPtr<IDXGIFactory4> factory;
    check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "Preview factory");
    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = width;
    desc.Height = height;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 2;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.Scaling = DXGI_SCALING_STRETCH;
    ComPtr<IDXGISwapChain1> first;
    check(factory->CreateSwapChainForHwnd(queue.Get(), window, &desc, nullptr,
                                          nullptr, &first),
          "Preview swap chain");
    check(factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER),
          "Preview window association");
    ComPtr<IDXGISwapChain3> swap;
    check(first.As(&swap), "Preview swap interface");
    ComPtr<ID3D12Resource> back;
    check(
        swap->GetBuffer(swap->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&back)),
        "Preview buffer");
    check(allocator->Reset(), "Preview allocator reset");
    check(commands->Reset(allocator.Get(), nullptr), "Preview commands reset");
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {back.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST};
    commands->ResourceBarrier(1, &b);
    commands->CopyResource(back.Get(), found->second.resource.Get());
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    commands->ResourceBarrier(1, &b);
    finish();
    ShowWindow(window, SW_SHOWNORMAL);
    SetForegroundWindow(window);
    check(swap->Present(1, 0), "Preview Present");
    debugMessages();
    MSG message{};
    while (IsWindow(window)) {
      while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
      }
      if (IsWindow(window))
        MsgWaitForMultipleObjects(0, nullptr, FALSE, 100, QS_ALLINPUT);
    }
  }
  void run(wotm_capture::Frame &frame, const char *prefix,
           bool strict = false) {
    width = frame.width;
    height = frame.height;
    scale = frame.scale;
    if (frame.colors.size() > 32 || frame.depths.size() > 32 ||
        frame.textures.size() + frame.colors.size() + 1 > 65536)
      throw std::runtime_error("Capture descriptor limit");
    for (auto &image : frame.colors) {
      Texture t = texture({image}, true, false);
      UINT slot = UINT(colorSlots.size());
      device->CreateRenderTargetView(t.resource.Get(), nullptr,
                                     cpu(rtvs.Get(), slot, rtvStep));
      colorSlots[image.id] = slot;
      colors.emplace(image.id, std::move(t));
    }
    for (auto &image : frame.depths) {
      Texture t = texture({image}, false, true);
      UINT slot = UINT(depthSlots.size());
      D3D12_DEPTH_STENCIL_VIEW_DESC view{};
      view.Format = DXGI_FORMAT_D32_FLOAT;
      view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
      device->CreateDepthStencilView(t.resource.Get(), &view,
                                     cpu(dsvs.Get(), slot, dsvStep));
      depthSlots[image.id] = slot;
      depths.emplace(image.id, std::move(t));
    }
    for (auto &t : frame.textures)
      textures.push_back(texture(t.levels));
    white = texture({{0, 1, 1, {255, 255, 255, 255}}});
    draw(frame);
    struct Read {
      uint32_t id;
      ComPtr<ID3D12Resource> buffer;
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    };
    std::vector<Read> reads;
    for (auto &pair : colors) {
      auto &t = pair.second;
      barrier(t, D3D12_RESOURCE_STATE_COPY_SOURCE);
      auto desc = t.resource->GetDesc();
      Read read{};
      read.id = pair.first;
      UINT64 size;
      device->GetCopyableFootprints(&desc, 0, 1, 0, &read.footprint, nullptr,
                                    nullptr, &size);
      read.buffer = buffer(size, D3D12_HEAP_TYPE_READBACK);
      D3D12_TEXTURE_COPY_LOCATION dst{};
      dst.pResource = read.buffer.Get();
      dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      dst.PlacedFootprint = read.footprint;
      D3D12_TEXTURE_COPY_LOCATION src{};
      src.pResource = t.resource.Get();
      src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      commands->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
      reads.push_back(std::move(read));
    }
    finish();
    debugMessages();
    size_t badTotal = 0, pixelTotal = 0;
    bool comparisonFailed = false;
    for (auto &read : reads) {
      auto expected =
          std::find_if(frame.expected.begin(), frame.expected.end(),
                       [&](const auto &i) { return i.id == read.id; });
      if (expected == frame.expected.end())
        throw std::runtime_error("Missing expected image");
      uint8_t *data = nullptr;
      D3D12_RANGE range{0, SIZE_T(read.footprint.Footprint.RowPitch) *
                                   (height - 1) +
                               SIZE_T(width) * 4};
      check(read.buffer->Map(0, &range, reinterpret_cast<void **>(&data)),
            "Map result");
      std::vector<uint8_t> rgb(size_t(width) * height * 3), diff(rgb.size()),
          alphaDiff(rgb.size());
      size_t bad = 0, badAlpha = 0;
      uint64_t error = 0;
      int maxError = 0, maxAlpha = 0;
      for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x) {
          bool mismatch = false;
          for (UINT c = 0; c < 4; ++c) {
            size_t index = (size_t(y) * width + x) * 4 + c;
            int actual =
                data[size_t(y) * read.footprint.Footprint.RowPitch + x * 4 + c];
            int delta = std::abs(actual - int(expected->bytes[index]));
            if (c < 3) {
              rgb[(index / 4) * 3 + c] = uint8_t(actual);
              diff[(index / 4) * 3 + c] = uint8_t(std::min(delta * 8, 255));
              error += delta;
              maxError = std::max(maxError, delta);
              if (delta > 2)
                mismatch = true;
            } else {
              maxAlpha = std::max(maxAlpha, delta);
              if (delta > 2)
                ++badAlpha;
              for (int ac = 0; ac < 3; ++ac)
                alphaDiff[(index / 4) * 3 + ac] = uint8_t(delta);
            }
          }
          if (mismatch)
            ++bad;
        }
      if (std::getenv("PS2X_D3D12_RAW")) {
        std::ofstream raw(std::string(prefix) + "_" + std::to_string(read.id) +
                              ".rgba",
                          std::ios::binary);
        for (UINT y = 0; y < height; ++y)
          raw.write(reinterpret_cast<const char *>(
                        data + size_t(y) * read.footprint.Footprint.RowPitch),
                    width * 4);
        std::ofstream ref(std::string(prefix) + "_" + std::to_string(read.id) +
                              "_gl.rgba",
                          std::ios::binary);
        ref.write(reinterpret_cast<const char *>(expected->bytes.data()),
                  std::streamsize(expected->bytes.size()));
        if (!raw || !ref)
          throw std::runtime_error("Raw output failed");
      }
      D3D12_RANGE empty{0, 0};
      read.buffer->Unmap(0, &empty);
      badTotal += bad;
      pixelTotal += size_t(width) * height;
      auto save = [&](const char *suffix, const std::vector<uint8_t> &bytes) {
        std::string path =
            std::string(prefix) + "_" + std::to_string(read.id) + suffix;
        std::ofstream file(path, std::ios::binary);
        file << "P6\n" << width << " " << height << "\n255\n";
        file.write(reinterpret_cast<const char *>(bytes.data()),
                   std::streamsize(bytes.size()));
        if (!file)
          throw std::runtime_error("Write output failed");
      };
      save(".ppm", rgb);
      save("_diff.ppm", diff);
      save("_alpha_diff.ppm", alphaDiff);
      for (size_t pixel = 0; pixel < size_t(width) * height; ++pixel)
        std::copy_n(expected->bytes.data() + pixel * 4, 3,
                    rgb.data() + pixel * 3);
      save("_gl.ppm", rgb);
      std::printf("FBP %x: RGB error mean %.5f, max %d, >2 error pixels "
                  "%zu/%zu; alpha max %d, bad %zu\n",
                  read.id, double(error) / (size_t(width) * height * 3),
                  maxError, bad, size_t(width) * height, maxAlpha, badAlpha);
      // Apply the tolerance per framebuffer: unchanged buffers must not dilute
      // errors.
      const size_t allowed = strict ? 0 : size_t(width) * height / 1000;
      const auto format=std::find_if(frame.formats.begin(),frame.formats.end(),[&](const auto& item){return item.fbp==read.id;});
      const bool rgb24=format!=frame.formats.end()&&format->psm==1u;
      if(rgb24)std::printf("  PSMCT24: stored GPU alpha is not a framebuffer channel; RGB comparison applies\n");
      comparisonFailed |= bad > allowed || (!rgb24 && badAlpha > allowed);
    }
    std::printf("Replayed %zu draws, %zu textures, %zu pipelines; bad pixels "
                "%zu/%zu; debug layer clean\n",
                frame.draws.size(), frame.textures.size(), pipelines.size(),
                badTotal, pixelTotal);
    if (comparisonFailed)
      throw std::runtime_error("Reference difference exceeds per-target "
                               "tolerance (>2 color/alpha levels)");
  }
};
} // namespace wotm::d3d12
#ifndef WOTM_D3D12_REPLAY
namespace { std::unique_ptr<wotm::d3d12::GsReplay> liveRenderer;
bool liveFailed=false;
void liveError(const std::exception& e){liveFailed=true;std::fprintf(stderr,"[gs:d3d12] ERROR: %s\n",e.what());}
auto& getLiveRenderer() {
  if(!liveRenderer){auto r=std::make_unique<wotm::d3d12::GsReplay>();
    const char* debug=std::getenv("PS2X_D3D12_DEBUG");r->initialize(L"ps2xRuntime/src/lib/gs/gs_d3d12.hlsl",debug&&debug[0]=='1');
    liveRenderer=std::move(r);std::fprintf(stderr,"[gs:d3d12] Live renderer active; no OpenGL scene replay\n");}
  return *liveRenderer;
}}
bool ps2xD3D12Enabled() {static const bool enabled=[] {const char* p=std::getenv("PS2X_GS_BACKEND");return p&&std::strcmp(p,"d3d12")==0;}();return enabled;}
void ps2xD3D12Frame(wotm_capture::Frame& frame,uint32_t width,uint32_t height){if(liveFailed)return;try{getLiveRenderer().liveFrame(frame,width,height);}catch(const std::exception& e){liveError(e);}}
void ps2xD3D12Movie(const uint8_t* pixels,uint32_t width,uint32_t height){if(liveFailed)return;try{getLiveRenderer().liveMovie(pixels,width,height);}catch(const std::exception& e){liveError(e);}}
bool ps2xD3D12Present(void* window){if(liveFailed)return false;try{const char* v=std::getenv("PS2X_DISPLAY_VSYNC");getLiveRenderer().presentLive(static_cast<HWND>(window),v&&v[0]=='1');return true;}catch(const std::exception& e){liveError(e);return false;}}
void ps2xD3D12Shutdown(){liveRenderer.reset();}
#endif
#ifdef WOTM_D3D12_REPLAY
int wmain(int argc, wchar_t **argv) {
  try {
    if (argc != 4 && argc != 5)
      throw std::runtime_error("Usage: d3d12_replay capture.wdr shader.hlsl "
                               "output_prefix [--strict|--preview|--pack]");
    if (argc == 5 && std::wcscmp(argv[4], L"--strict") &&
        std::wcscmp(argv[4], L"--preview") && std::wcscmp(argv[4], L"--pack"))
      throw std::runtime_error("Unknown option");
    auto utf8 = [](const wchar_t *text) {
      int n = WideCharToMultiByte(CP_ACP, 0, text, -1, nullptr, 0, nullptr,
                                  nullptr);
      std::string s(size_t(n), 0);
      WideCharToMultiByte(CP_ACP, 0, text, -1, s.data(), n, nullptr, nullptr);
      s.pop_back();
      return s;
    };
    wotm_capture::Frame f;
    wotm_capture::File input(utf8(argv[1]).c_str(), true);
    input.frame(f);
    // Isolated diagnostics also exercise the live borrowed-storage contract.
    std::vector<wotm_capture::Vertex> borrowedSource;
    if(std::getenv("PS2X_D3D12_REPLAY_BORROW")) {
      borrowedSource=std::move(f.vertices);
      f.borrowedVertices=borrowedSource.data();f.borrowedVertexCount=borrowedSource.size();
    }
    wotm::d3d12::GsReplay renderer;
    renderer.initialize(argv[2]);
    if(argc==5 && !std::wcscmp(argv[4], L"--pack"))wotm_capture::packTransferTiles(f);
    renderer.run(f, utf8(argv[3]).c_str(),
                 argc == 5 && !std::wcscmp(argv[4], L"--strict"));
    if (argc == 5 && !std::wcscmp(argv[4], L"--preview"))
      renderer.preview(f.display);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Replay failed: %s\n", e.what());
    return 1;
  }
}
#endif

// Optional GS compute offload. The CPU worker owns this context and serializes
// VRAM transfers. Unsupported segments return without changing CPU VRAM.
#include "runtime/gs/gs_types.h"
#include <cstdint>
#include <cstddef>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <external/glad.h>
#include "runtime/gs/ps2_gs_memory.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <mutex>
#include <span>
namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t VramBytes = 4u*1024u*1024u;
struct Capture {
    std::span<uint8_t> before;
    std::vector<GSPrimitiveBatch> draws;
    uint64_t frame=0;
};
void require(bool value,const char *message) { if(!value) throw std::runtime_error(message); }
uint32_t readPixel(uint8_t *v,uint32_t psm,uint32_t bp,uint32_t bw,uint32_t x,uint32_t y) {
    switch(psm) {
    case GS_PSM_CT32:return GSMem::ReadCT32(v,bp,bw,x,y);
    case GS_PSM_CT24:return GSMem::ReadCT24(v,bp,bw,x,y);
    case GS_PSM_CT16:return GSMem::ReadCT16(v,bp,bw,x,y);
    case GS_PSM_CT16S:return GSMem::ReadCT16S(v,bp,bw,x,y);
    case GS_PSM_T8:return GSMem::ReadP8(v,bp,bw,x,y);
    case GS_PSM_T8H:return GSMem::ReadP8H(v,bp,bw,x,y);
    case GS_PSM_T4:return GSMem::ReadP4(v,bp,bw,x,y);
    case GS_PSM_T4HL:return GSMem::ReadP4HL(v,bp,bw,x,y);
    case GS_PSM_T4HH:return GSMem::ReadP4HH(v,bp,bw,x,y);
    default:throw std::runtime_error("Unsupported texture/palette format");
    }
}
uint32_t expand16(uint32_t c) {
    return ((c&31)<<3)|(((c>>5)&31)<<11)|(((c>>10)&31)<<19)|(((c>>15)&1)<<31);
}
uint32_t applyTexa(const GSTexaReg &a,uint32_t psm,uint32_t c) {
    if(psm==GS_PSM_CT32)return c;
    uint32_t alpha=c>>24;
    if(psm==GS_PSM_CT24)alpha=(a.aem&&(c&0xffffff)==0)?0:a.ta0;
    if(psm==GS_PSM_CT16||psm==GS_PSM_CT16S)alpha=(alpha&128)?a.ta1:((a.aem&&(c&0xffffff)==0)?0:a.ta0);
    return (c&0xffffff)|(alpha<<24);
}
uint32_t texel(uint8_t *v,const GSDrawState &s,uint32_t x,uint32_t y) {
    const auto &t=s.context.tex0;
    uint32_t value=readPixel(v,t.psm,t.tbp0,t.tbw,x,y), psm=t.psm;
    if(psm==GS_PSM_T8||psm==GS_PSM_T8H||psm==GS_PSM_T4||psm==GS_PSM_T4HL||psm==GS_PSM_T4HH) {
        const bool four=psm==GS_PSM_T4||psm==GS_PSM_T4HL||psm==GS_PSM_T4HH;
        uint32_t index=value&(four?15:255);
        if(t.csm==0) {
            bool small=t.cpsm==GS_PSM_CT16||t.cpsm==GS_PSM_CT16S;
            index=(((t.csa&(small?31:15))<<4)+index)&(small?511:255);
            index=(index&~24u)|((index&8)<<1)|((index&16)>>1);
        }
        psm=t.cpsm;
        value=readPixel(v,psm,t.cbp,std::max(1u,uint32_t(s.texclut.cbw)),s.texclut.cou+(index&15),s.texclut.cov+(index>>4));
    }
    if(psm==GS_PSM_CT16||psm==GS_PSM_CT16S)value=expand16(value);
    return applyTexa(s.texa,psm,value);
}
// Exact source-page snapshots keep decoded textures valid across CPU writes,
// transfers, palette updates and GPU draws. Page selection follows GSMem's
// PageId; a non-page-aligned block base can spill into the following page.
using TextureKey=std::array<uint32_t,16>;
struct DecodedTexture {
    uint64_t lastUse=0;
    std::vector<uint16_t> pages;
    std::vector<uint8_t> snapshot;
    std::vector<uint32_t> pixels;
};
thread_local uint64_t textureHits=0,textureMisses=0;
const std::vector<uint32_t> &decodedTexture(uint8_t *v,const GSDrawState &s,const TextureKey &key,uint32_t width,uint32_t height) {
    static thread_local std::map<TextureKey,DecodedTexture> cache;
    static thread_local uint64_t useSerial=0;
    auto found=cache.find(key);
    if(found==cache.end()) {
        if(cache.size()>=128) {
            auto oldest=std::min_element(cache.begin(),cache.end(),[](const auto &a,const auto &b){return a.second.lastUse<b.second.lastUse;});
            cache.erase(oldest);
        }
        DecodedTexture entry;
        std::array<bool,VramBytes/8192> pages{};
        auto cover=[&](uint32_t psm,uint32_t bp,uint32_t bw,uint32_t x0,uint32_t y0,uint32_t w,uint32_t h) {
            uint32_t pw=64,ph=32;
            if(psm==GS_PSM_T8){pw=128;ph=64;}
            else if(psm==GS_PSM_T4){pw=128;ph=128;}
            else if(psm==GS_PSM_CT16 || psm==GS_PSM_CT16S)ph=64;
            for(uint32_t y=y0/ph;y<=(y0+h-1)/ph;++y)
                for(uint32_t x=x0/pw;x<=(x0+w-1)/pw;++x) {
                    uint32_t page=bp/32+y*((bw*64)/pw)+x;
                    pages[page%pages.size()]=true;
                    if(bp%32)pages[(page+1)%pages.size()]=true;
                }
        };
        const auto &t=s.context.tex0;
        cover(t.psm,t.tbp0,t.tbw,0,0,width,height);
        if(t.psm==GS_PSM_T8 || t.psm==GS_PSM_T8H || t.psm==GS_PSM_T4 || t.psm==GS_PSM_T4HL || t.psm==GS_PSM_T4HH)
            cover(t.cpsm,t.cbp,std::max(1u,uint32_t(s.texclut.cbw)),s.texclut.cou,s.texclut.cov,16,32);
        for(uint16_t i=0;i<pages.size();++i)if(pages[i])entry.pages.push_back(i);
        found=cache.emplace(key,std::move(entry)).first;
    }
    auto &entry=found->second;
    entry.lastUse=++useSerial;
    bool same=entry.snapshot.size()==entry.pages.size()*8192;
    for(size_t i=0;same && i<entry.pages.size();++i)
        same=std::memcmp(entry.snapshot.data()+i*8192,v+size_t(entry.pages[i])*8192,8192)==0;
    if(same){++textureHits;return entry.pixels;}
    ++textureMisses;
    entry.pixels.clear();entry.pixels.reserve(size_t(width)*height);
    for(uint32_t y=0;y<height;++y)for(uint32_t x=0;x<width;++x)entry.pixels.push_back(texel(v,s,x,y));
    entry.snapshot.resize(entry.pages.size()*8192);
    for(size_t i=0;i<entry.pages.size();++i)std::memcpy(entry.snapshot.data()+i*8192,v+size_t(entry.pages[i])*8192,8192);
    size_t cacheBytes=0;
    for(const auto &[unused,item]:cache)cacheBytes+=item.snapshot.capacity()+item.pixels.capacity()*4;
    while(cacheBytes>64u*1024u*1024u && cache.size()>1) {
        auto oldest=cache.end();
        for(auto it=cache.begin();it!=cache.end();++it)
            if(it!=found && (oldest==cache.end() || it->second.lastUse<oldest->second.lastUse))oldest=it;
        cacheBytes-=oldest->second.snapshot.capacity()+oldest->second.pixels.capacity()*4;
        cache.erase(oldest);
    }
    return entry.pixels;
}
struct Prepared {
    uint32_t width=0,height=0,tilesX=0,tilesY=0;
    std::vector<uint32_t> commands,texels,indices;
    std::vector<std::array<uint32_t,2>> tiles,addresses;
};
uint32_t rgba(const GSVertex &v) {return v.r|(uint32_t(v.g)<<8)|(uint32_t(v.b)<<16)|(uint32_t(v.a)<<24);}
Prepared prepare(Capture &c) {
    // Lookup tables were initialized by the owning CPU backend.
    Prepared out;
    const auto &first=c.draws.front().state.context;
    require(first.frame.psm==GS_PSM_CT24||first.frame.psm==GS_PSM_CT32,"Unsupported color target");
    require(first.zbuf.psm==GS_PSM_Z24||first.zbuf.psm==GS_PSM_Z32,"Unsupported depth target");
    for(const auto &d:c.draws) {
        auto &s=d.state;auto &k=s.context;
        require(d.vertexCount==3&&(s.prim.type==GS_PRIM_TRIANGLE||s.prim.type==GS_PRIM_TRISTRIP||s.prim.type==GS_PRIM_TRIFAN),"Segment contains non-triangle primitives");
        require(k.frame.fbp==first.frame.fbp&&k.frame.fbw==first.frame.fbw&&k.frame.psm==first.frame.psm&&k.zbuf.zbp==first.zbuf.zbp&&k.zbuf.psm==first.zbuf.psm,"Segment changes targets; split it before GPU submission");
        require(k.scissor.x0<=k.scissor.x1&&k.scissor.y0<=k.scissor.y1&&k.scissor.x1<2048&&k.scissor.y1<2048,"Invalid scissor");
        // GS dithering only affects 16-bit color targets, which this slice rejects.
        // The present CPU reference ignores SCANMSK even when WotM sets 2.
        // This comparison backend mirrors that behavior; hardware-accurate
        // scan masking must be evaluated separately from CPU equivalence.
        for(auto &v:d.vertices)require(std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z)&&std::abs(v.x)<1e6f&&std::abs(v.y)<1e6f&&v.z>=0&&v.z<=4294967295.0,"Invalid vertex");
        out.width=std::max(out.width,uint32_t(k.scissor.x1)+1);out.height=std::max(out.height,uint32_t(k.scissor.y1)+1);
    }
    require(first.frame.fbw>0&&out.width<=first.frame.fbw*64,"Aliasing target stride");
    out.tilesX=(out.width+7)/8;out.tilesY=(out.height+7)/8;
    std::vector<std::vector<uint32_t>> tileLists(out.tilesX*out.tilesY);
    // Identity VRAM lets the existing GS swizzle implementation supply exact
    // word addresses, without reimplementing its address tables in GLSL.
    using AddressKey=std::array<uint32_t,7>;
    static thread_local std::map<AddressKey,std::vector<std::array<uint32_t,2>>> addressCache;
    AddressKey addressKey={first.frame.fbp,first.frame.fbw,first.frame.psm,first.zbuf.zbp,first.zbuf.psm,out.width,out.height};
    auto cached=addressCache.find(addressKey);
    if(cached==addressCache.end()) {
        std::vector<uint32_t> identity(VramBytes/4);
        for(uint32_t i=0;i<identity.size();++i)identity[i]=i;
        std::set<uint32_t> used;
        std::vector<std::array<uint32_t,2>> addresses;
        addresses.reserve(size_t(out.width)*out.height);
        for(uint32_t y=0;y<out.height;++y)for(uint32_t x=0;x<out.width;++x) {
            auto *v=reinterpret_cast<uint8_t*>(identity.data());
            uint32_t ca=GSMem::ReadCT32(v,first.frame.fbp<<5,first.frame.fbw,x,y);
            uint32_t za=GSMem::ReadZ32(v,first.zbuf.zbp<<5,first.frame.fbw,x,y);
            require(ca<identity.size()&&za<identity.size()&&used.insert(ca).second&&used.insert(za).second,"Color/depth target aliases between pixel owners");
            addresses.push_back({ca,za});
        }
        if(addressCache.size()>=16)addressCache.erase(addressCache.begin());
        cached=addressCache.emplace(addressKey,std::move(addresses)).first;
    }
    out.addresses=cached->second;
    // Textures are decoded from initial VRAM once per unique sampler image.
    // GPU sampling, filtering, shade/texture combine, tests and writes follow.
    std::map<std::array<uint32_t,16>,std::array<uint32_t,2>> textureCache;
    for(uint32_t n=0;n<c.draws.size();++n) {
        const auto &d=c.draws[n];const auto &s=d.state;const auto &k=s.context;const auto &t=k.tex0;
        std::array<uint32_t,48> p{};
        float xs[3],ys[3];
        for(uint32_t j=0;j<3;++j) {
            const auto &v=d.vertices[j];xs[j]=v.x-float(k.xyoffset.ofx>>4);ys[j]=v.y-float(k.xyoffset.ofy>>4);
            p[j*8]=std::bit_cast<uint32_t>(xs[j]);p[j*8+1]=std::bit_cast<uint32_t>(ys[j]);p[j*8+2]=uint32_t(v.z);p[j*8+3]=rgba(v);
            p[j*8+4]=std::bit_cast<uint32_t>(v.s);p[j*8+5]=std::bit_cast<uint32_t>(v.t);p[j*8+6]=std::bit_cast<uint32_t>(v.q);
            p[j*8+7]=v.u|(uint32_t(v.v)<<16);p[44+j]=v.fog;
        }
        p[24]=uint32_t(s.prim.iip)|(s.prim.tme<<1)|(s.prim.fst<<2)|(s.linearFilter<<3)|(s.prim.fge<<4)|(s.prim.abe<<5)|(k.zbuf.zmask<<6)|((k.frame.psm==GS_PSM_CT24)<<7)|(s.pabe<<8);
        p[25]=uint32_t(k.test);p[26]=uint32_t(k.fba);p[27]=k.frame.fbmsk;
        p[30]=s.textureWidth;p[31]=s.textureHeight;p[32]=uint32_t(k.clamp)&15;
        p[33]=uint32_t((k.clamp>>4)&1023)|(uint32_t((k.clamp>>14)&1023)<<16);
        p[34]=uint32_t((k.clamp>>24)&1023)|(uint32_t((k.clamp>>34)&1023)<<16);
        p[35]=t.tfx|(uint32_t(t.tcc)<<8);p[36]=uint32_t(k.alpha);p[37]=uint32_t(k.alpha>>32);p[38]=s.fogR|(uint32_t(s.fogG)<<8)|(uint32_t(s.fogB)<<16);
        p[40]=k.scissor.x0;p[41]=k.scissor.x1;p[42]=k.scissor.y0;p[43]=k.scissor.y1;p[47]=k.zbuf.psm==GS_PSM_Z24?0xffffffu:0xffffffffu;
        if(s.prim.tme) {
            require(s.textureWidth>0&&s.textureHeight>0&&s.textureWidth<=1024&&s.textureHeight<=1024,"Invalid texture size");
            auto extent=[](uint32_t size,uint32_t mode,uint32_t limits) {return mode<2?size:(mode==2?(limits>>16)+1:((limits&65535)|(limits>>16))+1);};
            uint32_t tw=extent(s.textureWidth,p[32]&3,p[33]),th=extent(s.textureHeight,(p[32]>>2)&3,p[34]);
            require(tw>0&&th>0&&tw<=1024&&th<=1024,"Invalid sampler extent");
            std::array<uint32_t,16> key={t.tbp0,t.tbw,t.psm,t.cbp,t.cpsm,t.csm,t.csa,s.texclut.cbw,s.texclut.cou,s.texclut.cov,s.texa.ta0,s.texa.ta1,uint32_t(s.texa.aem),tw,th,0};
            auto found=textureCache.find(key);
            if(found==textureCache.end()) {
                require(out.texels.size()+size_t(tw)*th<=64u*1024u*1024u,"Decoded texture budget exceeded");
                uint32_t offset=uint32_t(out.texels.size());
                const auto &decoded=decodedTexture(c.before.data(),s,key,tw,th);
                out.texels.insert(out.texels.end(),decoded.begin(),decoded.end());
                found=textureCache.emplace(key,std::array<uint32_t,2>{offset,tw}).first;
            }
            p[28]=found->second[0];p[29]=found->second[1];
        }
        out.commands.insert(out.commands.end(),p.begin(),p.end());
        int x0=std::clamp(int(std::floor(*std::min_element(xs,xs+3))),int(k.scissor.x0),int(k.scissor.x1));
        int x1=std::clamp(int(std::ceil(*std::max_element(xs,xs+3))),int(k.scissor.x0),int(k.scissor.x1));
        int y0=std::clamp(int(std::floor(*std::min_element(ys,ys+3))),int(k.scissor.y0),int(k.scissor.y1));
        int y1=std::clamp(int(std::ceil(*std::max_element(ys,ys+3))),int(k.scissor.y0),int(k.scissor.y1));
        for(int y=y0/8;y<=y1/8;++y)for(int x=x0/8;x<=x1/8;++x)tileLists[y*out.tilesX+x].push_back(n);
    }
    for(const auto &list:tileLists) {
        require(out.indices.size()+list.size()<=16u*1024u*1024u,"Tile index budget exceeded");
        out.tiles.push_back({uint32_t(out.indices.size()),uint32_t(list.size())});
        out.indices.insert(out.indices.end(),list.begin(),list.end());
    }
    if(out.texels.empty())out.texels.push_back(0);
    if(out.indices.empty())out.indices.push_back(0);
    return out;
}

void *loadGl(const char *name) {
    auto p=wglGetProcAddress(name);
    if(!p || p==reinterpret_cast<PROC>(1) || p==reinterpret_cast<PROC>(2) || p==reinterpret_cast<PROC>(3) || p==reinterpret_cast<PROC>(-1))
        p=GetProcAddress(GetModuleHandleW(L"opengl32.dll"),name);
    require(p!=nullptr,name);return reinterpret_cast<void*>(p);
}
class WorkerContext {
    HWND window=nullptr;HDC dc=nullptr;HGLRC context=nullptr;
    void close() noexcept {
        if(context){wglMakeCurrent(nullptr,nullptr);wglDeleteContext(context);context=nullptr;}
        if(dc){ReleaseDC(window,dc);dc=nullptr;}
        if(window){DestroyWindow(window);window=nullptr;}
    }
public:
    WorkerContext() {
        try {
            static std::once_flag registered;
            std::call_once(registered,[]{WNDCLASSW wc{};wc.style=CS_OWNDC;wc.lpfnWndProc=DefWindowProcW;wc.hInstance=GetModuleHandleW(nullptr);wc.lpszClassName=L"PS2XGsCompute";require(RegisterClassW(&wc)!=0 || GetLastError()==ERROR_CLASS_ALREADY_EXISTS,"Register compute window");});
            window=CreateWindowW(L"PS2XGsCompute",L"GS compute",WS_POPUP,0,0,16,16,nullptr,nullptr,GetModuleHandleW(nullptr),nullptr);
            require(window!=nullptr,"Create compute window");dc=GetDC(window);require(dc!=nullptr,"Compute DC");
            PIXELFORMATDESCRIPTOR pfd{};pfd.nSize=sizeof(pfd);pfd.nVersion=1;pfd.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER;pfd.iPixelType=PFD_TYPE_RGBA;pfd.cColorBits=32;
            int format=ChoosePixelFormat(dc,&pfd);require(format && SetPixelFormat(dc,format,&pfd),"Compute pixel format");
            context=wglCreateContext(dc);require(context && wglMakeCurrent(dc,context),"Compute bootstrap context");
            using Create=HGLRC(WINAPI*)(HDC,HGLRC,const int*);
            auto create=reinterpret_cast<Create>(loadGl("wglCreateContextAttribsARB"));
            const int attrs[]={0x2091,4,0x2092,3,0x9126,1,0};
            HGLRC modern=create(dc,nullptr,attrs);require(modern!=nullptr,"OpenGL 4.3 unavailable");
            wglMakeCurrent(nullptr,nullptr);wglDeleteContext(context);context=modern;
            require(wglMakeCurrent(dc,context)!=0,"Activate compute context");
        } catch(...) {close();throw;}
    }
    ~WorkerContext(){close();}
};
struct GlApi {
    PFNGLATTACHSHADERPROC AttachShader = nullptr;
    PFNGLBEGINQUERYPROC BeginQuery = nullptr;
    PFNGLBINDBUFFERPROC BindBuffer = nullptr;
    PFNGLBINDBUFFERBASEPROC BindBufferBase = nullptr;
    PFNGLBUFFERDATAPROC BufferData = nullptr;
    PFNGLBUFFERSUBDATAPROC BufferSubData = nullptr;
    PFNGLCOMPILESHADERPROC CompileShader = nullptr;
    PFNGLCREATEPROGRAMPROC CreateProgram = nullptr;
    PFNGLCREATESHADERPROC CreateShader = nullptr;
    PFNGLDELETEBUFFERSPROC DeleteBuffers = nullptr;
    PFNGLDELETEPROGRAMPROC DeleteProgram = nullptr;
    PFNGLDELETEQUERIESPROC DeleteQueries = nullptr;
    PFNGLDELETESHADERPROC DeleteShader = nullptr;
    PFNGLDISPATCHCOMPUTEPROC DispatchCompute = nullptr;
    PFNGLENDQUERYPROC EndQuery = nullptr;
    PFNGLGENBUFFERSPROC GenBuffers = nullptr;
    PFNGLGENQUERIESPROC GenQueries = nullptr;
    PFNGLGETBUFFERSUBDATAPROC GetBufferSubData = nullptr;
    PFNGLGETERRORPROC GetError = nullptr;
    PFNGLGETPROGRAMINFOLOGPROC GetProgramInfoLog = nullptr;
    PFNGLGETPROGRAMIVPROC GetProgramiv = nullptr;
    PFNGLGETQUERYOBJECTUI64VPROC GetQueryObjectui64v = nullptr;
    PFNGLGETSHADERINFOLOGPROC GetShaderInfoLog = nullptr;
    PFNGLGETSHADERIVPROC GetShaderiv = nullptr;
    PFNGLGETSTRINGPROC GetString = nullptr;
    PFNGLGETUNIFORMLOCATIONPROC GetUniformLocation = nullptr;
    PFNGLLINKPROGRAMPROC LinkProgram = nullptr;
    PFNGLMEMORYBARRIERPROC MemoryBarrier = nullptr;
    PFNGLSHADERSOURCEPROC ShaderSource = nullptr;
    PFNGLUNIFORM1UIPROC Uniform1ui = nullptr;
    PFNGLUSEPROGRAMPROC UseProgram = nullptr;
    void load() {
        AttachShader = reinterpret_cast<PFNGLATTACHSHADERPROC>(loadGl("glAttachShader"));
        BeginQuery = reinterpret_cast<PFNGLBEGINQUERYPROC>(loadGl("glBeginQuery"));
        BindBuffer = reinterpret_cast<PFNGLBINDBUFFERPROC>(loadGl("glBindBuffer"));
        BindBufferBase = reinterpret_cast<PFNGLBINDBUFFERBASEPROC>(loadGl("glBindBufferBase"));
        BufferData = reinterpret_cast<PFNGLBUFFERDATAPROC>(loadGl("glBufferData"));
        BufferSubData = reinterpret_cast<PFNGLBUFFERSUBDATAPROC>(loadGl("glBufferSubData"));
        CompileShader = reinterpret_cast<PFNGLCOMPILESHADERPROC>(loadGl("glCompileShader"));
        CreateProgram = reinterpret_cast<PFNGLCREATEPROGRAMPROC>(loadGl("glCreateProgram"));
        CreateShader = reinterpret_cast<PFNGLCREATESHADERPROC>(loadGl("glCreateShader"));
        DeleteBuffers = reinterpret_cast<PFNGLDELETEBUFFERSPROC>(loadGl("glDeleteBuffers"));
        DeleteProgram = reinterpret_cast<PFNGLDELETEPROGRAMPROC>(loadGl("glDeleteProgram"));
        DeleteQueries = reinterpret_cast<PFNGLDELETEQUERIESPROC>(loadGl("glDeleteQueries"));
        DeleteShader = reinterpret_cast<PFNGLDELETESHADERPROC>(loadGl("glDeleteShader"));
        DispatchCompute = reinterpret_cast<PFNGLDISPATCHCOMPUTEPROC>(loadGl("glDispatchCompute"));
        EndQuery = reinterpret_cast<PFNGLENDQUERYPROC>(loadGl("glEndQuery"));
        GenBuffers = reinterpret_cast<PFNGLGENBUFFERSPROC>(loadGl("glGenBuffers"));
        GenQueries = reinterpret_cast<PFNGLGENQUERIESPROC>(loadGl("glGenQueries"));
        GetBufferSubData = reinterpret_cast<PFNGLGETBUFFERSUBDATAPROC>(loadGl("glGetBufferSubData"));
        GetError = reinterpret_cast<PFNGLGETERRORPROC>(loadGl("glGetError"));
        GetProgramInfoLog = reinterpret_cast<PFNGLGETPROGRAMINFOLOGPROC>(loadGl("glGetProgramInfoLog"));
        GetProgramiv = reinterpret_cast<PFNGLGETPROGRAMIVPROC>(loadGl("glGetProgramiv"));
        GetQueryObjectui64v = reinterpret_cast<PFNGLGETQUERYOBJECTUI64VPROC>(loadGl("glGetQueryObjectui64v"));
        GetShaderInfoLog = reinterpret_cast<PFNGLGETSHADERINFOLOGPROC>(loadGl("glGetShaderInfoLog"));
        GetShaderiv = reinterpret_cast<PFNGLGETSHADERIVPROC>(loadGl("glGetShaderiv"));
        GetString = reinterpret_cast<PFNGLGETSTRINGPROC>(loadGl("glGetString"));
        GetUniformLocation = reinterpret_cast<PFNGLGETUNIFORMLOCATIONPROC>(loadGl("glGetUniformLocation"));
        LinkProgram = reinterpret_cast<PFNGLLINKPROGRAMPROC>(loadGl("glLinkProgram"));
        MemoryBarrier = reinterpret_cast<PFNGLMEMORYBARRIERPROC>(loadGl("glMemoryBarrier"));
        ShaderSource = reinterpret_cast<PFNGLSHADERSOURCEPROC>(loadGl("glShaderSource"));
        Uniform1ui = reinterpret_cast<PFNGLUNIFORM1UIPROC>(loadGl("glUniform1ui"));
        UseProgram = reinterpret_cast<PFNGLUSEPROGRAMPROC>(loadGl("glUseProgram"));
    }
};
static constexpr const char *shaderSource=R"PS2XGLSL(#version 430 core
// Ordered, pixel-owned GS rasterization. One invocation owns one color/depth
// address pair for the entire segment. Input textures are immutable snapshots.
layout(local_size_x=8, local_size_y=8) in;
layout(std430,binding=0) buffer Vram { uint ram[]; };
layout(std430,binding=1) readonly buffer Draws { uint draws[]; };
layout(std430,binding=2) readonly buffer Textures { uint texels[]; };
layout(std430,binding=3) readonly buffer Tiles { uvec2 tiles[]; };
layout(std430,binding=4) readonly buffer Indices { uint indices[]; };
layout(std430,binding=5) readonly buffer Addresses { uvec2 addresses[]; };
uniform uint width;
uniform uint height;
uniform uint tilesX;

ivec4 unpackColor(uint c) { return ivec4(c&255u,(c>>8)&255u,(c>>16)&255u,c>>24); }
uint packColor(ivec4 c) { return uint(c.x)|(uint(c.y)<<8)|(uint(c.z)<<16)|(uint(c.w)<<24); }
float f(uint b,uint i) { return uintBitsToFloat(draws[b+i]); }
int wrapCoord(int c, int sz, uint mode, uint limits) {
    int mn=int(limits&65535u), mx=int(limits>>16);
    if(mode==0u) return int(uint(c)&uint(sz-1));
    if(mode==1u) return clamp(c,0,sz-1);
    if(mode==2u) return min(max(c,mn),mx);
    return int((uint(c)&uint(mn))|uint(mx));
}
ivec4 samplePoint(uint b,int u,int v) {
    uint modes=draws[b+32u];
    u=wrapCoord(u,int(draws[b+30u]),modes&3u,draws[b+33u]);
    v=wrapCoord(v,int(draws[b+31u]),(modes>>2)&3u,draws[b+34u]);
    return unpackColor(texels[draws[b+28u]+uint(v)*draws[b+29u]+uint(u)]);
}
ivec4 sampleTexture(uint b,vec3 w) {
    uint flags=draws[b+24u];
    precise float u,v;
    if((flags&4u)!=0u) {
        precise float uf=float(draws[b+7u]&65535u)*w.x+float(draws[b+15u]&65535u)*w.y+float(draws[b+23u]&65535u)*w.z;
        precise float vf=float(draws[b+7u]>>16)*w.x+float(draws[b+15u]>>16)*w.y+float(draws[b+23u]>>16)*w.z;
        u=float(uint(uf)&65535u)/16.0;
        v=float(uint(vf)&65535u)/16.0;
    } else {
        precise float s=f(b,4u)*w.x+f(b,12u)*w.y+f(b,20u)*w.z;
        precise float t=f(b,5u)*w.x+f(b,13u)*w.y+f(b,21u)*w.z;
        precise float q=f(b,6u)*w.x+f(b,14u)*w.y+f(b,22u)*w.z;
        precise float iq=float(1.0/double((abs(q)>1e-8)?q:1.0));
        u=s*iq*float(draws[b+30u]); v=t*iq*float(draws[b+31u]);
    }
    if(isnan(u)||isinf(u)||abs(u)>1e6) u=0.0;
    if(isnan(v)||isinf(v)||abs(v)>1e6) v=0.0;
    if((flags&8u)==0u) return samplePoint(b,int(u),int(v));
    precise float su=u-0.5, sv=v-0.5;
    int x=int(floor(su)), y=int(floor(sv));
    precise float fx=su-float(x), fy=sv-float(y);
    vec4 a=vec4(samplePoint(b,x,y)), c=vec4(samplePoint(b,x,y+1));
    precise vec4 top=a+(vec4(samplePoint(b,x+1,y))-a)*fx;
    precise vec4 bottom=c+(vec4(samplePoint(b,x+1,y+1))-c)*fx;
    precise vec4 value=top+(bottom-top)*fy;
    return clamp(ivec4(floor(value+0.5)),ivec4(0),ivec4(255));
}
bool alphaPass(uint test,int a) {
    if((test&1u)==0u) return true;
    uint mode=(test>>1)&7u; int ref=int((test>>4)&255u);
    if(mode==0u) return false; if(mode==1u) return true;
    if(mode==2u) return a<ref; if(mode==3u) return a<=ref;
    if(mode==4u) return a==ref; if(mode==5u) return a>=ref;
    if(mode==6u) return a>ref; return a!=ref;
}
ivec3 pickRGB(uint sel,ivec3 s,ivec3 d) { return sel==0u?s:(sel==1u?d:ivec3(0)); }
void main() {
    uvec2 p=gl_GlobalInvocationID.xy;
    if(p.x>=width||p.y>=height) return;
    uvec2 addr=addresses[p.y*width+p.x];
    uint color=ram[addr.x], depth=ram[addr.y];
    uvec2 range=tiles[gl_WorkGroupID.y*tilesX+gl_WorkGroupID.x];
    for(uint n=0u;n<range.y;++n) {
        uint b=indices[range.x+n]*48u;
        if(p.x<draws[b+40u]||p.x>draws[b+41u]||p.y<draws[b+42u]||p.y>draws[b+43u]) continue;
        precise float x0=f(b,0u), y0=f(b,1u), x1=f(b,8u), y1=f(b,9u), x2=f(b,16u), y2=f(b,17u);
        precise float denom=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);
        if(abs(denom)<0.001) continue;
        float winding=denom<0.0?-1.0:1.0;
        precise float inv=float(1.0/double(abs(denom)));
        precise float px=float(p.x)+0.5,py=float(p.y)+0.5;
        precise float w0=(((y1-y2)*(px-x2)+(x2-x1)*(py-y2))*winding)*inv;
        precise float w1=(((y2-y0)*(px-x2)+(x0-x2)*(py-y2))*winding)*inv;
        precise float w2=1.0-w0-w1;
        if(w0< -1e-4||w1< -1e-4||w2< -1e-4) continue;
        uint flags=draws[b+24u], test=draws[b+25u];
        bool ct24=(flags&128u)!=0u;
        ivec4 c=unpackColor(draws[b+19u]);
        if((flags&1u)!=0u) {
            precise vec4 shade=vec4(unpackColor(draws[b+3u]))*w0+vec4(unpackColor(draws[b+11u]))*w1+vec4(c)*w2;
            c=clamp(ivec4(shade),ivec4(0),ivec4(255));
        }
        if((flags&2u)!=0u) {
            ivec4 t=sampleTexture(b,vec3(w0,w1,w2));
            uint tfx=draws[b+35u]&255u;
            bool tcc=(draws[b+35u]&256u)!=0u;
            ivec4 combined=t;
            if(tfx==0u) { combined.rgb=(t.rgb*c.rgb)>>7; combined.a=tcc?((t.a*c.a)>>7):c.a; }
            else if(tfx==1u) combined.a=tcc?t.a:c.a;
            else { combined.rgb=((t.rgb*c.rgb)>>7)+ivec3(c.a); combined.a=tcc?(tfx==2u?t.a+c.a:t.a):c.a; }
            c=clamp(combined,ivec4(0),ivec4(255));
        }
        if((flags&16u)!=0u) {
            precise float ff=float(draws[b+44u])*w0+float(draws[b+45u])*w1+float(draws[b+46u])*w2;
            int fog=clamp(int(ff),0,255);
            c.rgb=((fog*c.rgb)>>8)+(((255-fog)*unpackColor(draws[b+38u]).rgb)>>8);
            c.rgb=c.rgb&ivec3(255);
        }
        bool wrgb=true,wa=true,wz=true;
        if(!alphaPass(test,c.a)) {
            uint fail=(test>>12)&3u;
            if(fail==0u) continue;
            if(fail==1u) wz=false;
            if(fail==2u) { wrgb=false;wa=false; }
            if(fail==3u) { wz=false;wa=ct24; }
        }
        if(!ct24&&(test&16384u)!=0u&&((color>>31)&1u)!=((test>>15)&1u)) continue;
        precise double zd=double(draws[b+2u])*double(w0)+double(draws[b+10u])*double(w1)+double(draws[b+18u])*double(w2);
        uint z=min(uint(zd+0.5),draws[b+47u]);
        uint zold=depth&draws[b+47u], ztest=(test>>17)&3u;
        if(ztest==0u||(ztest==2u&&z<zold)||(ztest==3u&&z<=zold)) continue;
        if(wrgb||wa) {
            if((flags&32u)!=0u&&!((flags&256u)!=0u&&(c.a&128)==0)) {
                ivec4 dst=unpackColor(color); if(ct24) dst.a=128;
                uint ar=draws[b+36u],sel=(ar>>4)&3u;
                int a=sel==0u?c.a:(sel==1u?dst.a:int(draws[b+37u]&255u));
                c.rgb=clamp(((pickRGB(ar&3u,c.rgb,dst.rgb)-pickRGB((ar>>2)&3u,c.rgb,dst.rgb))*a>>7)+pickRGB((ar>>6)&3u,c.rgb,dst.rgb),ivec3(0),ivec3(255));
            }
            if(wa&&!ct24&&(draws[b+26u]&1u)!=0u) c.a|=128;
            uint outColor=packColor(c),mask=draws[b+27u];
            outColor=(outColor&~mask)|(color&mask);
            if(!wa||ct24) outColor=(outColor&0x00ffffffu)|(color&0xff000000u);
            color=outColor;
        }
        if(wz&&(flags&64u)==0u) depth=(depth&~draws[b+47u])|(z&draws[b+47u]);
    }
    ram[addr.x]=color; ram[addr.y]=depth;
}
)PS2XGLSL";
class OpenGlGsSegmentBackend {
    WorkerContext context;
    GlApi api;
    GLuint program=0,buffers[6]{},timer=0;
    uint64_t segments=0,draws=0,gpuNs=0;
    std::array<size_t,6> capacities{};
    std::array<std::vector<uint8_t>,6> shadows;
    uint64_t uploadBytes=0,readbackBytes=0;
    double uploadMs=0,readbackMs=0;
    // Exact CPU mirrors detect every intervening CPU write, including fallback
    // draws, transfers and wraparound. No hash collisions or guessed dirty ranges.
    void upload(uint32_t index,const void *data,size_t bytes,bool cache=false) {
        api.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[index]);
        api.BindBufferBase(GL_SHADER_STORAGE_BUFFER,index,buffers[index]);
        auto *src=static_cast<const uint8_t*>(data);
        if(capacities[index]!=bytes) {
            api.BufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(bytes),data,GL_DYNAMIC_DRAW);
            capacities[index]=bytes;uploadBytes+=bytes;
        } else if(index==0 && shadows[0].size()==bytes) {
            constexpr size_t page=8192;
            for(size_t start=0;start<bytes;) {
                if(std::memcmp(src+start,shadows[0].data()+start,page)==0){start+=page;continue;}
                size_t end=start+page;
                while(end<bytes && std::memcmp(src+end,shadows[0].data()+end,page)!=0)end+=page;
                api.BufferSubData(GL_SHADER_STORAGE_BUFFER,GLintptr(start),GLsizeiptr(end-start),src+start);
                uploadBytes+=end-start;start=end;
            }
        } else if(!cache || shadows[index].size()!=bytes || std::memcmp(data,shadows[index].data(),bytes)!=0) {
            api.BufferSubData(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(bytes),data);uploadBytes+=bytes;
        }
        if(cache && index!=0)shadows[index].assign(src,src+bytes);
    }
public:
    OpenGlGsSegmentBackend() {
        api.load();
        std::fprintf(stderr,"[gs:gpu] %s / %s\n",api.GetString(GL_RENDERER),api.GetString(GL_VERSION));
        const char *src=shaderSource;
        GLuint shader=api.CreateShader(GL_COMPUTE_SHADER);api.ShaderSource(shader,1,&src,nullptr);api.CompileShader(shader);
        GLint ok=0;char log[16384]{};api.GetShaderiv(shader,GL_COMPILE_STATUS,&ok);api.GetShaderInfoLog(shader,sizeof(log),nullptr,log);
        if(!ok){api.DeleteShader(shader);throw std::runtime_error(std::string("Shader compile: ")+log);}
        program=api.CreateProgram();api.AttachShader(program,shader);api.LinkProgram(program);api.DeleteShader(shader);
        api.GetProgramiv(program,GL_LINK_STATUS,&ok);api.GetProgramInfoLog(program,sizeof(log),nullptr,log);require(ok!=0,log);
        api.GenBuffers(6,buffers);api.GenQueries(1,&timer);
    }
    ~OpenGlGsSegmentBackend(){std::fprintf(stderr,"[gs:gpu:summary] segments=%llu draws=%llu dispatch=%.3fms total\n",static_cast<unsigned long long>(segments),static_cast<unsigned long long>(draws),double(gpuNs)/1e6);api.DeleteQueries(1,&timer);api.DeleteBuffers(6,buffers);if(program)api.DeleteProgram(program);}
    std::vector<uint8_t> render(const Capture &c,const Prepared &p) {
        auto beginUpload=Clock::now();
        upload(0,c.before.data(),c.before.size());
        upload(1,p.commands.data(),p.commands.size()*4);
        upload(2,p.texels.data(),p.texels.size()*4,true);
        upload(3,p.tiles.data(),p.tiles.size()*8);
        upload(4,p.indices.data(),p.indices.size()*4);
        upload(5,p.addresses.data(),p.addresses.size()*8,true);
        uploadMs+=std::chrono::duration<double,std::milli>(Clock::now()-beginUpload).count();
        require(api.GetError()==GL_NO_ERROR,"GPU buffer upload failed");
        api.UseProgram(program);api.Uniform1ui(api.GetUniformLocation(program,"width"),p.width);api.Uniform1ui(api.GetUniformLocation(program,"height"),p.height);api.Uniform1ui(api.GetUniformLocation(program,"tilesX"),p.tilesX);
        api.BeginQuery(GL_TIME_ELAPSED,timer);api.DispatchCompute(p.tilesX,p.tilesY,1);api.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);api.EndQuery(GL_TIME_ELAPSED);
        auto beginReadback=Clock::now();
        std::vector<uint8_t> result(c.before.begin(),c.before.end());
        // Every shader store is to one of these validated addresses. Coalescing
        // their physical pages handles GS swizzling and VRAM wrapping exactly.
        std::array<bool,VramBytes/8192> written{};
        for(const auto &pair:p.addresses){written[pair[0]/2048]=true;written[pair[1]/2048]=true;}
        api.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[0]);
        for(size_t start=0;start<written.size();) {
            if(!written[start]){++start;continue;}
            size_t end=start+1;while(end<written.size() && written[end])++end;
            api.GetBufferSubData(GL_SHADER_STORAGE_BUFFER,GLintptr(start*8192),GLsizeiptr((end-start)*8192),result.data()+start*8192);
            readbackBytes+=(end-start)*8192;start=end;
        }
        shadows[0]=result;
        readbackMs+=std::chrono::duration<double,std::milli>(Clock::now()-beginReadback).count();
        GLuint64 ns=0;api.GetQueryObjectui64v(timer,GL_QUERY_RESULT,&ns);
        require(api.GetError()==GL_NO_ERROR,"GPU dispatch/readback failed");
        ++segments;draws+=c.draws.size();gpuNs+=ns;
        if(segments==3 || segments%1000==0)
            std::fprintf(stderr,"[gs:gpu:transfer] segments=%llu upload=%.2fMiB/readback=%.2fMiB per segment upload=%.3fms readback/wait=%.3fms dispatch=%.3fms\n",
                static_cast<unsigned long long>(segments),double(uploadBytes)/segments/1048576,double(readbackBytes)/segments/1048576,
                uploadMs/segments,readbackMs/segments,double(gpuNs)/segments/1e6);
        return result;
    }
};

} // namespace
bool ps2xRenderGsGpuSegment(const uint8_t *vram,uint32_t size,const GSPrimitiveBatch *first,size_t stride,size_t count,std::vector<uint8_t> &output, std::vector<uint8_t> *before) {
    static const bool enabled=[] {const char *v=std::getenv("PS2X_GS_BACKEND");return v && std::strcmp(v,"gpu")==0;}();
    if(!enabled || size!=VramBytes || count<64 || count>8192)return false;
    static thread_local bool failed=false;
    static thread_local std::unique_ptr<OpenGlGsSegmentBackend> gpu;
    static thread_local uint64_t accepted=0,rejected=0;
    if(failed)return false;
    const auto &target=first->state.context;
    if((target.frame.psm!=GS_PSM_CT24 && target.frame.psm!=GS_PSM_CT32) || (target.zbuf.psm!=GS_PSM_Z24 && target.zbuf.psm!=GS_PSM_Z32))return false;
    for(size_t i=0;i<count;++i) {
        const auto &d=*reinterpret_cast<const GSPrimitiveBatch*>(reinterpret_cast<const uint8_t*>(first)+i*stride);
        const auto &k=d.state.context;
        if(d.vertexCount!=3 || d.state.prim.type<GS_PRIM_TRIANGLE || d.state.prim.type>GS_PRIM_TRIFAN ||
           k.frame.fbp!=target.frame.fbp || k.frame.fbw!=target.frame.fbw || k.frame.psm!=target.frame.psm || k.zbuf.zbp!=target.zbuf.zbp || k.zbuf.psm!=target.zbuf.psm)return false;
    }
    auto start=Clock::now();
    Capture c;c.before={const_cast<uint8_t*>(vram),size};c.draws.resize(count);
    for(size_t i=0;i<count;++i)std::memcpy(&c.draws[i],reinterpret_cast<const uint8_t*>(first)+i*stride,sizeof(GSPrimitiveBatch));
    Prepared p;
    auto prepStart=Clock::now();
    try {p=prepare(c);}catch(const std::exception &e){if(++rejected<=5)std::fprintf(stderr,"[gs:gpu] CPU fallback: %s\n",e.what());return false;}
    double prepMs=std::chrono::duration<double,std::milli>(Clock::now()-prepStart).count();
    try {
        if(!gpu)gpu=std::make_unique<OpenGlGsSegmentBackend>();
        if(before)before->assign(vram,vram+size);
        output=gpu->render(c,p);
        if(++accepted<=3 || accepted%1000==0)std::fprintf(stderr,"[gs:gpu] segments=%llu draws=%zu total=%.3fms\n",static_cast<unsigned long long>(accepted),count,std::chrono::duration<double,std::milli>(Clock::now()-start).count());
        if(accepted==3 || accepted%1000==0)std::fprintf(stderr,"[gs:gpu:prepare] %.3fms textureHits=%llu misses=%llu\n",prepMs,static_cast<unsigned long long>(textureHits),static_cast<unsigned long long>(textureMisses));
        return true;
    }catch(const std::exception &e){failed=true;std::fprintf(stderr,"[gs:gpu] Disabled after GPU error: %s\n",e.what());return false;}
}
#else
bool ps2xRenderGsGpuSegment(const uint8_t*,uint32_t,const GSPrimitiveBatch*,size_t,size_t,std::vector<uint8_t>&,std::vector<uint8_t>*){return false;}
#endif

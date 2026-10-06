#include "../motion_provenance.inc"
// OpenGL GS backend, skeleton (PS2X_GS_GPU=1).
//
// What it does: every primitive the game draws goes into a GPU render target
// (colour + depth) instead of the CPU rasterizer, with the GS scissor, depth
// test and alpha blend equation applied. At presentation the render target is
// read back into PS2 local memory so the existing display code (interlace,
// half-height, circuit blending) keeps working unchanged.
//
// What it does not do yet: textures (primitives are drawn with vertex colour
// only), alpha test, fog, per-framebuffer render targets, and rendering above
// PS2 resolution. Those are the next milestones; the point of this stage is the
// pipeline -- context, render target, draw path, presentation -- with the CPU
// rasterizer still one environment variable away.
#include "runtime/gs/gs_gl_backend.h"
#include "gs_draw_capture.inc"
#include "gs_frame_storage.inc"
#include "gs_vertex_convert.inc"
#include "gs_packed_world.inc"
#include <deque>
#ifdef _WIN32
extern bool ps2xD3D12Enabled();
extern void ps2xD3D12Frame(wotm_capture::Frame&, uint32_t, uint32_t);
extern void ps2xD3D12Movie(const uint8_t*, uint32_t, uint32_t);
#endif
#include "runtime/gs/ps2_gs_common.h"

// PS2X_GS_SHOT=<file.png>[,<frame>] writes one presented frame to a PNG so the
// GPU and CPU renderers can be compared (defined in gs_frontend.cpp).
void ps2xGsSaveFrame(const uint8_t *rgba, uint32_t width, uint32_t height, uint64_t frameIndex);
void ps2xGsSaveImage(const char *path, const uint8_t *rgba, uint32_t width, uint32_t height);

// How far down each frame buffer has been drawn this frame (gs_cpu_backend.cpp).
// The presenter uses it to tell a full frame from a 224-line field it has to
// line double; only the CPU rasterizer maintains it, so a GPU backend has to
// report its own draws or every presented frame comes out half height.
#include <atomic>
extern std::atomic<uint16_t> g_ps2xFbDrawnHeight[512];
extern std::atomic<uint64_t> g_ps2xWotmCompletedFrames;
extern std::atomic<bool> g_ps2xWotmWideActive;

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

extern int ps2xHostPadGlyphFamily(int port);
extern bool ps2xLoadXboxButtonAtlas(std::vector<uint8_t> &rgba);

// Native UI texture handle. It is intercepted before VRAM decoding; no guest
// texture memory is reserved or overwritten. CSA identifies the owning pad.
uint64_t ps2xNativeButtonTex0(int port)
{
    return 0x3fffull | (1ull << 14) | (0x13ull << 20) | (6ull << 26) |
           (6ull << 30) | (1ull << 34) | (1ull << 35) | (0x3fffull << 37) |
           (static_cast<uint64_t>(30 + (port == 1)) << 56);
}

namespace
{
    struct NativeGlGeometry {
        GSGlBackend *backend; const GSVertex *vertices; size_t count;
        const uint16_t *indices = nullptr; size_t uniqueCount = 0;
        const uint8_t *packed = nullptr;
        uint16_t u=0,v=0;
    };
    thread_local const NativeGlGeometry *s_nativeGlGeometry = nullptr;
    thread_local bool s_nearCullReferenceOnly = false;
}

void ps2xGlSubmitNearCullReference(GSGlBackend &backend, const GSPrimitiveBatch &batch)
{
    s_nearCullReferenceOnly = true;
    backend.Submit(batch);
    s_nearCullReferenceOnly = false;
}

// A scoped stream lets the existing renderer prepare the material and acquire
// its queue lock once, without widening the widely included backend interface.
void ps2xGlSubmitGeometry(GSGlBackend &backend, const GSPrimitiveBatch &prototype,
                         const GSVertex *vertices, size_t count)
{
    if (!vertices || count == 0u || count % 3u != 0u || prototype.vertexCount != 3u) return;
#if defined(_WIN32)
    const NativeGlGeometry stream{&backend, vertices, count};
    struct Scope {
        const NativeGlGeometry *previous = s_nativeGlGeometry;
        ~Scope() { s_nativeGlGeometry = previous; }
    } scope;
    s_nativeGlGeometry = &stream;
    backend.Submit(prototype);
#else
    GSPrimitiveBatch batch = prototype;
    for (size_t i = 0; i < count; i += 3) {
        std::copy_n(vertices + i, 3, batch.vertices.begin());
        backend.Submit(batch);
    }
#endif
}

// The input remains unique through frontend assembly and material preparation.
// Only the final GPU triangle stream is expanded, using one conversion per index.
void ps2xGlSubmitIndexedGeometry(GSGlBackend &backend, const GSPrimitiveBatch &prototype,
                               const GSVertex *vertices, size_t uniqueCount,
                               const uint16_t *indices, size_t count)
{
    if (!vertices || !indices || uniqueCount > 256u || count == 0u || count % 3u ||
        prototype.vertexCount != 3u) return;
#if defined(_WIN32)
    const NativeGlGeometry stream{&backend, vertices, count, indices, uniqueCount};
    struct Scope {
        const NativeGlGeometry *previous = s_nativeGlGeometry;
        ~Scope() { s_nativeGlGeometry = previous; }
    } scope;
    s_nativeGlGeometry = &stream;
    backend.Submit(prototype);
#else
    GSPrimitiveBatch batch = prototype;
    for (size_t i = 0; i < count; i += 3) {
        for (size_t j = 0; j < 3; ++j) batch.vertices[j] = vertices[indices[i+j]];
        backend.Submit(batch);
    }
#endif
}

// Packet and index pointers are scoped to this synchronous Submit call.
// Queued frames retain only owned host vertices and the existing material.
void ps2xGlSubmitPackedWorldGeometry(GSGlBackend& backend,const GSPrimitiveBatch& prototype,
    const uint8_t* raw,size_t uniqueCount,const uint16_t* indices,size_t count,uint16_t u,uint16_t v)
{
    if(!raw || !indices || uniqueCount<32u || uniqueCount>256u || count<uniqueCount ||
        count%3u || prototype.vertexCount!=3u || prototype.state.prim.fst ||
        prototype.state.prim.type!=GS_PRIM_TRISTRIP)std::abort();
    const NativeGlGeometry stream{&backend,nullptr,count,indices,uniqueCount,raw,u,v};
    struct Scope {
        const NativeGlGeometry *previous=s_nativeGlGeometry;
        ~Scope(){s_nativeGlGeometry=previous;}
    } scope;
    s_nativeGlGeometry=&stream;
    backend.Submit(prototype);
}

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <GL/gl.h>

namespace
{
    bool gpuReferenceEnabled()
    {
        static const bool enabled = [] {
            const char *value = std::getenv("PS2X_GS_GPU_REFERENCE");
            return value != nullptr && value[0] == '1';
        }();
        return enabled;
    }

    // ---- minimal GL 4.3 entry points (opengl32 exports GL 1.1 directly) ----
    using GLchar = char;
    using GLsizeiptr = ptrdiff_t;
    using GLintptr = ptrdiff_t;

    void *glProc(const char *name)
    {
        PROC p = wglGetProcAddress(name);
        if (!p || p == reinterpret_cast<PROC>(1) || p == reinterpret_cast<PROC>(2) ||
            p == reinterpret_cast<PROC>(3) || p == reinterpret_cast<PROC>(-1))
            p = GetProcAddress(GetModuleHandleW(L"opengl32.dll"), name);
        return reinterpret_cast<void *>(p);
    }

    struct Gl
    {
        void (APIENTRY *GenVertexArrays)(GLsizei, GLuint *) = nullptr;
        void (APIENTRY *BindVertexArray)(GLuint) = nullptr;
        void (APIENTRY *GenBuffers)(GLsizei, GLuint *) = nullptr;
        void (APIENTRY *BindBuffer)(GLenum, GLuint) = nullptr;
        void (APIENTRY *BufferData)(GLenum, GLsizeiptr, const void *, GLenum) = nullptr;
        void (APIENTRY *VertexAttribPointer)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void *) = nullptr;
        void (APIENTRY *EnableVertexAttribArray)(GLuint) = nullptr;
        GLuint (APIENTRY *CreateShader)(GLenum) = nullptr;
        void (APIENTRY *ShaderSource)(GLuint, GLsizei, const GLchar *const *, const GLint *) = nullptr;
        void (APIENTRY *CompileShader)(GLuint) = nullptr;
        void (APIENTRY *GetShaderiv)(GLuint, GLenum, GLint *) = nullptr;
        void (APIENTRY *GetShaderInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *) = nullptr;
        GLuint (APIENTRY *CreateProgram)() = nullptr;
        void (APIENTRY *AttachShader)(GLuint, GLuint) = nullptr;
        void (APIENTRY *LinkProgram)(GLuint) = nullptr;
        void (APIENTRY *GetProgramiv)(GLuint, GLenum, GLint *) = nullptr;
        void (APIENTRY *GetProgramInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *) = nullptr;
        void (APIENTRY *UseProgram)(GLuint) = nullptr;
        void (APIENTRY *DeleteShader)(GLuint) = nullptr;
        GLint (APIENTRY *GetUniformLocation)(GLuint, const GLchar *) = nullptr;
        void (APIENTRY *Uniform2f)(GLint, GLfloat, GLfloat) = nullptr;
        void (APIENTRY *Uniform1i)(GLint, GLint) = nullptr;
        void (APIENTRY *Uniform1f)(GLint, GLfloat) = nullptr;
        void (APIENTRY *Uniform2i)(GLint, GLint, GLint) = nullptr;
        void (APIENTRY *Uniform3f)(GLint, GLfloat, GLfloat, GLfloat) = nullptr;
        void (APIENTRY *Uniform4f)(GLint, GLfloat, GLfloat, GLfloat, GLfloat) = nullptr;
        void (APIENTRY *GenFramebuffers)(GLsizei, GLuint *) = nullptr;
        void (APIENTRY *BindFramebuffer)(GLenum, GLuint) = nullptr;
        void (APIENTRY *FramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint) = nullptr;
        GLenum (APIENTRY *CheckFramebufferStatus)(GLenum) = nullptr;
        void (APIENTRY *BlendFuncSeparate)(GLenum, GLenum, GLenum, GLenum) = nullptr;
        void (APIENTRY *BlendEquationSeparate)(GLenum, GLenum) = nullptr;
        void (APIENTRY *BlendColor)(GLfloat, GLfloat, GLfloat, GLfloat) = nullptr;
        void (APIENTRY *Uniform4fv)(GLint, GLsizei, const GLfloat *) = nullptr;
        void (APIENTRY *ClipControl)(GLenum, GLenum) = nullptr; // optional (GL 4.5)
        void (APIENTRY *CopyImageSubData)(GLuint, GLenum, GLint, GLint, GLint, GLint, GLuint, GLenum, GLint, GLint,
                                          GLint, GLint, GLsizei, GLsizei, GLsizei) = nullptr;

        bool load()
        {
            auto get = [](auto &fn, const char *name)
            {
                fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(glProc(name));
                return fn != nullptr;
            };
            return get(GenVertexArrays, "glGenVertexArrays") && get(BindVertexArray, "glBindVertexArray") &&
                   get(GenBuffers, "glGenBuffers") && get(BindBuffer, "glBindBuffer") &&
                   get(BufferData, "glBufferData") && get(VertexAttribPointer, "glVertexAttribPointer") &&
                   get(EnableVertexAttribArray, "glEnableVertexAttribArray") && get(CreateShader, "glCreateShader") &&
                   get(ShaderSource, "glShaderSource") && get(CompileShader, "glCompileShader") &&
                   get(GetShaderiv, "glGetShaderiv") && get(GetShaderInfoLog, "glGetShaderInfoLog") &&
                   get(CreateProgram, "glCreateProgram") && get(AttachShader, "glAttachShader") &&
                   get(LinkProgram, "glLinkProgram") && get(GetProgramiv, "glGetProgramiv") &&
                   get(GetProgramInfoLog, "glGetProgramInfoLog") && get(UseProgram, "glUseProgram") &&
                   get(DeleteShader, "glDeleteShader") && get(GetUniformLocation, "glGetUniformLocation") &&
                   get(Uniform2f, "glUniform2f") && get(Uniform1i, "glUniform1i") &&
                   get(Uniform1f, "glUniform1f") && get(Uniform2i, "glUniform2i") &&
                   get(Uniform3f, "glUniform3f") && get(Uniform4f, "glUniform4f") &&
                   get(GenFramebuffers, "glGenFramebuffers") &&
                   get(BindFramebuffer, "glBindFramebuffer") && get(FramebufferTexture2D, "glFramebufferTexture2D") &&
                   get(CheckFramebufferStatus, "glCheckFramebufferStatus") &&
                   get(BlendFuncSeparate, "glBlendFuncSeparate") &&
                   get(BlendEquationSeparate, "glBlendEquationSeparate") && get(BlendColor, "glBlendColor") &&
                   get(Uniform4fv, "glUniform4fv");
        }

        void loadOptional()
        {
            ClipControl = reinterpret_cast<decltype(ClipControl)>(glProc("glClipControl"));
            CopyImageSubData = reinterpret_cast<decltype(CopyImageSubData)>(glProc("glCopyImageSubData"));
        }
    };

    // Optional, asynchronous timing of the GL replay command span. Query
    // availability is checked first: instrumentation never waits for the GPU.
    // This span includes gaps while the CPU submits commands, not GPU busy time.
    struct ReplayTiming {
        void (APIENTRY *gen)(GLsizei,GLuint*)=nullptr;
        void (APIENTRY *begin)(GLenum,GLuint)=nullptr;
        void (APIENTRY *end)(GLenum)=nullptr;
        void (APIENTRY *available)(GLuint,GLenum,GLint*)=nullptr;
        void (APIENTRY *result)(GLuint,GLenum,uint64_t*)=nullptr;
        GLuint queries[8]{}; bool pending[8]{};
        uint64_t frames[8]{}, completed[8]{}; double cpuMs[8]{}; size_t draws[8]{};
        unsigned next=0; int active=-1; bool initialized=false;
        FILE *file=nullptr;
        std::chrono::steady_clock::time_point started;
        ~ReplayTiming() {if(file) std::fclose(file);}
        void start(uint64_t frame,size_t count) {
            if(!initialized) {
                initialized=true;
                const char *path=std::getenv("PS2X_GL_TIMING_CSV");
                if(!path || !*path) return;
                gen=reinterpret_cast<decltype(gen)>(glProc("glGenQueries"));
                begin=reinterpret_cast<decltype(begin)>(glProc("glBeginQuery"));
                end=reinterpret_cast<decltype(end)>(glProc("glEndQuery"));
                available=reinterpret_cast<decltype(available)>(glProc("glGetQueryObjectiv"));
                result=reinterpret_cast<decltype(result)>(glProc("glGetQueryObjectui64v"));
                if(!gen || !begin || !end || !available || !result) return;
                file=std::fopen(path,"w"); if(!file) return;
                std::fprintf(file,"present,completed_at_submit,draws,cpu_submit_ms,gpu_span_ms\n");gen(8,queries);
            }
            if(!file) return;
            for(unsigned i=0;i<8;++i) if(pending[i]) {
                GLint ready=0;available(queries[i],0x8867,&ready);
                if(ready) {
                    uint64_t ns=0;result(queries[i],0x8866,&ns);pending[i]=false;
                    std::fprintf(file,"%llu,%llu,%zu,%.6f,%.6f\n",frames[i],completed[i],draws[i],cpuMs[i],double(ns)/1e6);
                }
            }
            if(frame%120==0) std::fflush(file);
            if(pending[next]) return;
            active=int(next);next=(next+1)%8;
            frames[active]=frame;completed[active]=g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);draws[active]=count;
            started=std::chrono::steady_clock::now();begin(0x88BF,queries[active]);
        }
        void finish() {
            if(active<0) return;
            end(0x88BF);
            cpuMs[active]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            pending[active]=true;active=-1;
        }
    };

    constexpr GLenum kArrayBuffer = 0x8892;
    constexpr GLenum kStreamDraw = 0x88E0;
    constexpr GLenum kFragmentShader = 0x8B30;
    constexpr GLenum kVertexShader = 0x8B31;
    constexpr GLenum kCompileStatus = 0x8B81;
    constexpr GLenum kLinkStatus = 0x8B82;
    constexpr GLenum kFramebuffer = 0x8D40;
    constexpr GLenum kColorAttachment0 = 0x8CE0;
    constexpr GLenum kDepthAttachment = 0x8D00;
    constexpr GLenum kFramebufferComplete = 0x8CD5;
    constexpr GLenum kDepthComponent24 = 0x81A6;
    constexpr GLenum kDepthComponent32F = 0x8CAC;
    constexpr GLenum kLowerLeft = 0x8CA1;
    constexpr GLenum kZeroToOne = 0x935F;
    constexpr GLenum kClampToEdge = 0x812F; // GL 1.2, not in the GL 1.1 header
    constexpr GLenum kFuncAdd = 0x8006;
    constexpr GLenum kFuncSubtract = 0x800A;
    constexpr GLenum kFuncReverseSubtract = 0x800B;
    constexpr GLenum kConstantAlpha = 0x8003;
    constexpr GLenum kOneMinusConstantAlpha = 0x8004;
    constexpr GLenum kDstAlpha = 0x0304;
    constexpr GLenum kOneMinusDstAlpha = 0x0305;
    constexpr GLenum kTextureMaxLevel = 0x813D;
    constexpr GLenum kTextureLodBias = 0x8501;
    constexpr GLenum kNearestMipmapNearest = 0x2700;
    constexpr GLenum kLinearMipmapNearest = 0x2701;
    constexpr GLenum kNearestMipmapLinear = 0x2702;
    constexpr GLenum kLinearMipmapLinear = 0x2703;

    // GS TEX1.MMIN -> a GL minification filter.
    inline GLenum minFilterFor(uint32_t mmin, bool hasMips)
    {
        if (!hasMips)
            return (mmin == 1u || mmin >= 4u) ? GL_LINEAR : GL_NEAREST;
        switch (mmin)
        {
        case 2u: return kNearestMipmapNearest;
        case 3u: return kNearestMipmapLinear;
        case 4u: return kLinearMipmapNearest;
        case 5u: return kLinearMipmapLinear;
        case 1u: return GL_LINEAR;
        default: return GL_NEAREST;
        }
    }

    // Offscreen OpenGL 4.3 context (the drawing thread owns it).
class GlContext
    {
    public:
        // Leaves the caller's context current; claim() activates ours. The
        // caller is the host renderer's thread, so its context is raylib's:
        // sharing with it lets raylib draw our render target directly.
        bool create()
        {
            HDC callerDc = wglGetCurrentDC();
            HGLRC callerContext = wglGetCurrentContext();
            const bool ok = createInternal(callerContext);
            wglMakeCurrent(callerDc, callerContext);
            return ok;
        }

        bool sharesWithHost() const { return m_shared; }

        bool created() const { return m_context != nullptr; }

        // Presentation runs on the host renderer's thread, which has raylib's
        // context current: borrow the thread, then give its context back, or
        // raylib ends up drawing into ours.
        void claim()
        {
            if (!m_context)
                return;
            m_previousDc = wglGetCurrentDC();
            m_previousContext = wglGetCurrentContext();
            const BOOL ok = wglMakeCurrent(m_dc, m_context);
            static int s_reports = 0;
            if (!ok && s_reports++ < 4)
                std::fprintf(stderr, "[gs:gl] wglMakeCurrent(ours) failed: %lu\n", GetLastError());
        }

        void release()
        {
            if (!m_context)
                return;
            wglMakeCurrent(m_previousDc, m_previousContext);
            m_previousDc = nullptr;
            m_previousContext = nullptr;
        }

        ~GlContext()
        {
            if (m_context)
            {
                wglMakeCurrent(nullptr, nullptr);
                wglDeleteContext(m_context);
            }
            if (m_dc && m_window)
                ReleaseDC(m_window, m_dc);
            if (m_window)
                DestroyWindow(m_window);
        }

    private:
        bool createInternal(HGLRC shareWith)
        {
            static std::once_flag once;
            bool registered = true;
            std::call_once(once,
                           [&registered]
                           {
                               WNDCLASSW wc{};
                               wc.style = CS_OWNDC;
                               wc.lpfnWndProc = DefWindowProcW;
                               wc.hInstance = GetModuleHandleW(nullptr);
                               wc.lpszClassName = L"PS2XGsGl";
                               registered = RegisterClassW(&wc) != 0 || GetLastError() == ERROR_CLASS_ALREADY_EXISTS;
                           });
            if (!registered)
                return false;
            m_window = CreateWindowW(L"PS2XGsGl", L"GS", WS_POPUP, 0, 0, 16, 16, nullptr, nullptr,
                                     GetModuleHandleW(nullptr), nullptr);
            if (!m_window)
                return false;
            m_dc = GetDC(m_window);
            if (!m_dc)
                return false;
            PIXELFORMATDESCRIPTOR pfd{};
            pfd.nSize = sizeof(pfd);
            pfd.nVersion = 1;
            pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
            pfd.iPixelType = PFD_TYPE_RGBA;
            pfd.cColorBits = 32;
            const int format = ChoosePixelFormat(m_dc, &pfd);
            if (!format || !SetPixelFormat(m_dc, format, &pfd))
                return false;
            HGLRC bootstrap = wglCreateContext(m_dc);
            if (!bootstrap || !wglMakeCurrent(m_dc, bootstrap))
                return false;
            using Create = HGLRC(WINAPI *)(HDC, HGLRC, const int *);
            auto create = reinterpret_cast<Create>(glProc("wglCreateContextAttribsARB"));
            if (create)
            {
                const int attrs[] = {0x2091, 4, 0x2092, 5, 0x9126, 1, 0}; // 4.5 core (glClipControl)
                HGLRC modern = shareWith ? create(m_dc, shareWith, attrs) : nullptr;
                m_shared = modern != nullptr;
                if (!modern)
                    modern = create(m_dc, nullptr, attrs); // unshared: read back instead
                if (modern)
                {
                    wglMakeCurrent(nullptr, nullptr);
                    wglDeleteContext(bootstrap);
                    m_context = modern;
                    return wglMakeCurrent(m_dc, m_context) != 0;
                }
            }
            m_context = bootstrap;
            return true;
        }

        // Presentation runs on the host renderer's thread, which already has
        // raylib's context current: borrow the thread, then give its context
        // back, or raylib draws into ours and crashes.

        HWND m_window = nullptr;
        HDC m_dc = nullptr;
        HGLRC m_context = nullptr;
        HDC m_previousDc = nullptr;
        HGLRC m_previousContext = nullptr;
        bool m_shared = false;
    };

    constexpr const char *kVertexShaderSource = R"GLSL(#version 430 core
layout(location = 0) in vec2 aPos;    // GS screen pixels
layout(location = 1) in float aDepth; // 0..1
layout(location = 2) in vec4 aColor;  // rgb = value/255, a = value/128
layout(location = 3) in vec3 aTex;    // S,T,Q (already divided for UV mode)
layout(location = 4) in float aFog;  // per-vertex FOG, 0..255
uniform vec2 uTargetSize;
// 1 when glClipControl put clip space Z in 0..1, so the PS2 depth can be used
// as it is; otherwise it has to survive the -1..1 round trip.
uniform int uZeroToOne;
out vec4 vColor;
flat out vec4 vFlatColor;
out vec3 vTex;
out float vFog;
void main()
{
    vec2 ndc = vec2(aPos.x / uTargetSize.x * 2.0 - 1.0, 1.0 - aPos.y / uTargetSize.y * 2.0);
    gl_Position = vec4(ndc, uZeroToOne == 1 ? aDepth : aDepth * 2.0 - 1.0, 1.0);
    vColor = aColor;
    vFlatColor = aColor;
    vTex = aTex;
    vFog = aFog;
}
)GLSL";

    constexpr const char *kFragmentShaderSource = R"GLSL(#version 430 core
in vec4 vColor;
flat in vec4 vFlatColor;
in vec3 vTex;
in float vFog;
out vec4 oColor;
uniform int uDebug;     // 0 normal, 1 depth, 2 wireframe, 3 texel only, 4 vertex colour only
uniform int uTextured;  // 0 = vertex colour only
uniform int uTcc;       // 1 = take alpha from the texture
uniform int uTfx;       // 0 modulate, 1 decal, 2 highlight, 3 highlight2
uniform int uIip;       // 1 = Gouraud colour/alpha, 0 = final vertex (flat)
uniform int uAlphaTest; // 1 = discard failing fragments
uniform int uAtst;      // GS ATST
uniform float uAref;    // 0..1
uniform ivec2 uWrap;    // CLAMP mode per axis: 0 repeat, 1 clamp, 2/3 region
uniform vec4 uRegion;   // MINU, MAXU, MINV, MAXV (texels)
uniform vec2 uTexSize;
uniform int uSceneRefraction;
uniform int uFog;         // PRIM.FGE
uniform vec3 uFogColor;   // FOGCOL, 0..1
uniform int uInvertAlphaTest; // second pass of an AFAIL mode: keep what failed
uniform int uPabe;            // per-pixel alpha blending
uniform int uDither;          // DTHE
uniform vec4 uDimx[4];        // DIMX, 4x4 signed offsets in 0..255 units
uniform sampler2D uTexture;

// GS CLAMP. Modes 0 and 1 are the sampler's own wrap modes; the two region
// modes restrict the addressed area and have to be done on the texel index.
float wrapAxis(float texel, int mode, float lo, float hi)
{
    if (mode == 2) // REGION_CLAMP
        return clamp(texel, lo, hi);
    if (mode == 3) // REGION_REPEAT: (u & MINU) | MAXU
        return float((int(texel) & int(lo)) | int(hi));
    return texel;
}
void main()
{
    if (uDebug == 1)
    {
        float d = pow(gl_FragCoord.z, 8.0);
        oColor = vec4(vec3(d), 1.0);
    }
    else if (uDebug == 2)
    {
        oColor = vec4(0.0, 1.0, 0.0, 1.0);
    }
    else
    {
        vec4 colour = uIip == 1 ? vColor : vFlatColor;
        vec4 texel = vec4(0.0);
        if (uDebug == 4)
        {
            oColor = vec4(vColor.rgb, 1.0);
            return;
        }
        if (uTextured == 1)
        {
            // Perspective-correct in GS terms: S,T,Q are interpolated linearly
            // in screen space and divided per pixel.
            vec2 st = vTex.xy / abs(vTex.z) * uTexSize;
            // A near-zero Q (the sky dome has vertices with Q = 0) makes this
            // infinite or NaN. The CPU rasterizer clamps the same way, so a bad
            // primitive samples texel 0 instead of reading arbitrary memory.
            if (any(isnan(st)) || any(isinf(st)) || any(greaterThan(abs(st), vec2(1.0e6))))
                st = vec2(0.0);
            st.x = wrapAxis(st.x, uWrap.x, uRegion.x, uRegion.y);
            st.y = wrapAxis(st.y, uWrap.y, uRegion.z, uRegion.w);
            vec2 uv = st / uTexSize;
            // TEX0 describes a 1024x256 view of a 640x224 displayed field.
            // GL framebuffer storage is bottom-up; ordinary uploaded textures are not.
            if (uSceneRefraction != 0)
                uv = vec2(st.x / 640.0, 1.0 - st.y / 224.0);
            texel = texture(uTexture, uv);
            // Alpha is compared and blended on the 128 = 1.0 scale, colour on
            // the 255 = 1.0 scale, so they are converted separately.
            float texAlpha = texel.a * (255.0 / 128.0);
            if (uTfx == 1) // DECAL: Cv = Ct, Av = TCC ? At : As
            {
                colour.rgb = texel.rgb;
                colour.a = uTcc == 1 ? texAlpha : colour.a;
            }
            else
            {
                // MODULATE: Cv = (Cs * Ct) >> 7. Both operands arrive as
                // value/255, so the exact scale is 255/128, not 2 (which is
                // 0.4% bright).
                colour.rgb = texel.rgb * colour.rgb * (255.0 / 128.0);
                colour.a = uTcc == 1 ? texAlpha * colour.a : colour.a;
                if (uTfx >= 2) // HIGHLIGHT / HIGHLIGHT2: Cv = (Cs * Ct >> 7) + As
                {
                    colour.rgb += vec3((uIip == 1 ? vColor.a : vFlatColor.a) * (128.0 / 255.0));
                    if (uTcc == 1)
                        colour.a = uTfx == 2 ? texAlpha + (uIip == 1 ? vColor.a : vFlatColor.a) : texAlpha;
                }
            }
        }
        if (uFog == 1)
        {
            // GS: Cv = (F * C >> 8) + ((255 - F) * FOGCOL >> 8).
            colour.rgb = colour.rgb * (vFog / 256.0) + uFogColor * ((255.0 - vFog) / 256.0);
        }
        if (uAlphaTest == 1)
        {
            // GS ATST: 0 NEVER, 1 ALWAYS, 2 LESS, 3 LEQUAL, 4 EQUAL,
            // 5 GEQUAL, 6 GREATER, 7 NOTEQUAL. Compared on the GS's own 0..255
            // alpha scale, so AREF goes in raw and the fragment alpha (carried
            // at 128 = 1.0) is scaled back up.
            float alpha8 = colour.a * 128.0;
            bool pass = true;
            if (uAtst == 0) pass = false;                              // NEVER
            else if (uAtst == 2) pass = alpha8 < uAref;                // LESS
            else if (uAtst == 3) pass = alpha8 <= uAref;               // LEQUAL
            else if (uAtst == 4) pass = abs(alpha8 - uAref) < 0.5;     // EQUAL
            else if (uAtst == 5) pass = alpha8 >= uAref;               // GEQUAL
            else if (uAtst == 6) pass = alpha8 > uAref;                // GREATER
            else if (uAtst == 7) pass = abs(alpha8 - uAref) >= 0.5;    // NOTEQUAL
            if (uInvertAlphaTest == 1)
                pass = !pass; // AFAIL pass: draw exactly the fragments that failed
            if (!pass)
                discard;
        }
        if (uDither == 1)
        {
            ivec2 cell = ivec2(gl_FragCoord.xy) & 3;
            colour.rgb += vec3(uDimx[cell.y][cell.x] / 255.0);
        }
        if (uPabe == 1 && colour.a < 1.0)
        {
            // Alpha bit 7 clear: this pixel is not blended. The blend is set up
            // as Cs*As + Cd*(1-As), so an alpha of 1 reproduces Cs exactly.
            colour.a = 1.0;
        }
        if (uDebug == 3)
            oColor = uTextured == 1 ? vec4(texel.rgb, 1.0)
                                    : vec4(1.0, 0.0, 1.0, 1.0); // magenta = untextured
        else
            oColor = clamp(colour, 0.0, 1.0);
    }
}
)GLSL";

    using GlVertex = wotm_capture::Vertex;



    // resize starts trivial vertex lifetimes without a redundant zero fill.
    // Every appended element is filled before it can be submitted.
    template<class T> struct VertexAllocator : std::allocator<T> {
        using value_type=T;
        template<class U> struct rebind {using other=VertexAllocator<U>;};
        VertexAllocator()=default;
        template<class U> VertexAllocator(const VertexAllocator<U>&) noexcept {}
        void construct(T* p) {::new(static_cast<void*>(p)) T;}
        template<class U,class... A> void construct(U* p,A&&... a) {
            ::new(static_cast<void*>(p)) U(std::forward<A>(a)...);
        }
        template<class U> bool operator==(const VertexAllocator<U>&) const noexcept {return true;}
    };
    static_assert(std::is_trivial_v<GlVertex> && sizeof(GlVertex)==44);
    using GlVertices=std::vector<GlVertex,VertexAllocator<GlVertex>>;

    // One decoded PS2 texture. Decoding happens on the game thread (it reads PS2
    // memory); the upload happens on the GL thread. Once decoded an entry never
    // changes: when the game overwrites the memory it came from, the entry is
    // retired and the next use decodes a new one. That is what lets the game
    // thread record frame N+1 while the GL thread still draws frame N.
    struct TextureEntry
    {
        bool wotmFaceButtons = false;
        std::vector<uint8_t> rgba;
        uint32_t width = 1u, height = 1u;
        // Mip levels 1..MXL, each half the size of the one before.
        struct Level
        {
            std::vector<uint8_t> rgba;
            uint32_t width = 1u, height = 1u;
        };
        std::vector<Level> mips;
        bool uploaded = false;
        // A transfer replayed into a render target: used by exactly one draw
        // and then dropped, so it shares one reusable GL texture instead of
        // allocating (and leaking) one of its own every frame.
        bool transient = false;
        unsigned int glId = 0u;
        // Source footprint in 256-byte VRAM blocks: texels, then palette.
        uint32_t texBegin = 0u, texEnd = 0u;
        uint32_t clutBegin = 0u, clutEnd = 0u;

        bool sourcedFrom(uint32_t begin, uint32_t end) const
        {
            return (texBegin < end && begin < texEnd) || (clutBegin < clutEnd && clutBegin < end && begin < clutEnd);
        }
    };

    // Conservative VRAM footprint of a rectangle, in 256-byte blocks widened to
    // whole 8 KiB pages (swizzling scatters a rectangle across its pages).
    inline void blockRange(uint32_t baseBlock, uint32_t widthPixels, uint32_t heightPixels, uint32_t psm,
                           uint32_t &begin, uint32_t &end)
    {
        // Count storage pages, not packed pixel bytes. CT24 and the high-byte
        // indexed formats occupy CT32 pages; short, wide rectangles can touch
        // several pages even when their packed bytes fit in one page.
        uint32_t pageWidth = 64u, pageHeight = 32u;
        switch (psm)
        {
        case GS_PSM_CT16: case GS_PSM_CT16S: case GS_PSM_Z16: case GS_PSM_Z16S:
            pageHeight = 64u;
            break;
        case GS_PSM_T8:
            pageWidth = 128u; pageHeight = 64u;
            break;
        case GS_PSM_T4:
            pageWidth = 128u; pageHeight = 128u;
            break;
        }
        const uint64_t pagesX = (static_cast<uint64_t>(std::max<uint32_t>(widthPixels, 1u)) + pageWidth - 1u) / pageWidth;
        const uint64_t pagesY = (static_cast<uint64_t>(std::max<uint32_t>(heightPixels, 1u)) + pageHeight - 1u) / pageHeight;
        baseBlock &= 0x3FFFu;
        begin = baseBlock & ~31u;
        const uint64_t limit = (baseBlock + pagesX * pagesY * 32u + 31u) & ~31ull;
        // VRAM addresses wrap at 4 MiB. A single interval cannot represent the
        // two halves of a wrapped range, so conservatively cover all VRAM.
        if (limit > 0x4000u)
        {
            begin = 0u;
            end = 0x4000u;
        }
        else
            end = static_cast<uint32_t>(limit);
    }

    constexpr uint32_t kNoFbp = 0xFFFFFFFFu;

    inline bool isPaletted(uint8_t psm)
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T8H || psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
    }
}

struct GSGlBackend::Impl
{
    Gl gl{};
    ReplayTiming replayTiming;
    GlContext context{};
    std::mutex glMutex; // guards the context and all GL state below
    bool ready = false;
    bool failed = false;
    std::thread::id thread{};

    GLuint program = 0u;
    GLuint vao = 0u;
    GLuint vbo = 0u;
    GLuint sceneSnapshot = 0u; // Reused GPU-only copy for same-target refraction.
    // One render target per FRAME buffer the game draws into, created on first
    // use. The game's own double buffering then gives us ping-pong for free:
    // the buffer being displayed is not the one being drawn.
    struct RenderTarget
    {
        GLuint fbo = 0u;
        GLuint color = 0u;
        uint32_t passHeight = 1u; // tallest scissor this buffer has been drawn with
        uint64_t lastFrame = 0u;
        uint32_t attachedZbp = ~0u; // which depth texture its FBO currently has
    };
    std::map<uint32_t, RenderTarget> targets; // keyed by FRAME.FBP (pages)

    // Trace only. The scissor can hold still while the picture moves, because a
    // translation lives in XYOFFSET, not in the scissor. These record the
    // vertical origin the scene was submitted with and the vertex Y extent it
    // covered, so a bobbing picture can be attributed rather than guessed at.
    static bool tracing()
    {
        static const bool on = std::getenv("PS2X_GS_GPU_TRACE") != nullptr;
        return on;
    }
    std::map<int32_t, size_t> traceOfy;
    float traceMinY = 0.0f;
    float traceMaxY = 0.0f;
    bool traceAnyY = false;

    // Depth is addressed separately from colour on the GS, and this game aims
    // several frame buffers at one Z buffer, so depth lives in its own store.
    struct DepthTarget
    {
        GLuint texture = 0u;
        uint64_t lastFrame = ~0ull;
    };
    std::map<uint32_t, DepthTarget> depths; // keyed by ZBUF.ZBP (pages)

    DepthTarget *depthFor(uint32_t zbp)
    {
        auto it = depths.find(zbp);
        if (it != depths.end())
            return &it->second;
        if (depths.size() >= 16u)
            return nullptr;
        DepthTarget target{};
        glGenTextures(1, &target.texture);
        glBindTexture(GL_TEXTURE_2D, target.texture);
        glTexImage2D(GL_TEXTURE_2D, 0, zeroToOne ? kDepthComponent32F : kDepthComponent24,
                     static_cast<GLsizei>(width), static_cast<GLsizei>(height), 0, GL_DEPTH_COMPONENT,
                     zeroToOne ? GL_FLOAT : GL_UNSIGNED_INT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        check("depth target");
        return &depths.emplace(zbp, target).first->second;
    }
    uint32_t presentedFbp = 0u;
    GLint targetSizeUniform = -1;
    GLint debugUniform = -1;
    GLint texturedUniform = -1, tccUniform = -1, tfxUniform = -1, iipUniform = -1;
    GLint alphaTestUniform = -1, atstUniform = -1, arefUniform = -1;
    GLint wrapUniform = -1, regionUniform = -1, texSizeUniform = -1, sceneRefractionUniform = -1;
    GLint zeroToOneUniform = -1;
    GLint fogUniform = -1, fogColorUniform = -1;
    GLint invertAlphaTestUniform = -1, pabeUniform = -1, ditherUniform = -1, dimxUniform = -1;
    bool zeroToOne = false;

    // The PS2's own frame buffer size, and the pixels we actually render into.
    static constexpr uint32_t kNativeWidth = 640u;
    static constexpr uint32_t kNativeHeight = 448u;
    uint32_t scale = 1u;
    uint32_t width = kNativeWidth;
    uint32_t height = kNativeHeight;

    // The game thread only records geometry (no GL calls -- an OpenGL context
    // can only be current on one thread). Present replays the list.
    struct DrawCall
    {
        size_t first = 0u, count = 0u;
        std::shared_ptr<MotionProvenance::Mesh> motion;
        uint64_t alpha = 0u, test = 0u;
        uint32_t zmsk = 0u;
        bool abe = false;
        int sx = 0, sy = 0, sw = 0, sh = 0;
        std::shared_ptr<TextureEntry> texture;
        bool linear = false;
        bool repeatU = false, repeatV = false;
        uint32_t tcc = 0u, tfx = 0u;
        uint32_t iip = 1u;
        uint32_t alphaTest = 0u, atst = 7u;
        float aref = 0.0f;
        int wrapU = 0, wrapV = 0;
        float region[4]{};
        uint32_t tbp0 = 0u, cbp = 0u;
        uint8_t texPsm = 0u;
        uint32_t framePsm = ~0u;
        uint32_t fbp = 0u;                  // which render target this draws into
        uint32_t zbp = 0u;                  // and which Z buffer it tests against
        uint32_t textureFbp = kNoFbp;       // sample that target instead of VRAM
        bool clear = false;                 // a framebuffer clear, not a draw
        float clearColor[4]{};
        bool fog = false;
        float fogColor[3]{};
        uint32_t afail = 0u;                // 0 keep, 1 fb only, 2 zb only, 3 rgb only
        bool pabe = false;
        bool dither = false;
        float dimx[16]{};
        uint32_t fbmsk = 0u;
        uint32_t mmin = 0u;   // TEX1.MMIN
        bool mmag = false;    // TEX1.MMAG
        float lodBias = 0.0f; // TEX1.K

        bool sameStateAs(const DrawCall &o) const
        {
            return motion == o.motion && alpha == o.alpha && test == o.test && zmsk == o.zmsk && abe == o.abe && sx == o.sx && sy == o.sy &&
                   sw == o.sw && sh == o.sh && texture == o.texture && linear == o.linear && repeatU == o.repeatU &&
                   repeatV == o.repeatV && tcc == o.tcc && tfx == o.tfx && iip == o.iip && alphaTest == o.alphaTest && atst == o.atst &&
                   aref == o.aref && wrapU == o.wrapU && wrapV == o.wrapV && framePsm == o.framePsm && fbp == o.fbp &&
                   textureFbp == o.textureFbp && zbp == o.zbp && fog == o.fog && afail == o.afail && pabe == o.pabe &&
                   dither == o.dither && fbmsk == o.fbmsk && mmin == o.mmin && mmag == o.mmag &&
                   lodBias == o.lodBias && std::memcmp(dimx, o.dimx, sizeof(dimx)) == 0 &&
                   std::memcmp(fogColor, o.fogColor, sizeof(fogColor)) == 0 &&
                   std::memcmp(region, o.region, sizeof(region)) == 0;
        }
    };
    // Consecutive triangles of a mesh share these prepared material registers.
    // Reuse ends on a register change, VRAM transfer, clear, reset or new frame.
    GSDrawState preparedState{};
    DrawCall preparedCall{};
    uint64_t preparedFrame = 0u;
    size_t preparedCallCount = 0u;
    int preparedGlyphFamily = -1;
    bool preparedValid = false;
    uint64_t preparedHits = 0u, preparedMisses = 0u, preparedChecks = 0u;
    std::mutex recordMutex;
    GlVertices vertices;
    std::vector<DrawCall> calls;
    struct CompletedDraws {GlVertices vertices;std::vector<DrawCall> calls;GSPresentationRequest display;uint64_t frame=0;};
    std::deque<CompletedDraws> completedDraws;
    // Empty CPU recording storage only; no queued or in-flight frame is reused.
    GsFrameStoragePool<GlVertices,std::vector<DrawCall>> frameStorage;
    uint64_t storagePresents=0,storageVertexGrowths=0,storageVerticesMoved=0;
    static bool reuseFrameStorage() {
        static const bool enabled=[] {const char* p=std::getenv("PS2X_GS_FRAME_STORAGE");return !p || p[0]!='0';}();
        return enabled;
    }
    static bool frameStorageStats() {
        static const bool enabled=std::getenv("PS2X_GS_FRAME_STORAGE_STATS")!=nullptr;
        return enabled;
    }
    void acquireFrameStorage(GlVertices& v,std::vector<DrawCall>& c) {
        // Caller holds recordMutex; targets must have no live elements.
        if(reuseFrameStorage())frameStorage.acquire(v,c);
    }
    void recycleFrameStorage(GlVertices& v,std::vector<DrawCall>& c) {
        // GPU replay has copied borrowed vertices to its own fenced upload.
        // Release frame-owned textures/motion before taking the recording lock.
        v.clear();c.clear();
        std::lock_guard<std::mutex> lock(recordMutex);
        if(reuseFrameStorage())frameStorage.recycle(v,c);
        if(frameStorageStats() && ++storagePresents%240u==0u)
            std::fprintf(stderr,"[gs:frame-storage] enabled=%d presents=%llu acquired=%llu retained=%llu triangle-growths=%llu vertices-moved=%llu recording-capacity=%zu cached-vertex-bytes=%zu\n",
                reuseFrameStorage(),(unsigned long long)storagePresents,
                (unsigned long long)frameStorage.acquires,(unsigned long long)frameStorage.retains,
                (unsigned long long)storageVertexGrowths,(unsigned long long)storageVerticesMoved,
                vertices.capacity(),frameStorage.vertexBytes());
    }

    // Where the frame being presented ends. Set by MarkPresentBoundary while
    // the frontend holds the GS state lock.
    size_t boundaryCalls = 0u;
    size_t boundaryVertices = 0u;
    bool haveBoundary = false;
    uint64_t referencePresentToken = 0u;

    // Per-present replayed call counts. A frame split across two presents shows
    // up as neighbouring presents whose counts differ wildly.
    std::vector<size_t> recentCallCounts;
    void noteReplaySize(size_t count)
    {
        recentCallCounts.push_back(count);
        if (recentCallCounts.size() < 120u)
            return;
        size_t lo = ~size_t(0), hi = 0u, total = 0u, jumps = 0u;
        for (size_t i = 0; i < recentCallCounts.size(); ++i)
        {
            const size_t value = recentCallCounts[i];
            lo = std::min<size_t>(lo, value);
            hi = std::max<size_t>(hi, value);
            total += value;
            if (i > 0)
            {
                const size_t a = recentCallCounts[i - 1], b = value;
                const size_t big = std::max<size_t>(a, b), small = std::min<size_t>(a, b);
                if (big > 20u && small * 10u < big * 6u) // neighbour differs by >40%
                    ++jumps;
            }
        }
        std::fprintf(stderr, "[gs:gl] calls/present over %zu frames: min=%zu max=%zu mean=%zu, %zu big frame-to-frame jumps\n",
                     recentCallCounts.size(), lo, hi, total / recentCallCounts.size(), jumps);
        recentCallCounts.clear();
    }

    // Decoded textures, keyed by the registers that determine their content.
    struct TextureKey
    {
        uint32_t tbp0, cbp;
        uint16_t cov;
        uint8_t tbw, psm, tw, th, cpsm, csm, csa, ta0, ta1, aem, cbw, cou;
        uint8_t mxl;                 // mip levels change what has to be decoded
        uint64_t miptbp1, miptbp2;

        bool operator<(const TextureKey &o) const { return std::memcmp(this, &o, sizeof(TextureKey)) < 0; }
    };
    std::map<TextureKey, std::shared_ptr<TextureEntry>> textures;
    // Entries the game has invalidated. Their GL texture can only be deleted on
    // the GL thread, and only once no in-flight frame still references them.
    std::vector<std::shared_ptr<TextureEntry>> retired;
    std::shared_ptr<TextureEntry> xboxButtons;
    std::shared_ptr<TextureEntry> retailButtons;
    bool triedXboxButtons = false;

    std::shared_ptr<TextureEntry> controllerTexture(const std::shared_ptr<TextureEntry> &original, int port = 0)
    {
        // Select at every draw, including cache hits: hot-plugging must not
        // leave the old family visible until the game reloads its textures.
        if (!original->wotmFaceButtons || ps2xHostPadGlyphFamily(port) != 1)
            return original;
        if (!triedXboxButtons) {
            triedXboxButtons = true;
            auto replacement = std::make_shared<TextureEntry>();
            if (ps2xLoadXboxButtonAtlas(replacement->rgba)) {
                replacement->width = replacement->height = 256u;
                xboxButtons = std::move(replacement);
                std::fprintf(stderr, "[pad:glyphs] loaded DirectXTK Xbox buttons for shared retail atlas\n");
            }
        }
        return xboxButtons ? xboxButtons : original;
    }

    // A port-owned texture used by the original roster meshes. A reserved TEX0
    // signature makes this independent of retail VRAM addresses and page timing.
    std::shared_ptr<TextureEntry> rosterPlaceholder;
    std::shared_ptr<TextureEntry> rosterQuestionTexture() {
        if(rosterPlaceholder)return rosterPlaceholder;
        auto entry=std::make_shared<TextureEntry>();
        entry->width=entry->height=64;
        entry->rgba.resize(64u*64u*4u);
        struct Point {float x,y;};
        const Point curve[][3]={{{20,24},{20,12},{32,12}},{{32,12},{45,12},{45,24}},
            {{45,24},{45,31},{34,36}},{{34,36},{31,38},{31,43}}};
        for(unsigned y=0;y<64;++y)for(unsigned x=0;x<64;++x) {
            float coverage=0;
            for(unsigned sy=0;sy<4;++sy)for(unsigned sx=0;sx<4;++sx) {
                const float px=x+(sx+0.5f)/4.0f,py=y+(sy+0.5f)/4.0f;
                float distance=(px-31)*(px-31)+(py-52)*(py-52);
                for(const auto& q:curve)for(unsigned i=0;i<=32;++i) {
                    const float t=i/32.0f,u=1-t;
                    const float dx=px-(u*u*q[0].x+2*u*t*q[1].x+t*t*q[2].x);
                    const float dy=py-(u*u*q[0].y+2*u*t*q[1].y+t*t*q[2].y);
                    distance=std::min<float>(distance,dx*dx+dy*dy);
                }
                if(distance<=9.0f)coverage+=1.0f/16.0f;
            }
            // Retail character cards map V bottom-to-top.
            auto* pixel=entry->rgba.data()+((63u-y)*64u+x)*4u;
            const uint8_t shade=static_cast<uint8_t>(48.0f+coverage*150.0f);
            pixel[0]=pixel[1]=pixel[2]=shade;pixel[3]=128;
        }
        rosterPlaceholder=entry;
        return entry;
    }

    // Decodes (or finds) the texture TEX0 selects. Game thread, recordMutex held.
    std::shared_ptr<TextureEntry> textureFor(GSCpuBackend &cpu, const GSDrawState &state)
    {
        const GSTex0Reg &tex = state.context.tex0;
        if(tex.tbp0==0x3fffu && tex.cbp==0x3fffu && tex.psm==0x13u &&
           tex.tw==6u && tex.th==6u && tex.csa==29u)return rosterQuestionTexture();
        if (retailButtons && tex.tbp0 == 0x3fffu && tex.cbp == 0x3fffu &&
            tex.psm == 0x13u && tex.tw == 6u && tex.th == 6u && tex.csa >= 30u)
            return controllerTexture(retailButtons, tex.csa - 30u);
        TextureKey key{};
        std::memset(&key, 0, sizeof(key));
        key.tbp0 = tex.tbp0;
        key.cbp = tex.cbp;
        key.cov = state.texclut.cov;
        key.tbw = tex.tbw;
        key.psm = tex.psm;
        key.tw = tex.tw;
        key.th = tex.th;
        key.cpsm = tex.cpsm;
        key.csm = tex.csm;
        key.csa = tex.csa;
        key.ta0 = state.texa.ta0;
        key.ta1 = state.texa.ta1;
        key.aem = state.texa.aem ? 1u : 0u;
        key.cbw = state.texclut.cbw;
        key.cou = state.texclut.cou;
        key.mxl = static_cast<uint8_t>((state.context.tex1 >> 2) & 7u);
        if (key.mxl != 0u)
        {
            key.miptbp1 = state.context.miptbp1;
            key.miptbp2 = state.context.miptbp2;
        }

        auto it = textures.find(key);
        if (it != textures.end())
        {
            // Sample the cached image against settled VRAM without changing
            // the image used by either renderer. Detect missed invalidations.
            static const bool verifyCache = std::getenv("PS2X_GS_TEXTURE_VERIFY") != nullptr;
            static uint64_t cacheHits = 0, checks = 0, mismatches = 0;
            if (verifyCache && (++cacheHits % 257u) == 0u) {
                const auto &cached = *it->second;
                std::vector<uint8_t> fresh(cached.rgba.size());
                cpu.DecodeTextureRgba(state, cached.width, cached.height, fresh.data());
                ++checks;
                if (fresh != cached.rgba) {
                    ++mismatches;
                    if (mismatches <= 12u)
                        std::fprintf(stderr,"[gs:texture-cache] MISMATCH tbp=%x cbp=%x psm=%x size=%ux%u range=%x..%x clut=%x..%x\n",
                            tex.tbp0,tex.cbp,tex.psm,cached.width,cached.height,cached.texBegin,cached.texEnd,cached.clutBegin,cached.clutEnd);
                }
                if (checks == 1u || checks % 128u == 0u)
                    std::fprintf(stderr,"[gs:texture-cache] checks=%llu mismatches=%llu\n",checks,mismatches);
            }
            return controllerTexture(it->second);
        }

        auto entry = std::make_shared<TextureEntry>();
        entry->width = std::max<uint32_t>(state.textureWidth, 1u);
        entry->height = std::max<uint32_t>(state.textureHeight, 1u);
        entry->rgba.resize(static_cast<size_t>(entry->width) * entry->height * 4u);
        cpu.DecodeTextureRgba(state, entry->width, entry->height, entry->rgba.data());
        // Identify content, never a reusable VRAM address. RGB is invariant
        // across TEXA/alpha variants of this palette. All users of the atlas
        // (menu meshes, dialogs and gameplay prompts) pass through here.
        if (entry->width == 64u && entry->height == 64u) {
            uint64_t hash = 14695981039346656037ull;
            for (size_t i = 0; i < entry->rgba.size(); ++i)
                if ((i & 3u) != 3u) hash = (hash ^ entry->rgba[i]) * 1099511628211ull;
            entry->wotmFaceButtons = hash == 0x37511119c07b4515ull || // front-end palette
                                    hash == 0x7715e9a44dbf7091ull;  // in-level palette
            if (entry->wotmFaceButtons && !retailButtons) retailButtons = entry;
        }
        const uint32_t rowPixels = std::max<uint32_t>(static_cast<uint32_t>(tex.tbw) * 64u, entry->width);
        blockRange(tex.tbp0, rowPixels, entry->height, tex.psm, entry->texBegin,
                   entry->texEnd);

        // Mip levels. MIPTBP1/2 pack three {TBP (14 bits), TBW (6 bits)} pairs
        // each; MTBA instead lays the levels out straight after TBP0.
        // PS2X_GS_MIPS=0 keeps level 0 only, for comparing against the mip path.
        static const bool mipsEnabled = [] {
            const char *p = std::getenv("PS2X_GS_MIPS");
            return !(p && p[0] == '0');
        }();
        const uint32_t mxl =
            mipsEnabled ? std::min<uint32_t>(static_cast<uint32_t>((state.context.tex1 >> 2) & 7u), 6u) : 0u;
        if (mxl > 0u)
        {
            const bool autoBase = ((state.context.tex1 >> 9) & 1u) != 0u;
            uint32_t autoTbp = tex.tbp0;
            uint32_t autoTbw = tex.tbw;
            for (uint32_t level = 1; level <= mxl; ++level)
            {
                const uint32_t levelWidth = std::max<uint32_t>(entry->width >> level, 1u);
                const uint32_t levelHeight = std::max<uint32_t>(entry->height >> level, 1u);
                uint32_t tbp = 0u, tbw = 0u;
                if (autoBase)
                {
                    const uint32_t previousWidth = std::max<uint32_t>(entry->width >> (level - 1u), 1u);
                    const uint32_t previousHeight = std::max<uint32_t>(entry->height >> (level - 1u), 1u);
                    const uint32_t bytes = previousWidth * previousHeight * GSInternal::bitsPerPixel(tex.psm) / 8u;
                    autoTbp += std::max<uint32_t>(bytes / 256u, 1u);
                    autoTbw = std::max<uint32_t>(autoTbw >> 1, 1u);
                    tbp = autoTbp;
                    tbw = autoTbw;
                }
                else
                {
                    const uint64_t packed = level <= 3u ? state.context.miptbp1 : state.context.miptbp2;
                    const uint32_t slot = (level - 1u) % 3u;
                    tbp = static_cast<uint32_t>((packed >> (slot * 20u)) & 0x3FFFu);
                    tbw = static_cast<uint32_t>((packed >> (slot * 20u + 14u)) & 0x3Fu);
                }
                if (tbp == 0u)
                    break; // no level here: stop and use what we have
                GSDrawState levelState = state;
                levelState.context.tex0.tbp0 = tbp;
                levelState.context.tex0.tbw = static_cast<uint8_t>(std::max<uint32_t>(tbw, 1u));
                TextureEntry::Level mip{};
                mip.width = levelWidth;
                mip.height = levelHeight;
                mip.rgba.resize(static_cast<size_t>(levelWidth) * levelHeight * 4u);
                cpu.DecodeTextureRgba(levelState, levelWidth, levelHeight, mip.rgba.data());
                uint32_t mipBegin = 0u, mipEnd = 0u;
                blockRange(tbp, std::max<uint32_t>(tbw * 64u, levelWidth), levelHeight,
                           tex.psm, mipBegin, mipEnd);
                entry->texBegin = std::min<uint32_t>(entry->texBegin, mipBegin);
                entry->texEnd = std::max<uint32_t>(entry->texEnd, mipEnd);
                entry->mips.push_back(std::move(mip));
            }
        }
        if (isPaletted(tex.psm))
        {
            const uint32_t clutRows = (tex.cpsm == GS_PSM_CT16 || tex.cpsm == GS_PSM_CT16S) ? 32u : 16u;
            blockRange(tex.cbp,
                       std::max<uint32_t>(std::max<uint32_t>(state.texclut.cbw, 1u) * 64u,
                                          state.texclut.cou + 16u),
                       state.texclut.cov + clutRows, tex.cpsm, entry->clutBegin, entry->clutEnd);
        }
        ++textureDecodes;
        if (!entry->mips.empty())
            ++textureMipmapped;
        // PS2X_GS_TEXDUMP=<dir>: one PNG per decoded texture, to check the
        // swizzle, palette and TEXA handling against the CPU rasterizer.
        static const char *dumpDir = std::getenv("PS2X_GS_TEXDUMP");
        static const uint64_t dumpLimit = [] {
            const char *p = std::getenv("PS2X_GS_TEXDUMP_LIMIT");
            return p ? std::strtoull(p, nullptr, 10) : 200ull;
        }();
        static const bool dumpSmall = std::getenv("PS2X_GS_TEXDUMP_SMALL") != nullptr;
        // Mipmapped textures are dumped whatever their decode number: they only
        // show up in levels, long after the first 200.
        static uint32_t mipDumps = 0u;
        if (dumpDir && (!dumpSmall || (entry->width <= 128u && entry->height <= 128u)) &&
            (textureDecodes <= dumpLimit || (!entry->mips.empty() && mipDumps++ < 40u)))
        {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/tex_%03llu_%04x_psm%02x_%ux%u.png", dumpDir,
                          (unsigned long long)textureDecodes, tex.tbp0, tex.psm, entry->width, entry->height);
            ps2xGsSaveImage(path, entry->rgba.data(), entry->width, entry->height);
            for (size_t level = 0; level < entry->mips.size(); ++level)
            {
                std::snprintf(path, sizeof(path), "%s/tex_%03llu_%04x_psm%02x_mip%zu_%ux%u.png", dumpDir,
                              (unsigned long long)textureDecodes, tex.tbp0, tex.psm, level + 1u,
                              entry->mips[level].width, entry->mips[level].height);
                ps2xGsSaveImage(path, entry->mips[level].rgba.data(), entry->mips[level].width,
                                entry->mips[level].height);
            }
        }

        // The game cycles textures through VRAM; without a cap a long session
        // would keep every address it ever used.
        if (textures.size() >= 2048u)
            dropUnreferenced();
        textures.emplace(key, entry);
        return controllerTexture(entry);
    }

    void dropUnreferenced()
    {
        for (auto it = textures.begin(); it != textures.end();)
        {
            if (it->second.use_count() == 1)
            {
                retired.push_back(std::move(it->second));
                it = textures.erase(it);
            }
            else
                ++it;
        }
    }

    // A transfer wrote [begin,end) blocks: every texture decoded from that range
    // is stale. recordMutex held.
    void invalidate(uint32_t begin, uint32_t end)
    {
        preparedValid = false;
        for (auto it = textures.begin(); it != textures.end();)
        {
            if (it->second->sourcedFrom(begin, end))
            {
                retired.push_back(std::move(it->second));
                it = textures.erase(it);
                ++textureInvalidations;
            }
            else
                ++it;
        }
    }

    uint64_t frames = 0u, primitives = 0u, drawCalls = 0u, readbackBytes = 0u;
    uint64_t textureUploads = 0u, textureDecodes = 0u, textureInvalidations = 0u, textureMipmapped = 0u;

    // Direct presentation: the host renderer reads these from its own thread.
    bool direct = false;
    std::atomic<unsigned int> publishedTexture{0u};
    std::atomic<uint32_t> publishedWidth{0u}, publishedHeight{0u};       // PS2 pixels, for the aspect ratio
    std::atomic<uint32_t> publishedSourceWidth{0u}, publishedSourceHeight{0u}; // target pixels, for the source rect
    std::atomic<uint32_t> publishedTargetWidth{0u}, publishedTargetHeight{0u};
    uint32_t frameFbp = 0u, frameFbw = 10u, framePsm = 0u;
    // Every FRAME buffer the game has drawn into: a texture whose TBP0 is one of
    // these is reading back a render target, not ordinary texture memory.
    // Every FRAME buffer the game has drawn into, with the format it draws in.
    struct KnownTarget
    {
        uint32_t blocks = 0u;    // FBP << 5
        uint8_t psm = 0u;
        uint8_t fbw = 0u;
        uint64_t lastDrawFrame = 0u;
    };
    std::map<uint32_t, KnownTarget> knownTargets;
    uint64_t renderTargetReads = 0u;
    // The VRAM ranges recent transfers wrote, and when. A displayed buffer
    // written by a transfer more recently than anything drew into it is holding
    // an image the GPU never made (the intro movies).
    struct TransferMark
    {
        uint32_t begin = 0u, end = 0u;
        uint64_t frame = 0u;
    };
    std::vector<TransferMark> transfers; // ring of the last kTransferMarks
    size_t transferCursor = 0u;
    static constexpr size_t kTransferMarks = 64u;
    uint64_t presentsFromCpu = 0u;
    uint64_t blits = 0u;

    // The transfer currently being fed by UploadImage, when it is aimed at a
    // frame buffer we render into.
    struct PendingBlit
    {
        bool active = false;
        uint32_t fbp = 0u;
        uint32_t x = 0u, y = 0u, width = 0u, height = 0u;
        uint8_t psm = 0u;
        size_t expectedBytes = 0u;
        std::vector<uint8_t> data;
    };
    PendingBlit pendingBlit;

    void recordBlit(const PendingBlit &blit);

    void noteTransfer(uint32_t begin, uint32_t end)
    {
        if (transfers.size() < kTransferMarks)
            transfers.push_back(TransferMark{begin, end, frames});
        else
        {
            transfers[transferCursor] = TransferMark{begin, end, frames};
            transferCursor = (transferCursor + 1u) % kTransferMarks;
        }
    }

    // The most recent frame in which a transfer wrote anywhere in [begin,end).
    uint64_t lastTransferInto(uint32_t begin, uint32_t end) const
    {
        uint64_t latest = 0u;
        for (const TransferMark &mark : transfers)
            if (mark.begin < end && begin < mark.end)
                latest = std::max<uint64_t>(latest, mark.frame);
        return latest;
    }
    std::vector<uint8_t> readback;

    // Every entry point that touches GL takes this.
    // Reports the first GL error seen at each tagged site.
    static void check(const char *tag)
    {
        const GLenum err = glGetError();
        if (err == GL_NO_ERROR)
            return;
        static std::map<std::string, uint32_t> seen;
        if (seen[tag]++ == 0u)
            std::fprintf(stderr, "[gs:gl] %s -> GL error 0x%x\n", tag, err);
    }

    struct Scope
    {
        explicit Scope(Impl &impl) : owner(impl), lock(impl.glMutex) { impl.context.claim(); }
        ~Scope() { owner.context.release(); }
        Impl &owner;
        std::unique_lock<std::mutex> lock;
    };

    bool ensure()
    {
        if (ready)
            return true;
        if (failed)
            return false;
        if (!context.create() || !gl.load())
        {
            std::fprintf(stderr, "[gs:gl] no OpenGL 4.3 context: staying on the CPU rasterizer\n");
            failed = true;
            return false;
        }
        context.claim();
        const char *renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
        const char *version = reinterpret_cast<const char *>(glGetString(GL_VERSION));
        std::fprintf(stderr, "[gs:gl] %s, OpenGL %s\n", renderer ? renderer : "?", version ? version : "?");
        gl.loadOptional();
        if (gl.ClipControl)
        {
            while (glGetError() != GL_NO_ERROR)
            {
            }
            gl.ClipControl(kLowerLeft, kZeroToOne);
            // The entry point exists on any modern driver, but calling it in a
            // pre-4.5 context just raises an error and changes nothing.
            zeroToOne = glGetError() == GL_NO_ERROR;
        }
        std::fprintf(stderr, "[gs:gl] depth range %s\n", zeroToOne ? "0..1 (exact PS2 Z)" : "-1..1");
        // PS2X_GS_SCALE=<n>: render at n times the PS2's 640x448.
        static const uint32_t requestedScale = [] {
            const char *p = std::getenv("PS2X_GS_SCALE");
            const long v = p ? std::strtol(p, nullptr, 10) : 1;
            return static_cast<uint32_t>(v < 1 ? 1 : (v > 8 ? 8 : v));
        }();
        scale = requestedScale;
        width = kNativeWidth * scale;
        height = kNativeHeight * scale;
        if (scale != 1u)
            std::fprintf(stderr, "[gs:gl] render resolution %ux%u (%ux)\n", width, height, scale);
        const bool built = buildProgram() && buildTargets();
        context.release();
        if (!built)
        {
            failed = true;
            return false;
        }
        // Direct presentation needs a context sharing objects with raylib's.
        // PS2X_GS_GPU_READBACK=1 forces the old read-back-and-upload path.
        static const bool forceReadback = [] {
            const char *p = std::getenv("PS2X_GS_GPU_READBACK");
            return p && p[0] == '1';
        }();
        direct = context.sharesWithHost() && !forceReadback;
        std::fprintf(stderr, "[gs:gl] presentation: %s\n",
                     direct ? "render target straight to the window" : "read back to the host upload path");
        ready = true;
        s_activeImpl = this;
        return true;
    }

    static Impl *s_activeImpl;

    GLuint compile(GLenum type, const char *source, const char *what)
    {
        const GLuint shader = gl.CreateShader(type);
        gl.ShaderSource(shader, 1, &source, nullptr);
        gl.CompileShader(shader);
        GLint ok = 0;
        gl.GetShaderiv(shader, kCompileStatus, &ok);
        if (!ok)
        {
            char log[1024]{};
            gl.GetShaderInfoLog(shader, sizeof(log) - 1, nullptr, log);
            std::fprintf(stderr, "[gs:gl] %s shader failed: %s\n", what, log);
            return 0u;
        }
        return shader;
    }

    bool buildProgram()
    {
        const GLuint vs = compile(kVertexShader, kVertexShaderSource, "vertex");
        const GLuint fs = compile(kFragmentShader, kFragmentShaderSource, "fragment");
        if (!vs || !fs)
            return false;
        program = gl.CreateProgram();
        gl.AttachShader(program, vs);
        gl.AttachShader(program, fs);
        gl.LinkProgram(program);
        GLint ok = 0;
        gl.GetProgramiv(program, kLinkStatus, &ok);
        if (!ok)
        {
            char log[1024]{};
            gl.GetProgramInfoLog(program, sizeof(log) - 1, nullptr, log);
            std::fprintf(stderr, "[gs:gl] link failed: %s\n", log);
            return false;
        }
        gl.DeleteShader(vs);
        gl.DeleteShader(fs);
        check("program");
        targetSizeUniform = gl.GetUniformLocation(program, "uTargetSize");
        debugUniform = gl.GetUniformLocation(program, "uDebug");
        texturedUniform = gl.GetUniformLocation(program, "uTextured");
        tccUniform = gl.GetUniformLocation(program, "uTcc");
        tfxUniform = gl.GetUniformLocation(program, "uTfx");
        iipUniform = gl.GetUniformLocation(program, "uIip");
        alphaTestUniform = gl.GetUniformLocation(program, "uAlphaTest");
        atstUniform = gl.GetUniformLocation(program, "uAtst");
        arefUniform = gl.GetUniformLocation(program, "uAref");
        wrapUniform = gl.GetUniformLocation(program, "uWrap");
        regionUniform = gl.GetUniformLocation(program, "uRegion");
        texSizeUniform = gl.GetUniformLocation(program, "uTexSize");
        sceneRefractionUniform = gl.GetUniformLocation(program, "uSceneRefraction");
        zeroToOneUniform = gl.GetUniformLocation(program, "uZeroToOne");
        fogUniform = gl.GetUniformLocation(program, "uFog");
        fogColorUniform = gl.GetUniformLocation(program, "uFogColor");
        invertAlphaTestUniform = gl.GetUniformLocation(program, "uInvertAlphaTest");
        pabeUniform = gl.GetUniformLocation(program, "uPabe");
        ditherUniform = gl.GetUniformLocation(program, "uDither");
        dimxUniform = gl.GetUniformLocation(program, "uDimx");

        gl.GenVertexArrays(1, &vao);
        gl.BindVertexArray(vao);
        gl.GenBuffers(1, &vbo);
        gl.BindBuffer(kArrayBuffer, vbo);
        gl.VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(GlVertex), reinterpret_cast<void *>(0));
        gl.EnableVertexAttribArray(0);
        gl.VertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(GlVertex), reinterpret_cast<void *>(8));
        gl.EnableVertexAttribArray(1);
        gl.VertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(GlVertex), reinterpret_cast<void *>(12));
        gl.EnableVertexAttribArray(2);
        gl.VertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(GlVertex), reinterpret_cast<void *>(28));
        gl.EnableVertexAttribArray(3);
        gl.VertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(GlVertex), reinterpret_cast<void *>(40));
        gl.EnableVertexAttribArray(4);
        check("vao");
        return true;
    }

    bool buildTargets()
    {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClearDepth(0.0); // PS2 depth: larger is nearer
        return targetFor(0u) != nullptr;
    }

    // GL thread only.
    RenderTarget *targetFor(uint32_t fbp)
    {
        auto it = targets.find(fbp);
        if (it != targets.end())
            return &it->second;
        if (targets.size() >= 16u)
            return nullptr; // a game that scattered buffers would exhaust VRAM
        RenderTarget target{};
        glGenTextures(1, &target.color);
        glBindTexture(GL_TEXTURE_2D, target.color);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(width), static_cast<GLsizei>(height), 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, kClampToEdge);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, kClampToEdge);
        gl.GenFramebuffers(1, &target.fbo);
        gl.BindFramebuffer(kFramebuffer, target.fbo);
        gl.FramebufferTexture2D(kFramebuffer, kColorAttachment0, GL_TEXTURE_2D, target.color, 0);
        // The depth attachment is chosen per draw, from ZBUF.
        if (gl.CheckFramebufferStatus(kFramebuffer) != kFramebufferComplete)
        {
            std::fprintf(stderr, "[gs:gl] incomplete framebuffer for FBP 0x%x\n", fbp);
            return nullptr;
        }
        glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
        glDisable(GL_SCISSOR_TEST);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        check("target");
        return &targets.emplace(fbp, target).first->second;
    }

    // Uploads a decoded texture on first use. GL thread only. Texture unit 0 is
    // current by default and is the only one used, so there is no glActiveTexture.
    GLuint blitTexture = 0u; // shared by every transient (blit) texture

    void bindTexture(TextureEntry &entry, const DrawCall &call)
    {
        if (entry.transient)
        {
            if (blitTexture == 0u)
                glGenTextures(1, &blitTexture);
            glBindTexture(GL_TEXTURE_2D, blitTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(entry.width),
                         static_cast<GLsizei>(entry.height), 0, GL_RGBA, GL_UNSIGNED_BYTE, entry.rgba.data());
            ++textureUploads;
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, kClampToEdge);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, kClampToEdge);
            return;
        }
        if (entry.glId == 0u)
        {
            GLuint id = 0u;
            glGenTextures(1, &id);
            entry.glId = id;
        }
        glBindTexture(GL_TEXTURE_2D, entry.glId);
        if (!entry.uploaded)
        {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(entry.width),
                         static_cast<GLsizei>(entry.height), 0, GL_RGBA, GL_UNSIGNED_BYTE, entry.rgba.data());
            for (size_t level = 0; level < entry.mips.size(); ++level)
            {
                const TextureEntry::Level &mip = entry.mips[level];
                glTexImage2D(GL_TEXTURE_2D, static_cast<GLint>(level + 1u), GL_RGBA8,
                             static_cast<GLsizei>(mip.width), static_cast<GLsizei>(mip.height), 0, GL_RGBA,
                             GL_UNSIGNED_BYTE, mip.rgba.data());
            }
            glTexParameteri(GL_TEXTURE_2D, kTextureMaxLevel, static_cast<GLint>(entry.mips.size()));
            entry.uploaded = true;
            ++textureUploads;
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        static_cast<GLint>(minFilterFor(call.mmin, !entry.mips.empty())));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, call.mmag ? GL_LINEAR : GL_NEAREST);
        // GS LOD = (log2(1/Q) << L) + K; GL derives the same log term itself, so
        // only K (4.8 fixed point) carries over, as a bias.
        glTexParameterf(GL_TEXTURE_2D, kTextureLodBias, call.lodBias);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, call.repeatU ? GL_REPEAT : kClampToEdge);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, call.repeatV ? GL_REPEAT : kClampToEdge);
    }

    // Frees the GL textures of entries no in-flight frame references. GL thread.
    void collectRetired()
    {
        std::vector<std::shared_ptr<TextureEntry>> keep;
        std::vector<std::shared_ptr<TextureEntry>> dead;
        {
            std::lock_guard<std::mutex> lock(recordMutex);
            dead.swap(retired);
        }
        for (auto &entry : dead)
        {
            if (entry.use_count() > 1)
            {
                keep.push_back(std::move(entry));
                continue;
            }
            if (entry->glId != 0u)
            {
                const GLuint id = entry->glId;
                glDeleteTextures(1, &id);
            }
        }
        if (!keep.empty())
        {
            std::lock_guard<std::mutex> lock(recordMutex);
            for (auto &entry : keep)
                retired.push_back(std::move(entry));
        }
    }

    // GS ALPHA: Cv = (A - B) * C / 128 + D, with A, B, D in {Cs, Cd, 0} and C in
    // {As, Ad, FIX}. Expanded, the source and destination each get a coefficient
    // of the form 0, C, -C, 1 or 1-C, which is exactly what a GL blend function
    // plus an add/subtract equation can express.
    void applyBlend(uint64_t alpha, bool abe)
    {
        if (!abe)
        {
            glDisable(GL_BLEND);
            return;
        }
        auto operand = [](uint64_t v) { return static_cast<uint32_t>(v & 3u) == 3u ? 2u : static_cast<uint32_t>(v & 3u); };
        const uint32_t a = operand(alpha), b = operand(alpha >> 2), c = static_cast<uint32_t>((alpha >> 4) & 3u),
                       d = operand(alpha >> 6);
        const uint32_t fix = static_cast<uint32_t>((alpha >> 32) & 0xFFu);

        int cS = 0, oS = 0, cD = 0, oD = 0; // multiples of C and of 1
        auto bump = [&](uint32_t op, int dc, int dOne)
        {
            if (op == 0u) { cS += dc; oS += dOne; }
            else if (op == 1u) { cD += dc; oD += dOne; }
        };
        bump(a, +1, 0);
        bump(b, -1, 0);
        bump(d, 0, +1);

        GLenum cFactor = GL_SRC_ALPHA, oneMinusC = GL_ONE_MINUS_SRC_ALPHA;
        if (c == 1u)
        {
            cFactor = kDstAlpha;
            oneMinusC = kOneMinusDstAlpha;
        }
        else if (c >= 2u)
        {
            // FIX is on the same 128 = 1.0 scale as alpha.
            cFactor = kConstantAlpha;
            oneMinusC = kOneMinusConstantAlpha;
            gl.BlendColor(0.0f, 0.0f, 0.0f, static_cast<float>(fix) / 128.0f);
        }

        // Returns the GL factor and the sign the term carries.
        auto term = [&](int cc, int oc, GLenum &factor)
        {
            if (cc == 0 && oc == 0) { factor = GL_ZERO; return 1; }
            if (cc == 0) { factor = GL_ONE; return oc > 0 ? 1 : -1; }
            if (oc == 0) { factor = cFactor; return cc > 0 ? 1 : -1; }
            if (cc < 0) { factor = oneMinusC; return 1; }     // 1 - C
            factor = GL_ONE;                                  // 1 + C: not expressible, clamps anyway
            return 1;
        };
        GLenum srcFactor = GL_ONE, dstFactor = GL_ZERO;
        const int srcSign = term(cS, oS, srcFactor);
        const int dstSign = term(cD, oD, dstFactor);

        GLenum equation = kFuncAdd;
        if (srcSign < 0 && dstSign > 0)
            equation = kFuncReverseSubtract; // dst * f - src * f
        else if (dstSign < 0 && srcSign > 0)
            equation = kFuncSubtract;
        else if (srcSign < 0 && dstSign < 0)
            srcFactor = dstFactor = GL_ZERO; // -(a+b): nothing sensible, write black

        glEnable(GL_BLEND);
        // Alpha is not blended on the GS: the frame buffer gets As straight.
        gl.BlendEquationSeparate(equation, kFuncAdd);
        gl.BlendFuncSeparate(srcFactor, dstFactor, GL_ONE, GL_ZERO);
    }

    void applyDepth(uint64_t test, uint32_t zmsk)
    {
        // The GS writes Z whenever ZMSK is 0, whatever the test says -- the CPU
        // rasterizer does exactly that and ignores ZTE. OpenGL will not touch
        // the depth buffer at all while GL_DEPTH_TEST is disabled (glDepthMask
        // is ignored), so the test stays enabled and ZTST=ALWAYS becomes
        // GL_ALWAYS. Disabling it here meant the game's full screen Z priming
        // sprites (ZTST=ALWAYS, ZMSK=0) wrote no depth, which left stale depth
        // behind and let geometry show through walls.
        const uint32_t ztst = (test >> 17) & 3u;
        const bool zte = ((test >> 16) & 1u) != 0u;
        glEnable(GL_DEPTH_TEST);
        if (!zte)
            glDepthFunc(GL_ALWAYS);
        else
            glDepthFunc(ztst == 0u   ? GL_NEVER
                        : ztst == 1u ? GL_ALWAYS
                        : ztst == 2u ? GL_GEQUAL
                                     : GL_GREATER);
        glDepthMask(zmsk ? GL_FALSE : GL_TRUE);
    }

    // Replays one frame of recorded draw calls, switching render target
    // whenever the game switched FRAME buffer. GL thread only.
    wotm_capture::Frame* traceCaptureSnapshot = nullptr;
    std::string traceCapturePath;
    void captureCheckpoint(size_t completed) {
        if(!traceCaptureSnapshot)return;
        const char* cursor=std::getenv("PS2X_D3D12_CAPTURE_DRAWS");bool wanted=false;
        if(!cursor)return;
        while(*cursor){char* end=nullptr;auto n=std::strtoull(cursor,&end,10);if(end==cursor)break;
            if(n==completed)wanted=true;cursor=*end==','?end+1:end;}
        if(!wanted)return;
        auto& shot=*traceCaptureSnapshot;auto fullDraws=shot.draws;shot.draws.resize(completed);
        shot.expected.clear();for(const auto& t:targets)shot.expected.push_back(captureImage(t.first,t.second.color,false));
        try{const std::string name=traceCapturePath+".draw"+std::to_string(completed)+".wdr";
            wotm_capture::File file(name.c_str(),false);file.frame(shot);
            std::fprintf(stderr,"[gs:d3d12-checkpoint] %s\n",name.c_str());
        }catch(const std::exception& e){std::fprintf(stderr,"[gs:d3d12-checkpoint] %s\n",e.what());}
        shot.draws=std::move(fullDraws);shot.expected.clear();
    }
    // Opt-in same-frame differential capture; no D3D12 dependency in the game.
    wotm_capture::Image captureImage(uint32_t id, GLuint texture, bool depth) {
        wotm_capture::Image image; image.id=id;image.width=width;image.height=height;
        image.bytes.resize(size_t(width)*height*4);
        glBindTexture(GL_TEXTURE_2D,texture);
        glGetTexImage(GL_TEXTURE_2D,0,depth?GL_DEPTH_COMPONENT:GL_RGBA,depth?GL_FLOAT:GL_UNSIGNED_BYTE,image.bytes.data());
        const size_t row=size_t(width)*4;
        for(uint32_t y=0;y<height/2;++y)
            std::swap_ranges(image.bytes.begin()+y*row,image.bytes.begin()+(y+1)*row,image.bytes.begin()+(height-1-y)*row);
        return image;
    }
    std::map<uint32_t, uint32_t> passHeights; // presentation-thread owned
    void preparePacket(wotm_capture::Frame& snapshot,const GlVertices& verts,const std::vector<DrawCall>& list,bool borrow) {
        for (const auto& h : wotm_capture::resolvePassHeights(list, passHeights))
            snapshot.heights.push_back({h.first, h.second});
        static_assert(sizeof(GlVertex)==sizeof(wotm_capture::Vertex));
        static const bool borrowVertices=[] {
            const char* p=std::getenv("PS2X_D3D12_BORROW_VERTICES");return !p || p[0]!='0';
        }();
        if(borrow && borrowVertices) {
            snapshot.borrowedVertices=verts.data();snapshot.borrowedVertexCount=verts.size();
        } else snapshot.vertices.assign(verts.begin(),verts.end());
        if(MotionProvenance::vectorsEnabled()) {
            snapshot.motion.resize(verts.size());
            snapshot.motionResetSerial=MotionProvenance::resetSerial();
            for(const auto& c:list)snapshot.motionKinds.push_back(c.motion && c.motion->epoch==snapshot.motionResetSerial && c.motion->shadow ? 1 : 0);
            for(const auto& c:list)snapshot.motionGenerations.push_back(c.motion && c.motion->epoch==snapshot.motionResetSerial ? c.motion->frame : 0);
            for(const auto& c:list)if(c.motion && c.motion->usable()) {
                for(size_t i=c.first;i+2<c.first+c.count;i+=3){
                    bool valid=true;
                    for(size_t j=i;j<i+3;++j){const auto& v=verts[j];snapshot.motion[j]=MotionProvenance::previousVertex(*c.motion,{v.x,v.y,v.z,v.q});valid&=snapshot.motion[j][3]==1;}
                    if(!valid)for(size_t j=i;j<i+3;++j)snapshot.motion[j]={};
                }
            }
        }
        std::map<const TextureEntry*,uint32_t> textureIds;
        std::map<uint32_t,uint32_t> formats;
        for(const auto& c:list){
            auto format=formats.emplace(c.fbp,c.framePsm);
            if(!format.second&&format.first->second!=c.framePsm)format.first->second=~0u;

            wotm_capture::Draw d{};
#define COPY_CAPTURE(field) d.field=c.field
            COPY_CAPTURE(first);COPY_CAPTURE(count);COPY_CAPTURE(fbp);COPY_CAPTURE(zbp);COPY_CAPTURE(textureFbp);
            COPY_CAPTURE(alpha);COPY_CAPTURE(test);COPY_CAPTURE(zmsk);COPY_CAPTURE(abe);COPY_CAPTURE(clear);COPY_CAPTURE(fbmsk);
            COPY_CAPTURE(sx);COPY_CAPTURE(sy);COPY_CAPTURE(sw);COPY_CAPTURE(sh);COPY_CAPTURE(tcc);COPY_CAPTURE(tfx);COPY_CAPTURE(iip);
            COPY_CAPTURE(alphaTest);COPY_CAPTURE(atst);COPY_CAPTURE(afail);COPY_CAPTURE(pabe);COPY_CAPTURE(dither);COPY_CAPTURE(fog);
            COPY_CAPTURE(aref);COPY_CAPTURE(lodBias);COPY_CAPTURE(wrapU);COPY_CAPTURE(wrapV);COPY_CAPTURE(repeatU);COPY_CAPTURE(repeatV);COPY_CAPTURE(mmin);COPY_CAPTURE(mmag);
#undef COPY_CAPTURE
            std::copy_n(c.region,4,d.region);std::copy_n(c.fogColor,3,d.fogColor);std::copy_n(c.dimx,16,d.dimx);std::copy_n(c.clearColor,4,d.clearColor);
            if(c.texture){
                auto inserted=textureIds.emplace(c.texture.get(),uint32_t(snapshot.textures.size()));d.texture=inserted.first->second;
                d.transient=c.texture->transient;
                if(inserted.second){
                    wotm_capture::Texture t;
                    if(borrow){
                        if(!c.texture->transient)t.owner=c.texture;
                        t.levels.push_back({0,c.texture->width,c.texture->height,{}});t.levels.back().borrowed=c.texture->rgba.data();
                        for(const auto& m:c.texture->mips){t.levels.push_back({0,m.width,m.height,{}});t.levels.back().borrowed=m.rgba.data();}
                    } else {
                        t.levels.push_back({c.tbp0,c.texture->width,c.texture->height,c.texture->rgba});
                        for(const auto& m:c.texture->mips)t.levels.push_back({0,m.width,m.height,m.rgba});
                    }
                    snapshot.textures.push_back(std::move(t));
                }
            }
            snapshot.draws.push_back(d);
        }
        for(const auto& format:formats)snapshot.formats.push_back({format.first,format.second});
    }
    void captureReplay(const GlVertices& verts,const std::vector<DrawCall>& list,uint32_t display) {
        static const char* path=std::getenv("PS2X_D3D12_CAPTURE");
        static const char* when=std::getenv("PS2X_D3D12_CAPTURE_FRAME");
        static const char* many=std::getenv("PS2X_D3D12_CAPTURE_FRAMES");
        bool selected=frames==(when?std::strtoull(when,nullptr,10):600ull);
        if(many){selected=false;const char* cursor=many;while(*cursor){char* end=nullptr;
            const auto value=std::strtoull(cursor,&end,10);if(end==cursor)break;
            if(frames==value)selected=true;cursor=(*end==',')?end+1:end;}}
        static const bool onRefraction = std::getenv("PS2X_D3D12_CAPTURE_REFRACTION") != nullptr;
        static uint32_t refractionCaptures = 0;
        if (onRefraction) {
            selected = path && refractionCaptures < 3u && std::any_of(list.begin(), list.end(), [](const DrawCall& c) {
                return c.tcc == 0u && c.wrapU == 2 && c.wrapV == 2 &&
                       c.region[1] == 638.f && c.region[3] == 222.f;
            });
            if (selected) ++refractionCaptures;
        }
        if(!path||!selected){replay(verts,list);return;}
        const std::string destination=std::string(path)+((many||onRefraction)?("."+std::to_string(frames)+".wdr"):"");
        wotm_capture::Frame snapshot;snapshot.width=width;snapshot.height=height;snapshot.scale=scale;snapshot.display=display;
        preparePacket(snapshot,verts,list,false);
        for(const auto& c:list){targetFor(c.fbp);depthFor(c.zbp);}
        for(const auto& t:targets)snapshot.colors.push_back(captureImage(t.first,t.second.color,false));
        for(const auto& t:depths)snapshot.depths.push_back(captureImage(t.first,t.second.texture,true));
        traceCaptureSnapshot=&snapshot;traceCapturePath=destination;
        replay(verts,list);
        traceCaptureSnapshot=nullptr;
        for(const auto& t:targets)snapshot.expected.push_back(captureImage(t.first,t.second.color,false));
        try{wotm_capture::File file(destination.c_str(),false);file.frame(snapshot);
            std::fprintf(stderr,"[gs:d3d12-capture] %zu draws, %zu vertices, %zu textures -> %s\n",list.size(),verts.size(),snapshot.textures.size(),destination.c_str());
        }catch(const std::exception& e){std::fprintf(stderr,"[gs:d3d12-capture] %s\n",e.what());}
    }

    void replay(const GlVertices &verts, const std::vector<DrawCall> &list)
    {
        check("before replay");
        // How tall each pass drew. The game renders a field the display doubles,
        // and different buffers use different heights, so each target is scaled
        // on its own to fill the full-height render target.
        const auto heights = wotm_capture::resolvePassHeights(list, passHeights);

        gl.UseProgram(program);
        // PS2X_GS_GPU_DEBUG: "depth" = depth as grey, "wire" = wireframe,
        // "tex" = texel only, "vcol" = vertex colour only.
        static const int debugMode = []
        {
            const char *p = std::getenv("PS2X_GS_GPU_DEBUG");
            if (!p)
                return 0;
            return p[0] == 'd' ? 1 : (p[0] == 'w' ? 2 : (p[0] == 't' ? 3 : (p[0] == 'v' ? 4 : 0)));
        }();
        gl.Uniform1i(debugUniform, debugMode);
        gl.Uniform1i(zeroToOneUniform, zeroToOne ? 1 : 0);
        glPolygonMode(GL_FRONT_AND_BACK, debugMode == 2 ? GL_LINE : GL_FILL);
        gl.BindVertexArray(vao);
        gl.BindBuffer(kArrayBuffer, vbo);
        if (!verts.empty())
            gl.BufferData(kArrayBuffer, static_cast<GLsizeiptr>(verts.size() * sizeof(GlVertex)), verts.data(),
                          kStreamDraw);
        check("bind/use");

        uint32_t currentFbp = kNoFbp;
        uint32_t currentZbp = ~0u;
        float scaleY = 1.0f;
        for (const DrawCall &call : list)
        {
            if (call.fbp != currentFbp || call.zbp != currentZbp)
            {
                RenderTarget *target = targetFor(call.fbp);
                DepthTarget *depth = depthFor(call.zbp);
                if (!target || !depth)
                    continue;
                const bool newColorTarget = call.fbp != currentFbp;
                currentFbp = call.fbp;
                currentZbp = call.zbp;

                gl.BindFramebuffer(kFramebuffer, target->fbo);
                if (target->attachedZbp != call.zbp)
                {
                    gl.FramebufferTexture2D(kFramebuffer, kDepthAttachment, GL_TEXTURE_2D, depth->texture, 0);
                    target->attachedZbp = call.zbp;
                }
                // PS2X_GS_ZCLEAR=1 restores a synthetic depth clear once per
                // Z buffer per present. Off by default: the game primes Z with
                // its own full screen sprites (as it must, since the CPU
                // rasterizer has no such clear and works), and ours fired per
                // present -- which cuts a game frame in half, because the game
                // renders at ~27 fps against a ~54 Hz present rate.
                static const bool syntheticZClear = [] {
                    const char *p = std::getenv("PS2X_GS_ZCLEAR");
                    return p && p[0] == '1';
                }();
                if (syntheticZClear && depth->lastFrame != frames)
                {
                    depth->lastFrame = frames;
                    glDisable(GL_SCISSOR_TEST);
                    glDepthMask(GL_TRUE);
                    glClear(GL_DEPTH_BUFFER_BIT);
                }
                if (newColorTarget)
                {
                    glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
                    // uTargetSize is in PS2 pixels: the shader maps the game's
                    // own coordinates onto the render target's real size.
                    // The scissor height this buffer is being drawn with NOW.
                    // Not the tallest ever seen: this game renders 224-line
                    // fields, and one earlier 448-line pass would then make
                    // every field draw at half scale for the rest of the run --
                    // and with the two display buffers remembering different
                    // maxima, the picture alternated between two scales.
                    // A present with no draws for this buffer keeps its value.
                    auto measured = heights.find(call.fbp);
                    const uint32_t passHeight = measured != heights.end() && measured->second != 0u
                                                    ? measured->second
                                                    : std::max<uint32_t>(target->passHeight, 1u);
                    target->passHeight = passHeight;
                    scaleY = static_cast<float>(height) / static_cast<float>(passHeight);
                    gl.Uniform2f(targetSizeUniform, static_cast<float>(kNativeWidth),
                                 static_cast<float>(passHeight));
                    if (target->lastFrame != frames)
                    {
                        // Colour is NOT cleared per frame: the game draws static
                        // backdrops once and composites over them (clearing
                        // loses the menu's sky). PS2X_GS_CLEAR=1 clears anyway,
                        // to tell backdrop loss from ghosting.
                        static const bool clearColour = [] {
                            const char *p = std::getenv("PS2X_GS_CLEAR");
                            return p && p[0] == '1';
                        }();
                        target->lastFrame = frames;
                        if (clearColour)
                        {
                            glDisable(GL_SCISSOR_TEST);
                            glClear(GL_COLOR_BUFFER_BIT);
                        }
                    }
                }
                glEnable(GL_SCISSOR_TEST);
            }
            if (currentFbp == kNoFbp)
                continue;
            const int sx = static_cast<int>(call.sx * static_cast<int>(scale));
            const int sw = static_cast<int>(call.sw * static_cast<int>(scale));
            const int sy = static_cast<int>(call.sy * scaleY);
            const int sh = static_cast<int>(call.sh * scaleY + 0.5f);
            glScissor(sx, static_cast<GLint>(height) - sy - sh, sw, sh);
            if (call.clear)
            {
                glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                glClearColor(call.clearColor[0], call.clearColor[1], call.clearColor[2], call.clearColor[3]);
                glClear(GL_COLOR_BUFFER_BIT); // scissored, like the GS clear
                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                captureCheckpoint(size_t(&call-list.data())+1);
                continue;
            }
            applyBlend(call.alpha, call.abe);
            applyDepth(call.test, call.zmsk);
            // FBMSK protects the bits that are set. Whole channels map to a
            // colour mask; partial masks are not expressible and are ignored.
            glColorMask((call.fbmsk & 0x000000FFu) != 0x000000FFu, (call.fbmsk & 0x0000FF00u) != 0x0000FF00u,
                        (call.fbmsk & 0x00FF0000u) != 0x00FF0000u, (call.fbmsk & 0xFF000000u) != 0xFF000000u);
            gl.Uniform1i(pabeUniform, call.pabe ? 1 : 0);
            gl.Uniform1i(ditherUniform, call.dither ? 1 : 0);
            if (call.dither)
                gl.Uniform4fv(dimxUniform, 4, call.dimx);
            gl.Uniform1i(invertAlphaTestUniform, 0);
            const bool textured = call.texture || call.textureFbp != kNoFbp;
            gl.Uniform1i(texturedUniform, textured ? 1 : 0);
            gl.Uniform1i(tccUniform, static_cast<GLint>(call.tcc));
            gl.Uniform1i(tfxUniform, static_cast<GLint>(call.tfx));
            gl.Uniform1i(iipUniform, static_cast<GLint>(call.iip));
            gl.Uniform1i(alphaTestUniform, static_cast<GLint>(call.alphaTest));
            gl.Uniform1i(atstUniform, static_cast<GLint>(call.atst));
            gl.Uniform1f(arefUniform, call.aref);
            gl.Uniform1i(fogUniform, call.fog ? 1 : 0);
            gl.Uniform3f(fogColorUniform, call.fogColor[0], call.fogColor[1], call.fogColor[2]);
            gl.Uniform2i(wrapUniform, call.wrapU, call.wrapV);
            gl.Uniform4f(regionUniform, call.region[0], call.region[1], call.region[2], call.region[3]);
            const bool sceneRefraction = call.textureFbp != kNoFbp && call.tcc == 0u &&
                call.tfx == 0u && call.wrapU == 2 && call.wrapV == 2 &&
                call.region[0] == 1.f && call.region[1] == 638.f &&
                call.region[2] == 1.f && call.region[3] == 222.f;
            gl.Uniform1i(sceneRefractionUniform, sceneRefraction ? 1 : 0);
            if (call.textureFbp != kNoFbp)
            {
                // Reading a render target (the drive-in movie screen): sample
                // the GPU's copy, since nothing writes the GS memory any more.
                auto it = targets.find(call.textureFbp);
                if (it == targets.end())
                {
                    gl.Uniform1i(texturedUniform, 0);
                }
                else
                {
                    gl.Uniform2f(texSizeUniform, sceneRefraction ? 1024.f : static_cast<float>(width),
                                 sceneRefraction ? 256.f : static_cast<float>(height));
                    if (sceneRefraction && call.textureFbp == call.fbp) {
                        if (!sceneSnapshot) {
                            glGenTextures(1, &sceneSnapshot);
                            glBindTexture(GL_TEXTURE_2D, sceneSnapshot);
                            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                                         GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, kClampToEdge);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, kClampToEdge);
                        } else glBindTexture(GL_TEXTURE_2D, sceneSnapshot);
                        // Snapshot after preceding draws, before sampling this target.
                        // Sampling an attached texture directly is undefined in GL.
                        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height);
                    } else glBindTexture(GL_TEXTURE_2D, it->second.color);
                }
            }
            else if (call.texture)
            {
                gl.Uniform2f(texSizeUniform, static_cast<float>(call.texture->width),
                             static_cast<float>(call.texture->height));
                bindTexture(*call.texture, call);
            }
            glDrawArrays(GL_TRIANGLES, static_cast<GLint>(call.first), static_cast<GLsizei>(call.count));
            ++drawCalls;

            if (call.afail != 0u)
            {
                // AFAIL: fragments that failed the alpha test are still written,
                // to some of the buffers. Draw them again with the test inverted
                // and only those buffers writable.
                gl.Uniform1i(invertAlphaTestUniform, 1);
                const bool writesColor = call.afail != 2u;          // ZB_ONLY writes no colour
                const bool writesAlpha = call.afail == 1u;          // RGB_ONLY leaves alpha alone
                const bool writesDepth = call.afail == 2u;          // only ZB_ONLY touches Z
                glColorMask(writesColor && (call.fbmsk & 0x000000FFu) != 0x000000FFu,
                            writesColor && (call.fbmsk & 0x0000FF00u) != 0x0000FF00u,
                            writesColor && (call.fbmsk & 0x00FF0000u) != 0x00FF0000u,
                            writesColor && writesAlpha && (call.fbmsk & 0xFF000000u) != 0xFF000000u);
                glDepthMask(writesDepth && !call.zmsk ? GL_TRUE : GL_FALSE);
                glDrawArrays(GL_TRIANGLES, static_cast<GLint>(call.first), static_cast<GLsizei>(call.count));
                ++drawCalls;
                gl.Uniform1i(invertAlphaTestUniform, 0);
            }
            captureCheckpoint(size_t(&call-list.data())+1);
        }
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        check("draw");
    }
};

GSGlBackend::Impl *GSGlBackend::Impl::s_activeImpl = nullptr;

std::atomic<uint64_t> g_ps2xGsPresentedGameFrame{0};
static bool completedHandoffEnabled() {
    static const bool enabled=[] {const char* p=std::getenv("PS2X_FRAME_INTERPOLATION");return p && p[0]=='1';}();
    return enabled && ps2xD3D12Enabled();
}
bool ps2xGsHasCompletedFrame() {
    auto* impl=GSGlBackend::Impl::s_activeImpl;
    if(!impl || !completedHandoffEnabled())return false;
    std::lock_guard<std::mutex> lock(impl->recordMutex);
    return !impl->completedDraws.empty();
}
void ps2xGsSealCompletedFrame(uint64_t frame,const GSPresentationRequest& display) {
    auto* impl=GSGlBackend::Impl::s_activeImpl;
    if(!impl || !completedHandoffEnabled())return;
    std::lock_guard<std::mutex> lock(impl->recordMutex);
    if(impl->calls.empty())return;
    if(impl->completedDraws.size()<4) {
        impl->completedDraws.emplace_back();auto& out=impl->completedDraws.back();
        out.vertices.swap(impl->vertices);out.calls.swap(impl->calls);out.display=display;out.frame=frame;
        impl->acquireFrameStorage(impl->vertices,impl->calls);
    } else {
        // Preserve every draw when the host is behind. Coalesce rather than drop
        // framebuffer feedback/effects; skipped history is rejected by interpolation.
        auto& out=impl->completedDraws.back();const size_t offset=out.vertices.size();
        out.vertices.insert(out.vertices.end(),impl->vertices.begin(),impl->vertices.end());
        for(auto& call:impl->calls){call.first+=offset;out.calls.push_back(std::move(call));}
        impl->vertices.clear();impl->calls.clear();out.display=display;out.frame=frame;
    }
    impl->preparedValid=false;impl->haveBoundary=false;
}

// The host renderer asks, on its own thread, whether there is a GPU frame to
// draw directly. Returns false whenever the read-back path is in use.
bool ps2xGsGpuPresentTexture(Ps2xGpuFrame *out)
{
    GSGlBackend::Impl *impl = GSGlBackend::Impl::s_activeImpl;
    if (!impl || !impl->direct || !out)
        return false;
    const unsigned int id = impl->publishedTexture.load(std::memory_order_acquire);
    if (id == 0u)
        return false;
    out->texture = id;
    out->displayWidth = impl->publishedWidth.load(std::memory_order_relaxed);
    out->displayHeight = impl->publishedHeight.load(std::memory_order_relaxed);
    out->sourceWidth = impl->publishedSourceWidth.load(std::memory_order_relaxed);
    out->sourceHeight = impl->publishedSourceHeight.load(std::memory_order_relaxed);
    out->textureWidth = impl->publishedTargetWidth.load(std::memory_order_relaxed);
    out->textureHeight = impl->publishedTargetHeight.load(std::memory_order_relaxed);
    return out->displayWidth != 0u && out->displayHeight != 0u;
}

GSGlBackend::GSGlBackend() : m_impl(std::make_unique<Impl>()) {Impl::s_activeImpl=m_impl.get();}
GSGlBackend::~GSGlBackend()
{
    if (Impl::s_activeImpl == m_impl.get())
        Impl::s_activeImpl = nullptr;
}

bool GSGlBackend::Available() const { return !m_impl->failed; }

void GSGlBackend::Initialize(uint8_t *vram, uint32_t vramSize) { m_cpu.Initialize(vram, vramSize); }

uint64_t g_ps2xGeometryDigest = 1469598103934665603ull;

// Replays an existing GS capture through CPU-side GL submission only. No GL
// context or presentation: this isolates material preparation and vertex packing.
int ps2xBenchmarkGlSubmission(std::vector<uint8_t> &vram, const std::vector<GSPrimitiveBatch> &draws)
{
    if (std::getenv("PS2X_GS_STRIP_PACKET_BENCH")) {
        extern int ps2xBenchmarkNativeStripPackets();
        return ps2xBenchmarkNativeStripPackets();
    }
    GSGlBackend backend;
    backend.Initialize(vram.data(), static_cast<uint32_t>(vram.size()));
    const bool bulk = std::getenv("PS2X_GS_SUBMIT_BULK") != nullptr;
    const bool indexedBench=std::getenv("PS2X_GS_SUBMIT_INDEXED_BENCH")!=nullptr;
    const bool indexedOn=[] {const char *p=std::getenv("PS2X_NATIVE_INDEXED");return !p || p[0]!='0';}();
    struct ReplayStream {
        GSPrimitiveBatch prototype;
        std::vector<GSVertex> expanded,unique;
        std::vector<uint16_t> indices;
    };
    std::vector<ReplayStream> streams;
    if (indexedBench) {
        size_t indexedStreams=0,totalExpanded=0,totalUnique=0;
        for (size_t i=0;i<draws.size();) {
            ReplayStream stream;stream.prototype=draws[i];
            if (draws[i].vertexCount!=3u) {++i;streams.push_back(std::move(stream));continue;}
            do {
                stream.expanded.insert(stream.expanded.end(),draws[i].vertices.begin(),draws[i].vertices.end());
                ++i;
            } while (i<draws.size() && stream.expanded.size()<768u && draws[i].vertexCount==3u &&
                std::memcmp(&stream.prototype.state,&draws[i].state,sizeof(GSDrawState))==0);
            constexpr size_t bytes=offsetof(GSVertex,fog)+sizeof(uint8_t);
            for (const auto &v:stream.expanded) {
                size_t j=0;
                while (j<stream.unique.size() && std::memcmp(&v,&stream.unique[j],bytes)) ++j;
                if (j==stream.unique.size()) stream.unique.push_back(v);
                stream.indices.push_back(uint16_t(j));
            }
            if (stream.unique.size()>256u) {stream.indices.clear();stream.unique.clear();}
            else {++indexedStreams;totalExpanded+=stream.expanded.size();totalUnique+=stream.unique.size();}
            streams.push_back(std::move(stream));
        }
        std::fprintf(stderr,"[gs:indexed-bench] streams=%zu indexed=%zu expanded=%zu unique=%zu enabled=%u\n",
            streams.size(),indexedStreams,totalExpanded,totalUnique,indexedOn);
    }
    std::vector<GSVertex> vertices;
    vertices.reserve(768);
    auto replay = [&] {
        backend.Reset();
        g_ps2xGeometryDigest = 1469598103934665603ull;
        if (indexedBench) {
            for (const auto &stream:streams) {
                if (stream.expanded.empty()) backend.Submit(stream.prototype);
                else if (indexedOn && !stream.indices.empty())
                    ps2xGlSubmitIndexedGeometry(backend,stream.prototype,stream.unique.data(),stream.unique.size(),
                                              stream.indices.data(),stream.indices.size());
                else ps2xGlSubmitGeometry(backend,stream.prototype,stream.expanded.data(),stream.expanded.size());
            }
            return;
        }
        for (size_t i = 0; i < draws.size();) {
            if (!bulk || draws[i].vertexCount != 3u) { backend.Submit(draws[i++]); continue; }
            const auto &first = draws[i];
            vertices.clear();
            do {
                vertices.insert(vertices.end(), draws[i].vertices.begin(), draws[i].vertices.end());
                ++i;
            } while (i < draws.size() && vertices.size() < 768u && draws[i].vertexCount == 3u &&
                     std::memcmp(&first.state, &draws[i].state, sizeof(GSDrawState)) == 0);
            ps2xGlSubmitGeometry(backend, first, vertices.data(), vertices.size());
        }
    };
    for (unsigned i = 0; i < 8; ++i) replay();
    std::vector<double> samples;
    for (unsigned sample = 0; sample < 21; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (unsigned repeat = 0; repeat < 64; ++repeat) {
            replay();
        }
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        samples.push_back(double(ns) / (64.0 * draws.size()));
    }
    std::sort(samples.begin(), samples.end());
    std::fprintf(stderr, "[gs:submit-bench] draws=%zu median_ns_per_draw=%.2f min=%.2f max=%.2f (submission only)\n",
                 draws.size(), samples[10], samples.front(), samples.back());
    if (std::getenv("PS2X_GS_GEOMETRY_DIGEST"))
        std::fprintf(stderr, "[gs:geometry-digest] %016llx\n", (unsigned long long)g_ps2xGeometryDigest);
    return 0;
}

void GSGlBackend::Reset()
{
    m_cpu.Reset();
    std::lock_guard<std::mutex> lock(m_impl->recordMutex);
    m_impl->vertices.clear();
    m_impl->calls.clear();
    m_impl->completedDraws.clear();
    m_impl->preparedValid = false;
}

namespace
{
    const bool s_featureCensusEnabled=[] {
        const char* p=std::getenv("PS2X_GS_FEATURES");return p && p[0]=='1';
    }();
    // PS2X_GS_FEATURES=1: a running tally of the GS features the game asks for,
    // printed every 10 s. Counts primitives, not pixels.
    void featureCensus(const GSPrimitiveBatch &batch)
    {
        const GSDrawState &s = batch.state;
        const GSContext &c = s.context;

        struct Counts
        {
            uint64_t total = 0u;
            uint64_t date = 0u, datm = 0u, pabe = 0u, noColClamp = 0u, dither = 0u, fba = 0u, aa1 = 0u;
            uint64_t afail[4]{};
            uint64_t alphaTest = 0u, mipmapped = 0u, mipTrilinear = 0u;
            uint64_t framePsm[64]{}, zPsm[64]{};
            uint64_t fbmsk = 0u, scanmsk = 0u, scanmskValue[4]{};
            std::map<uint32_t, uint64_t> blends; // A | B<<2 | C<<4 | D<<6 | FIX<<8
            std::map<uint32_t, uint64_t> framePairs; // FBP<<16 | ZBP
            std::map<uint64_t, uint64_t> mipModes; // LCM | MXL<<1 | MMAG<<4 | MMIN<<5 | L<<8 | K<<16
        };
        static std::mutex mutex;
        static Counts n;
        static auto t0 = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex);

        ++n.total;
        const uint64_t test = c.test;
        if ((test >> 14) & 1u)
        {
            ++n.date;
            if ((test >> 15) & 1u)
                ++n.datm;
        }
        if (test & 1u)
        {
            ++n.alphaTest;
            ++n.afail[(test >> 12) & 3u];
        }
        if (s.pabe)
            ++n.pabe;
        if ((s.colclamp & 1ull) == 0ull)
            ++n.noColClamp;
        if (s.dthe & 1ull)
            ++n.dither;
        if (c.fba & 1ull)
            ++n.fba;
        if (s.prim.aa1)
            ++n.aa1;
        if (c.frame.fbmsk != 0u)
            ++n.fbmsk;
        if (s.scanmsk & 3ull)
            ++n.scanmsk;
        ++n.scanmskValue[s.scanmsk & 3ull];
        if (s.prim.tme)
        {
            const uint32_t mxl = static_cast<uint32_t>((c.tex1 >> 2) & 7u);
            const uint32_t mmin = static_cast<uint32_t>((c.tex1 >> 6) & 7u);
            if (mxl > 0u)
            {
                ++n.mipmapped;
                const uint64_t lcm = c.tex1 & 1u;
                const uint64_t mmag = (c.tex1 >> 5) & 1u;
                const uint64_t l = (c.tex1 >> 19) & 3u;
                const uint64_t k = (c.tex1 >> 32) & 0xFFFu;
                n.mipModes[lcm | (uint64_t(mxl) << 1) | (mmag << 4) | (uint64_t(mmin) << 5) | (l << 8) | (k << 16)] += 1u;
            }
            if (mmin >= 4u)
                ++n.mipTrilinear;
        }
        // Which Z buffer each FRAME buffer is paired with. Giving every FRAME
        // its own depth texture is only correct if that pairing is one to one.
        n.framePairs[(static_cast<uint32_t>(c.frame.fbp) << 16) | (c.zbuf.zbp & 0xFFFFu)] += 1u;
        ++n.framePsm[c.frame.psm & 63u];
        ++n.zPsm[c.zbuf.psm & 63u];
        if (s.prim.abe)
        {
            const uint32_t a = static_cast<uint32_t>(c.alpha & 3u);
            const uint32_t b = static_cast<uint32_t>((c.alpha >> 2) & 3u);
            const uint32_t cc = static_cast<uint32_t>((c.alpha >> 4) & 3u);
            const uint32_t d = static_cast<uint32_t>((c.alpha >> 6) & 3u);
            const uint32_t fix = static_cast<uint32_t>((c.alpha >> 32) & 0xFFu);
            n.blends[a | (b << 2) | (cc << 4) | (d << 6) | (fix << 8)] += 1u;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now - t0 < std::chrono::seconds(10))
            return;
        t0 = now;
        auto pct = [&](uint64_t v) { return 100.0 * static_cast<double>(v) / static_cast<double>(n.total ? n.total : 1u); };
        std::fprintf(stderr,
                     "[gs:feat] %llu prims | DATE %.2f%% (DATM %.2f%%) PABE %.2f%% COLCLAMP=0 %.2f%% DTHE %.2f%% "
                     "FBA %.2f%% AA1 %.2f%% FBMSK %.2f%% SCANMSK %.2f%%\n",
                     (unsigned long long)n.total, pct(n.date), pct(n.datm), pct(n.pabe), pct(n.noColClamp),
                     pct(n.dither), pct(n.fba), pct(n.aa1), pct(n.fbmsk), pct(n.scanmsk));
        std::fprintf(stderr,
                     "[gs:feat] alpha test %.2f%% | AFAIL keep %.2f%% fb-only %.2f%% zb-only %.2f%% rgb-only %.2f%% | "
                     "mipmapped %.2f%% (trilinear %.2f%%)\n",
                     pct(n.alphaTest), pct(n.afail[0]), pct(n.afail[1]), pct(n.afail[2]), pct(n.afail[3]),
                     pct(n.mipmapped), pct(n.mipTrilinear));
        std::fprintf(stderr, "[gs:feat] SCANMSK values: 0(off) %.2f%% 1(reserved) %.2f%% 2(skip even) %.2f%% 3(skip odd) %.2f%%\n",
                     pct(n.scanmskValue[0]), pct(n.scanmskValue[1]), pct(n.scanmskValue[2]), pct(n.scanmskValue[3]));
        for (uint32_t i = 0; i < 64u; ++i)
            if (n.framePsm[i])
                std::fprintf(stderr, "[gs:feat] FRAME psm %02x: %.2f%%\n", i, pct(n.framePsm[i]));
        for (const auto &pair : n.framePairs)
            std::fprintf(stderr, "[gs:feat] FRAME fbp 0x%x <- ZBUF zbp 0x%x: %.2f%%\n", pair.first >> 16,
                         pair.first & 0xFFFFu, pct(pair.second));
        for (uint32_t i = 0; i < 64u; ++i)
            if (n.zPsm[i])
                std::fprintf(stderr, "[gs:feat] ZBUF psm %02x: %.2f%%\n", i, pct(n.zPsm[i]));
        {
            std::vector<std::pair<uint64_t, uint64_t>> mips(n.mipModes.begin(), n.mipModes.end());
            std::sort(mips.begin(), mips.end(), [](const auto &l, const auto &r) { return l.second > r.second; });
            for (size_t i = 0; i < mips.size() && i < 8u; ++i)
            {
                const uint64_t k = mips[i].first;
                const int32_t rawK = static_cast<int32_t>((k >> 16) & 0xFFFu);
                std::fprintf(stderr, "[gs:feat] mip LCM=%llu MXL=%llu MMAG=%llu MMIN=%llu L=%llu K=%.3f: %.2f%%\n",
                             k & 1u, (k >> 1) & 7u, (k >> 4) & 1u, (k >> 5) & 7u, (k >> 8) & 3u,
                             (rawK >= 2048 ? rawK - 4096 : rawK) / 256.0, pct(mips[i].second));
            }
        }
        std::vector<std::pair<uint32_t, uint64_t>> rows(n.blends.begin(), n.blends.end());
        std::sort(rows.begin(), rows.end(), [](const auto &l, const auto &r) { return l.second > r.second; });
        static const char *kOperand[4] = {"Cs", "Cd", "0", "?"};
        static const char *kFactor[4] = {"As", "Ad", "FIX", "?"};
        for (size_t i = 0; i < rows.size() && i < 12u; ++i)
        {
            const uint32_t k = rows[i].first;
            std::fprintf(stderr, "[gs:feat] blend (%s-%s)*%s+%s fix=%u: %.2f%%\n", kOperand[k & 3u],
                         kOperand[(k >> 2) & 3u], kFactor[(k >> 4) & 3u], kOperand[(k >> 6) & 3u], (k >> 8) & 0xFFu,
                         pct(rows[i].second));
        }
    }
}

void GSGlBackend::Submit(const GSPrimitiveBatch &batch)
{
    Impl &impl = *m_impl;
    const NativeGlGeometry *stream = s_nativeGlGeometry && s_nativeGlGeometry->backend == this
        ? s_nativeGlGeometry : nullptr;
    NativeGlGeometry decodedStream{};
    auto materializePacked=[&] {
        if(!stream || !stream->packed)return;
        static thread_local std::array<GSVertex,256> decoded;
        for(size_t i=0;i<stream->uniqueCount;++i)
            decoded[i]=wotm_vertex::decodePackedGSVertex<GSVertex>(stream->packed+i*48u,stream->u,stream->v);
        decodedStream=*stream;decodedStream.vertices=decoded.data();decodedStream.packed=nullptr;
        stream=&decodedStream;
    };
    if(stream && stream->packed) {
        // Preserve all existing CPU, tracing and conversion diagnostics. Their
        // independent GSVertex reference is materialized only when requested.
        static const bool decodedRequired=[] {
            const char *linear=std::getenv("PS2X_GL_LINEAR_INDEXED");
            const char *world=std::getenv("PS2X_GL_WORLD_VERTICES");
            return (linear && linear[0]=='0') || (world && world[0]=='0') ||
                std::getenv("PS2X_GL_INDEXED_VERIFY") || std::getenv("PS2X_GL_WORLD_VERIFY") ||
                std::getenv("PS2X_GL_WORLD_BENCH");
        }();
        if(gpuReferenceEnabled() || impl.failed || Impl::tracing() || decodedRequired) {
            materializePacked();
        }
    }
    if (s_nearCullReferenceOnly) {
        if (gpuReferenceEnabled()) m_cpu.Submit(batch);
        return;
    }
    if (gpuReferenceEnabled() || impl.failed) {
        materializePacked();
        if (stream) {
            GSPrimitiveBatch reference = batch;
            for (size_t i = 0; i < stream->count; i += 3) {
                if (stream->indices)
                    for (size_t j = 0; j < 3; ++j) reference.vertices[j] = stream->vertices[stream->indices[i+j]];
                else std::copy_n(stream->vertices + i, 3, reference.vertices.begin());
                m_cpu.Submit(reference);
            }
        } else m_cpu.Submit(batch);
    }
    if (impl.failed) return;
    // Preserve per-primitive diagnostic counts, even when submission is batched.
    if(s_featureCensusEnabled)
        for (size_t i = 0, n = stream ? stream->count / 3 : 1; i < n; ++i) featureCensus(batch);
    const uint8_t count = batch.vertexCount;
    if (count < 2u)
        return; // points are not drawn yet

    const GSDrawState &state = batch.state;
    const GSContext &ctx = state.context;
    static const bool preparedEnabled = [] {
        const char *p = std::getenv("PS2X_GS_PREPARED"); return !p || p[0] != '0';
    }();
    static const bool preparedVerify = std::getenv("PS2X_GS_PREPARED_VERIFY") != nullptr;
    std::lock_guard<std::mutex> lock(impl.recordMutex);
    // Include both ports: shared menu glyphs use port 0, the cheats footer can
    // belong to port 1. Family changes must take effect without a VRAM upload.
    const int glyphFamily = ps2xHostPadGlyphFamily(0) | (ps2xHostPadGlyphFamily(1) << 2);
    const bool reuse = preparedEnabled && impl.preparedValid && impl.preparedFrame == impl.frames &&
        !impl.calls.empty() && impl.preparedCallCount == impl.calls.size() && !impl.calls.back().clear &&
        impl.preparedGlyphFamily == glyphFamily &&
        impl.preparedCall.motion == MotionProvenance::activeMesh() &&
        std::memcmp(&impl.preparedState, &state, sizeof(state)) == 0;
    if (!reuse || preparedVerify)
    {
        Impl::DrawCall call{};
        call.motion = MotionProvenance::activeMesh();
        call.alpha = ctx.alpha;
        call.test = ctx.test;
        call.zmsk = ctx.zbuf.zmask;
        call.abe = state.prim.abe;
        call.sx = static_cast<int>(ctx.scissor.x0);
        call.sy = static_cast<int>(ctx.scissor.y0);
        call.sw = static_cast<int>(ctx.scissor.x1) - call.sx + 1;
        call.sh = static_cast<int>(ctx.scissor.y1) - call.sy + 1;
        call.linear = state.linearFilter;
        call.mmin = static_cast<uint32_t>((ctx.tex1 >> 6) & 7u);
        call.mmag = ((ctx.tex1 >> 5) & 1u) != 0u;
        {
            const int32_t rawK = static_cast<int32_t>((ctx.tex1 >> 32) & 0xFFFu);
            call.lodBias = static_cast<float>(rawK >= 2048 ? rawK - 4096 : rawK) / 256.0f;
        }
        call.wrapU = static_cast<int>(ctx.clamp & 0x3u);
        call.wrapV = static_cast<int>((ctx.clamp >> 2) & 0x3u);
        call.repeatU = call.wrapU == 0;
        call.repeatV = call.wrapV == 0;
        call.region[0] = static_cast<float>((ctx.clamp >> 4) & 0x3FFu);  // MINU
        call.region[1] = static_cast<float>((ctx.clamp >> 14) & 0x3FFu); // MAXU
        call.region[2] = static_cast<float>((ctx.clamp >> 24) & 0x3FFu); // MINV
        call.region[3] = static_cast<float>((ctx.clamp >> 34) & 0x3FFu); // MAXV
        call.fog = state.prim.fge;
        call.fogColor[0] = static_cast<float>(state.fogR) / 255.0f;
        call.fogColor[1] = static_cast<float>(state.fogG) / 255.0f;
        call.fogColor[2] = static_cast<float>(state.fogB) / 255.0f;
        call.tcc = ctx.tex0.tcc;
        call.tfx = ctx.tex0.tfx;
        call.iip = state.prim.iip ? 1u : 0u;
        call.tbp0 = ctx.tex0.tbp0;
        call.cbp = ctx.tex0.cbp;
        call.texPsm = ctx.tex0.psm;
        // TEST: ATE bit 0, ATST bits 1..3, AREF bits 4..11. AFAIL (bits 12..13) is
        // not handled: the shader discards, which is AFAIL = KEEP.
        call.alphaTest = static_cast<uint32_t>(ctx.test & 1u);
        call.atst = static_cast<uint32_t>((ctx.test >> 1) & 7u);
        call.aref = static_cast<float>((ctx.test >> 4) & 0xFFu); // raw 0..255, like the GS
        call.afail = static_cast<uint32_t>((ctx.test >> 12) & 3u);
        if (call.atst == 1u)
            call.alphaTest = 0u; // ALWAYS
        if (call.alphaTest == 0u)
            call.afail = 0u;
        call.pabe = state.pabe;
        call.dither = (state.dthe & 1ull) != 0ull;
        if (call.dither)
            for (uint32_t i = 0; i < 16u; ++i)
            {
                // DIMX packs sixteen 3-bit signed offsets, four bits apart.
                const int32_t raw = static_cast<int32_t>((state.dimx >> (i * 4u)) & 0x7u);
                call.dimx[i] = static_cast<float>(raw >= 4 ? raw - 8 : raw);
            }
        call.fbmsk = ctx.frame.fbmsk;
    
        call.fbp = ctx.frame.fbp;
        call.framePsm = ctx.frame.psm;
        call.zbp = ctx.zbuf.zbp;
    
        {
            Impl::KnownTarget &known = impl.knownTargets[ctx.frame.fbp];
            known.blocks = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
            known.psm = ctx.frame.psm;
            known.fbw = ctx.frame.fbw;
            known.lastDrawFrame = impl.frames;
        }
        {
            // Same bookkeeping the CPU rasterizer does per pixel, once per draw.
            const uint32_t page = ctx.frame.fbp & 0x1FFu;
            const uint16_t drawnTo = static_cast<uint16_t>(ctx.scissor.y1 + 1u);
            uint16_t seen = g_ps2xFbDrawnHeight[page].load(std::memory_order_relaxed);
            while (drawnTo > seen &&
                   !g_ps2xFbDrawnHeight[page].compare_exchange_weak(seen, drawnTo, std::memory_order_relaxed))
            {
            }
        }
        if (state.prim.tme)
        {
            // Render-to-texture: the GPU holds that image and GS memory does not.
            // Page alignment alone means nothing -- ordinary textures land on page
            // boundaries all the time -- so require the format and stride to match
            // the buffer, and that buffer to have been drawn into recently.
            // PS2X_GS_RTT=1 re-enables this. It is off because the test cannot
            // actually tell a render-to-texture read from an ordinary texture that
            // happens to share a page-aligned base, pixel format and stride with a
            // buffer the game draws into -- and this game has ~660 such draws per
            // frame, which sampled the live framebuffer and made buildings and
            // textures flash. The drive-in movie screen
            // works from VRAM, because the transfers that fill it invalidate the
            // cached texture and it is decoded again.
            static const bool rttEnabled = [] {
                const char *p = std::getenv("PS2X_GS_RTT");
                return p && p[0] == '1';
            }();
            const uint32_t texPage = ctx.tex0.tbp0 / 32u;
            auto known = impl.knownTargets.find(texPage);
            // Ghidra: kitePacketInit (0x212470) fixes this CLAMP; texture ID 0
            // in texmParticleTexture (0x205b40) reads the displayed CT24 buffer.
            // Scope this to that exact packet, not the unsafe general RTT heuristic.
            static const bool sceneReads = [] { const char* p = std::getenv("PS2X_GS_SCENE_REFRACTION");
                return !p || p[0] != '0'; }();
            const bool sceneRefraction = sceneReads && ctx.tex0.psm == 1u && ctx.tex0.tbw == 10u &&
                ctx.tex0.tw == 10u && ctx.tex0.th == 8u && ctx.tex0.tcc == 0u && ctx.tex0.tfx == 0u &&
                ctx.clamp == 0x378019f801aull;
            static uint32_t sceneReports = 0;
            if (sceneRefraction && sceneReports++ < 8u)
                std::fprintf(stderr, "[gs:scene-refraction] frame=%llu source=%x destination=%x known=%d\n",
                    (unsigned long long)impl.frames, texPage, ctx.frame.fbp, known != impl.knownTargets.end());
            const bool isRenderTarget = (rttEnabled || sceneRefraction) && known != impl.knownTargets.end() &&
                                        known->second.blocks == ctx.tex0.tbp0 &&
                                        (sceneRefraction || texPage != ctx.frame.fbp) && // scene reads use a GPU snapshot
                                        known->second.psm == ctx.tex0.psm && known->second.fbw == ctx.tex0.tbw &&
                                        known->second.lastDrawFrame + 2u >= impl.frames;
            if (isRenderTarget)
            {
                call.textureFbp = texPage;
                ++impl.renderTargetReads;
            }
            else
            {
                call.texture = impl.textureFor(m_cpu, state);
                if (call.texture && call.texture == impl.xboxButtons) {
                    call.linear = call.mmag = true;
                    call.mmin = 1u;
                    // Region clamp registers remain in retail's 64x64 texels.
                    for (int axis = 0; axis < 2; ++axis) {
                        const int mode = axis == 0 ? call.wrapU : call.wrapV;
                        if (mode == 2) {
                            call.region[axis * 2] *= 4.0f;
                            call.region[axis * 2 + 1] = (call.region[axis * 2 + 1] + 1.0f) * 4.0f - 1.0f;
                        } else if (mode == 3) {
                            // REGION_REPEAT uses AND/OR masks, not min/max bounds.
                            call.region[axis * 2] = (call.region[axis * 2] + 1.0f) * 4.0f - 1.0f;
                            call.region[axis * 2 + 1] *= 4.0f;
                        }
                    }
                }
            }
        }
    
        if (reuse && preparedVerify)
        {
            if (!call.sameStateAs(impl.preparedCall) || !call.sameStateAs(impl.calls.back()) || call.tbp0 != impl.preparedCall.tbp0 ||
                call.cbp != impl.preparedCall.cbp || call.texPsm != impl.preparedCall.texPsm)
            {
                std::fprintf(stderr, "[gs:prepared] MISMATCH frame=%llu\n", (unsigned long long)impl.frames);
                std::abort();
            }
            ++impl.preparedChecks;
        }
        impl.preparedState = state;
        impl.preparedCall = call;
        impl.preparedFrame = impl.frames;
        impl.preparedGlyphFamily = glyphFamily;
        impl.preparedValid = true;
        impl.frameFbp = ctx.frame.fbp;
        impl.frameFbw = ctx.frame.fbw;
        impl.framePsm = static_cast<uint32_t>(ctx.frame.psm);
        if (impl.calls.empty() || (impl.haveBoundary && impl.calls.size() == impl.boundaryCalls) ||
            impl.calls.back().clear || !impl.calls.back().sameStateAs(call))
        {
            call.first = impl.vertices.size();
            call.count = 0u;
            impl.calls.push_back(call);
        }
        impl.preparedCallCount = impl.calls.size();
    }
    if (reuse) ++impl.preparedHits; else ++impl.preparedMisses;
    if (preparedVerify && ((impl.preparedHits + impl.preparedMisses) % 1000000u) == 0u)
        std::fprintf(stderr, "[gs:prepared] hits=%llu misses=%llu checked=%llu mismatches=0\n",
            (unsigned long long)impl.preparedHits, (unsigned long long)impl.preparedMisses,
            (unsigned long long)impl.preparedChecks);

    const float ofx = static_cast<float>(ctx.xyoffset.ofx) / 16.0f;
    // The game renders interlaced fields: it alternates XYOFFSET.OFY by half a
    // scanline every frame so that, on a CRT, the two fields land on different
    // physical lines and reconstruct full vertical detail. We are not a CRT --
    // FFMD=1 line-doubles both fields onto the same host rows -- so that half
    // line stops being a field offset and becomes the whole picture translating
    // half a pixel every frame, which reads as a vertical bob. Drop the
    // fraction so both fields share one origin. Flooring rather than rounding
    // matters: rounding would send .0 and .5 to opposite lines and make it
    // worse. (PS2X_GS_OFY_SNAP=0 keeps the exact offset.)
    static const bool snapOfy = [] {
        const char *p = std::getenv("PS2X_GS_OFY_SNAP");
        return !(p && p[0] == '0');
    }();
    float ofy = static_cast<float>(ctx.xyoffset.ofy) / 16.0f;
    if (snapOfy)
        ofy = std::floor(ofy);
    if (Impl::tracing())
        impl.traceOfy[static_cast<int32_t>(ctx.xyoffset.ofy)] += 1u;
    const float texW = static_cast<float>(state.textureWidth != 0u ? state.textureWidth : 1u);
    const float texH = static_cast<float>(state.textureHeight != 0u ? state.textureHeight : 1u);
    // Hud::buildPacketHead (0x1446a8) writes TEST=0x8d: depth is disabled.
    // Other overlay sprites use ALWAYS with depth writes masked. Retail's
    // fontInitPacket (0x1fd458) uses PRIM=0x156 / TEST=0x3008d and inherits
    // ZBUF without setting ZMSK. These fixed-UV font sprites still need the
    // same horizontal projection as the camera-relative tutorial panel/icons.
    // Keep the exception scoped to that font setup, including its 512x512
    // PSMT8H atlas selected after initialization; depth-tested sprites and
    // perspective meshes stay unchanged.
    const bool retailFont = state.prim.type == GS_PRIM_SPRITE &&
        state.prim.fst && state.prim.tme && state.prim.abe &&
        ctx.test == 0x3008dull && ctx.tex0.psm == 0x1bu &&
        state.textureWidth == 512u && state.textureHeight == 512u;
    const bool wideUi = g_ps2xWotmWideActive.load(std::memory_order_relaxed) &&
        state.prim.fst && (!(ctx.test & (1ull << 16u)) || retailFont ||
            (ctx.zbuf.zmask && ((ctx.test >> 17u) & 3u) == 1u)) &&
        ctx.scissor.x0 == 0u && ctx.scissor.x1 == 639u && impl.calls.back().textureFbp == kNoFbp;
    auto convert = [&](const GSVertex &v)
    {
        GlVertex out{};
        out.x = v.x - ofx;
        if (wideUi) out.x = 320.0f + (out.x - 320.0f) * (45.0f / 56.0f);
        out.y = v.y - ofy;
        if (Impl::tracing())
        {
            impl.traceMinY = impl.traceAnyY ? std::min<float>(impl.traceMinY, out.y) : out.y;
            impl.traceMaxY = impl.traceAnyY ? std::max<float>(impl.traceMaxY, out.y) : out.y;
            impl.traceAnyY = true;
        }
        out.z = static_cast<float>(v.z / 16777215.0);
        // Colour channels are framebuffer values (255 = white); alpha is a
        // blend and alpha-test operand, where 128 is 1.0.
        out.r = static_cast<float>(v.r) / 255.0f;
        out.g = static_cast<float>(v.g) / 255.0f;
        out.b = static_cast<float>(v.b) / 255.0f;
        out.a = static_cast<float>(v.a) / 128.0f;
        out.fog = static_cast<float>(v.fog);
        if (state.prim.fst)
        {
            // UV is in 12.4 fixed-point texels; the shader wants S/T/Q, so hand
            // it the normalised coordinate with Q = 1.
            out.s = static_cast<float>(v.u) / 16.0f / texW;
            out.t = static_cast<float>(v.v) / 16.0f / texH;
            out.q = 1.0f;
        }
        else
        {
            // S,T,Q go through untouched: the GS interpolates them linearly in
            // screen space and divides per pixel, and so does the shader.
            out.s = v.s;
            out.t = v.t;
            out.q = v.q;
        }
        return out;
    };

    if (count >= 3u)
    {
        const GSVertex *input = stream ? stream->vertices : batch.vertices.data();
        const size_t vertexCount = stream ? stream->count : 3u;
        static const bool sharedVertices=[] {
            const char *p=std::getenv("PS2X_GL_SHARED_VERTICES");return !p || p[0]!='0';
        }();
        static const bool sharedVerify=std::getenv("PS2X_GL_SHARED_VERIFY")!=nullptr;
        // Adjacent strip triangles share their last two vertices; triangle
        // fans share the center and previous edge. Reuse packed values only
        // within this immutable material stream, preserving input order.
        constexpr size_t vertexBytes=offsetof(GSVertex,fog)+sizeof(uint8_t);
        static const bool bulkVertices=[] {
            const char *p=std::getenv("PS2X_GL_BULK_VERTICES");return !p || p[0]!='0';
        }();
        const size_t firstVertex=impl.vertices.size();
        if(bulkVertices) {
            if(Impl::frameStorageStats() && firstVertex+vertexCount>impl.vertices.capacity()) {
                ++impl.storageVertexGrowths;impl.storageVerticesMoved+=firstVertex;
            }
            impl.vertices.resize(firstVertex+vertexCount);
        }
        auto append=[&](size_t i,const GlVertex& value) {
            if(bulkVertices)impl.vertices[firstVertex+i]=value;
            else impl.vertices.push_back(value);
        };
        if (stream && stream->indices && !Impl::tracing()) {
            // A referenced vertex is immutable within this material stream.
            // ADC gaps do not require adjacency checks or duplicate conversion.
            static thread_local std::array<GlVertex,256> converted;
            std::array<uint8_t,256> ready{};
            static const bool indexedVerify=std::getenv("PS2X_GL_INDEXED_VERIFY")!=nullptr;
            static const bool linearIndexed=[] {
                const char *p=std::getenv("PS2X_GL_LINEAR_INDEXED");return !p || p[0]!='0';
            }();
            // Dense strips reference their input repeatedly. Convert in input
            // order first, then assemble triangles without a per-index cache
            // probe. Sparse streams retain lazy conversion. Tracing above
            // retains its original order and never converts unused vertices.
            const bool linear=linearIndexed && stream->uniqueCount<=vertexCount;
            static const bool worldVertices=[] {
                const char *p=std::getenv("PS2X_GL_WORLD_VERTICES");return !p || p[0]!='0';
            }();
            static const bool worldVerify=std::getenv("PS2X_GL_WORLD_VERIFY")!=nullptr;
            static const bool worldBench=std::getenv("PS2X_GL_WORLD_BENCH")!=nullptr;
            const bool world=linear && !state.prim.fst && !wideUi && stream->uniqueCount>=32u;
            if(stream->packed) {
                if(!world || !worldVertices)std::abort();
                wotm_vertex::convertPackedWorldVertices(converted.data(),stream->packed,stream->uniqueCount,ofx,ofy);
                static const bool packedVerify=std::getenv("PS2X_GL_PACKED_WORLD_VERIFY")!=nullptr;
                if(packedVerify) {
                    static thread_local std::array<GSVertex,256> decoded;
                    static thread_local std::array<GlVertex,256> reference;
                    for(size_t i=0;i<stream->uniqueCount;++i)
                        decoded[i]=wotm_vertex::decodePackedGSVertex<GSVertex>(stream->packed+i*48u,stream->u,stream->v);
                    wotm_vertex::convertWorldVertices(reference.data(),decoded.data(),stream->uniqueCount,ofx,ofy);
                    if(std::memcmp(reference.data(),converted.data(),stream->uniqueCount*sizeof(GlVertex))) {
                        std::fprintf(stderr,"[gs:packed-world] MISMATCH\n");std::abort();
                    }
                    static thread_local uint64_t checks=0,vertices=0;
                    ++checks;vertices+=stream->uniqueCount;
                    if(checks==1u || checks%100000u==0u)
                        std::fprintf(stderr,"[gs:packed-world] checks=%llu vertices=%llu mismatches=0\n",
                            (unsigned long long)checks,(unsigned long long)vertices);
                }
            } else if (world && worldVertices)
                wotm_vertex::convertWorldVertices(converted.data(),input,stream->uniqueCount,ofx,ofy);
            else if (linear)
                for (size_t i=0;i<stream->uniqueCount;++i) converted[i]=convert(input[i]);
            if (world && (worldVerify || worldBench)) {
                // Keep diagnostic scratch out of the normal Submit stack frame.
                static thread_local std::array<GlVertex,256> reference;
                if (worldBench && g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed)>=2350u) {
                    // Compare both paths on this exact immutable material stream.
                    // Diagnostic timers are absent from normal execution.
                    struct Timings {uint64_t calls=0,vertices=0,oldNs=0,newNs=0;};
                    static thread_local Timings timings;
                    if (timings.calls<1024u) {
                        auto oldConvert=[&] {
                            for(size_t i=0;i<stream->uniqueCount;++i)reference[i]=convert(input[i]);
                        };
                        auto newConvert=[&] {
                            wotm_vertex::convertWorldVertices(converted.data(),input,stream->uniqueCount,ofx,ofy);
                        };
                        auto timed=[&](auto&& action) {
                            const auto begin=std::chrono::steady_clock::now();
                            action();
                            return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now()-begin).count());
                        };
                        if (timings.calls&1u) {
                            timings.newNs+=timed(newConvert);timings.oldNs+=timed(oldConvert);
                        } else {
                            timings.oldNs+=timed(oldConvert);timings.newNs+=timed(newConvert);
                        }
                        ++timings.calls;timings.vertices+=stream->uniqueCount;
                        if(timings.calls==1u || timings.calls==1024u)
                            std::fprintf(stderr,"[gs:world-bench] calls=%llu vertices=%llu oldNs=%llu newNs=%llu\n",
                                (unsigned long long)timings.calls,(unsigned long long)timings.vertices,
                                (unsigned long long)timings.oldNs,(unsigned long long)timings.newNs);
                    } else
                        for(size_t i=0;i<stream->uniqueCount;++i)reference[i]=convert(input[i]);
                } else
                    for(size_t i=0;i<stream->uniqueCount;++i)reference[i]=convert(input[i]);
                if(std::memcmp(reference.data(),converted.data(),stream->uniqueCount*sizeof(GlVertex))) {
                    std::fprintf(stderr,"[gs:world-vertices] MISMATCH\n");std::abort();
                }
                static thread_local uint64_t checks=0,vertices=0;
                ++checks;vertices+=stream->uniqueCount;
                if(checks==1u || checks%100000u==0u)
                    std::fprintf(stderr,"[gs:world-vertices] checks=%llu vertices=%llu native=%d mismatches=0\n",
                        (unsigned long long)checks,(unsigned long long)vertices,int(worldVertices));
            }
            for (size_t i = 0; i < vertexCount; ++i) {
                const uint16_t index=stream->indices[i];
                if (index>=stream->uniqueCount) std::abort();
                if (!linear && !ready[index]) { converted[index]=convert(input[index]); ready[index]=1; }
                if (indexedVerify) {
                    const auto expected=convert(input[index]);
                    if (std::memcmp(&expected,&converted[index],sizeof(expected))) {
                        std::fprintf(stderr,"[gs:indexed-convert] MISMATCH\n");std::abort();
                    }
                }
                append(i,converted[index]);
            }
        } else for (size_t i = 0; i < vertexCount; i+=3) {
            const bool indexed=stream && stream->indices;
            const bool canReuse=sharedVertices && stream && !indexed && i>=3 && !Impl::tracing();
            const bool strip=canReuse &&
                std::memcmp(input+i,input+i-2,vertexBytes)==0 &&
                std::memcmp(input+i+1,input+i-1,vertexBytes)==0;
            const bool fan=canReuse && !strip && state.prim.type==GS_PRIM_TRIFAN &&
                std::memcmp(input+i,input+i-3,vertexBytes)==0 &&
                std::memcmp(input+i+1,input+i-1,vertexBytes)==0;
            if(strip || fan) {
                const GlVertex a=impl.vertices[firstVertex+i-(strip ? 2u : 3u)];
                const GlVertex b=impl.vertices[firstVertex+i-1];
                if(sharedVerify) {
                    const auto x=convert(input[i]),y=convert(input[i+1]);
                    if(std::memcmp(&a,&x,sizeof(a)) || std::memcmp(&b,&y,sizeof(b))) {
                        std::fprintf(stderr,"[gs:shared-vertices] MISMATCH\n");std::abort();
                    }
                }
                append(i,a);append(i+1,b);
            } else {
                append(i,convert(input[indexed ? stream->indices[i] : i]));
                append(i+1,convert(input[indexed ? stream->indices[i+1] : i+1]));
            }
            append(i+2,convert(input[indexed ? stream->indices[i+2] : i+2]));
        }
        impl.calls.back().count += vertexCount;
        impl.primitives += vertexCount / 3;
        // Optional end-to-end stream digest for captured A/B replay: includes
        // converted host vertices and material, independently of batch sizes.
        static const bool digestEnabled = std::getenv("PS2X_GS_GEOMETRY_DIGEST") != nullptr;
        if (digestEnabled) {
            extern uint64_t g_ps2xGeometryDigest;
            auto hash = [&](const void *p, size_t n) {
                const auto *b = static_cast<const uint8_t *>(p);
                for (size_t j = 0; j < n; ++j) g_ps2xGeometryDigest = (g_ps2xGeometryDigest ^ b[j]) * 1099511628211ull;
            };
            for (size_t i = impl.vertices.size() - vertexCount; i < impl.vertices.size(); i += 3) {
                hash(&impl.vertices[i], 3 * sizeof(GlVertex));
                hash(&state, sizeof(state));
            }
        }
        return;
    }

    // Sprite: two opposite corners become an axis-aligned quad. Like the GS (and
    // the CPU rasterizer), the rectangle is normalised but the texture
    // coordinates are not swapped with it, and colour and Z come from vertex 2.
    GlVertex a = convert(batch.vertices[0]);
    GlVertex b = convert(batch.vertices[1]);
    if (a.x > b.x)
        std::swap(a.x, b.x);
    if (a.y > b.y)
        std::swap(a.y, b.y);
    a.z = b.z;
    a.r = b.r;
    a.g = b.g;
    a.b = b.b;
    a.a = b.a;
    GlVertex topRight = b; // (x1, y0)
    topRight.y = a.y;
    topRight.t = a.t;
    GlVertex bottomLeft = a; // (x0, y1)
    bottomLeft.y = b.y;
    bottomLeft.t = b.t;
    impl.vertices.push_back(a);
    impl.vertices.push_back(topRight);
    impl.vertices.push_back(b);
    impl.vertices.push_back(a);
    impl.vertices.push_back(b);
    impl.vertices.push_back(bottomLeft);
    impl.calls.back().count += 6u;
    ++impl.primitives;
}

void GSGlBackend::BeginTransfer(const GSTransferCommand &command)
{
    m_cpu.BeginTransfer(command);
    if (m_impl->failed || command.direction == 1u)
        return; // 1 = local -> host: reads nothing we cache

    // Whatever this writes, textures decoded from it are now stale.
    Impl &impl = *m_impl;
    const uint32_t width =
        std::max<uint32_t>(static_cast<uint32_t>(command.bitbltbuf.dbw) * 64u,
                           static_cast<uint32_t>(command.trxpos.dsax) + command.trxreg.rrw);
    const uint32_t height = static_cast<uint32_t>(command.trxpos.dsay) + command.trxreg.rrh;
    uint32_t begin = 0u, end = 0u;
    blockRange(command.bitbltbuf.dbp, width, std::max<uint32_t>(height, 1u),
               command.bitbltbuf.dpsm, begin, end);
    std::lock_guard<std::mutex> lock(impl.recordMutex);
    impl.invalidate(begin, end);
    impl.noteTransfer(begin, end);

    // A transfer into a buffer we render into has to be replayed into that
    // render target, or the GPU copy is missing whatever the game blits in.
    // Base, stride and pixel format must all match, so an ordinary texture
    // upload (dbw = 1 for a 64-pixel texture) cannot be mistaken for one.
    impl.pendingBlit.active = false;
    const uint32_t dstPage = command.bitbltbuf.dbp / 32u;
    auto known = impl.knownTargets.find(dstPage);
    if (known != impl.knownTargets.end() && known->second.blocks == command.bitbltbuf.dbp &&
        known->second.fbw == command.bitbltbuf.dbw && known->second.psm == command.bitbltbuf.dpsm &&
        command.trxreg.rrw != 0u && command.trxreg.rrh != 0u)
    {
        const uint32_t bpp = GSInternal::bitsPerPixel(command.bitbltbuf.dpsm);
        impl.pendingBlit.active = true;
        impl.pendingBlit.fbp = dstPage;
        impl.pendingBlit.x = command.trxpos.dsax;
        impl.pendingBlit.y = command.trxpos.dsay;
        impl.pendingBlit.width = command.trxreg.rrw;
        impl.pendingBlit.height = command.trxreg.rrh;
        impl.pendingBlit.psm = command.bitbltbuf.dpsm;
        impl.pendingBlit.expectedBytes =
            static_cast<size_t>(command.trxreg.rrw) * command.trxreg.rrh * bpp / 8u;
        impl.pendingBlit.data.clear();
        impl.pendingBlit.data.reserve(impl.pendingBlit.expectedBytes);
    }
}

// The block range the display circuit reads for this frame.
static void displayBufferRange(uint64_t dispfb, uint32_t &begin, uint32_t &end)
{
    const uint32_t fbp = static_cast<uint32_t>(dispfb & 0x1FFu);
    const uint32_t fbw = static_cast<uint32_t>((dispfb >> 9) & 0x3Fu);
    const uint32_t psm = static_cast<uint32_t>((dispfb >> 15) & 0x1Fu);
    blockRange(GSInternal::framePageBaseToBlock(fbp), std::max<uint32_t>(fbw * 64u, 64u), 448u,
               psm, begin, end);
}

// Was the buffer about to be displayed filled by transfers rather than by
// drawing? Answered for the frontend before it decides to pay for a snapshot.
void GSGlBackend::MarkPresentBoundary()
{
    Impl &impl = *m_impl;
    std::lock_guard<std::mutex> lock(impl.recordMutex);
    impl.boundaryCalls = impl.calls.size();
    impl.boundaryVertices = impl.vertices.size();
    impl.haveBoundary = true;
    // The last recorded draw belongs to the sealed frame. New geometry must
    // start a new draw even when its material is unchanged.
    impl.preparedValid = false;
    if (gpuReferenceEnabled())
        impl.referencePresentToken = m_cpu.EnqueuePresentSnapshot();
}

bool GSGlBackend::NeedsPresentSnapshot(const GSPresentationRequest &request)
{
    Impl &impl = *m_impl;
    if (impl.failed)
        return true; // no GPU: everything comes from local memory
    const bool circuit2 = (request.pmode & 0x2u) != 0u;
    const uint64_t dispfb = circuit2 ? request.dispfb2 : request.dispfb1;
    const uint32_t displayFbp = static_cast<uint32_t>(dispfb & 0x1FFu);

    // Render-target contents persist until overwritten, even when loading
    // pauses drawing for many fields. Expiring them after eight presents
    // switched the static loading artwork to an incomplete CPU VRAM copy.
    // Match the actual display layout; unknown/movie-only buffers still use
    // local memory. Matching host transfers are replayed by recordBlit.
    std::lock_guard<std::mutex> lock(impl.recordMutex);
    auto known = impl.knownTargets.find(displayFbp);
    const uint32_t displayWidth = (dispfb >> 9u) & 0x3fu;
    const uint32_t displayFormat = (dispfb >> 15u) & 0x1fu;
    const bool stale = known == impl.knownTargets.end() ||
                       known->second.fbw != displayWidth ||
                       known->second.psm != displayFormat;
    static uint64_t s_reports = 0u;
    if (stale && (s_reports++ % 300u) == 0u)
    {
        std::string drawn;
        for (const auto &entry : impl.knownTargets)
        {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "0x%x@%llu ", entry.first, (unsigned long long)entry.second.lastDrawFrame);
            drawn += buf;
        }
        std::fprintf(stderr, "[gs:gl] local-memory present: display fbp 0x%x, frame %llu, drawn into: %s\n",
                     displayFbp, (unsigned long long)impl.frames, drawn.empty() ? "(nothing)" : drawn.c_str());
    }
    return stale;
}

void GSGlBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    m_cpu.UploadImage(data, sizeBytes); // local memory stays authoritative
    if (m_impl->failed || !data || sizeBytes == 0u)
        return;
    Impl &impl = *m_impl;
    std::lock_guard<std::mutex> lock(impl.recordMutex);
    if (!impl.pendingBlit.active)
        return;
    // A transfer arrives in chunks; replay it once it is complete.
    Impl::PendingBlit &blit = impl.pendingBlit;
    blit.data.insert(blit.data.end(), data, data + sizeBytes);
    if (blit.data.size() < blit.expectedBytes)
        return;
    blit.active = false;
    impl.recordBlit(blit);
}

// Turns a completed frame-buffer transfer into a textured quad in the recorded
// draw list, so it replays into the render target in submission order.
void GSGlBackend::Impl::recordBlit(const PendingBlit &blit)
{
    Impl &impl = *this;
    auto entry = std::make_shared<TextureEntry>();
    entry->width = blit.width;
    entry->height = blit.height;
    entry->rgba.assign(static_cast<size_t>(blit.width) * blit.height * 4u, 255u);
    // Transfer data arrives in raster order in the destination's pixel format.
    const uint8_t *src = blit.data.data();
    const size_t available = blit.data.size();
    for (uint32_t i = 0; i < blit.width * blit.height; ++i)
    {
        uint8_t *dst = entry->rgba.data() + static_cast<size_t>(i) * 4u;
        switch (blit.psm)
        {
        case GS_PSM_CT32:
        {
            const size_t offset = static_cast<size_t>(i) * 4u;
            if (offset + 4u > available)
                break;
            dst[0] = src[offset];
            dst[1] = src[offset + 1u];
            dst[2] = src[offset + 2u];
            dst[3] = src[offset + 3u];
            break;
        }
        case GS_PSM_CT24:
        {
            const size_t offset = static_cast<size_t>(i) * 3u;
            if (offset + 3u > available)
                break;
            dst[0] = src[offset];
            dst[1] = src[offset + 1u];
            dst[2] = src[offset + 2u];
            dst[3] = 128u; // no alpha in 24-bit: opaque on the 128 = 1.0 scale
            break;
        }
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        {
            const size_t offset = static_cast<size_t>(i) * 2u;
            if (offset + 2u > available)
                break;
            const uint16_t pixel = static_cast<uint16_t>(src[offset] | (src[offset + 1u] << 8));
            dst[0] = static_cast<uint8_t>((pixel & 0x1Fu) << 3);
            dst[1] = static_cast<uint8_t>(((pixel >> 5) & 0x1Fu) << 3);
            dst[2] = static_cast<uint8_t>(((pixel >> 10) & 0x1Fu) << 3);
            dst[3] = (pixel & 0x8000u) ? 128u : 0u;
            break;
        }
        default:
            return; // paletted or depth destination: not a picture to blit
        }
    }

    Impl::DrawCall call{};
    call.fbp = blit.fbp;
    entry->transient = true;
    call.texture = entry;
    call.linear = true;
    call.mmag = true;
    call.tfx = 1u; // DECAL: the transferred pixels replace what is there
    call.tcc = 0u; // ... and keep their own alpha out of it
    call.abe = false;
    call.test = 0u;  // no depth test
    call.zmsk = 1u;  // and no depth write
    call.sx = 0;
    call.sy = 0;
    call.sw = static_cast<int>(blit.x + blit.width);
    call.sh = static_cast<int>(blit.y + blit.height);
    call.first = impl.vertices.size();
    call.count = 6u;

    const float x0 = static_cast<float>(blit.x);
    const float y0 = static_cast<float>(blit.y);
    const float x1 = x0 + static_cast<float>(blit.width);
    const float y1 = y0 + static_cast<float>(blit.height);
    auto push = [&](float x, float y, float u, float v)
    {
        GlVertex out{};
        out.x = x;
        out.y = y;
        out.z = 0.0f;
        out.r = out.g = out.b = 1.0f;
        out.a = 1.0f;
        out.s = u;
        out.t = v;
        out.q = 1.0f;
        out.fog = 255.0f;
        impl.vertices.push_back(out);
    };
    push(x0, y0, 0.0f, 0.0f);
    push(x1, y0, 1.0f, 0.0f);
    push(x1, y1, 1.0f, 1.0f);
    push(x0, y0, 0.0f, 0.0f);
    push(x1, y1, 1.0f, 1.0f);
    push(x0, y1, 0.0f, 1.0f);
    impl.calls.push_back(call);
    ++impl.blits;

    // The buffer now holds a picture the GPU made, so it can be presented from
    // the render target like any other.
    Impl::KnownTarget &known = impl.knownTargets[blit.fbp];
    known.lastDrawFrame = impl.frames;
}
void GSGlBackend::Flush() {} // drawing happens at Present, on the GL thread
void GSGlBackend::TextureFlush() { m_cpu.TextureFlush(); }

void GSGlBackend::Sync(GSSyncReason reason)
{
    m_cpu.Sync(reason);
}

PresentationFrame GSGlBackend::Present(const GSPresentationRequest &inputRequest)
{
    Impl &impl = *m_impl;
    GSPresentationRequest request=inputRequest;
    GlVertices verts;
    std::vector<Impl::DrawCall> calls;
    // Declared last: every return or exception releases only fully consumed
    // local storage, before the vectors themselves are destroyed.
    struct ReturnStorage {
        Impl& impl;GlVertices& vertices;std::vector<Impl::DrawCall>& calls;
        ~ReturnStorage(){impl.recycleFrameStorage(vertices,calls);}
    } returnStorage{impl,verts,calls};
    uint64_t completedFrame=0;
    {
        std::lock_guard<std::mutex> lock(impl.recordMutex);
        static const bool useBoundary = [] {
            const char *p = std::getenv("PS2X_GS_GPU_BOUNDARY");
            return !(p && p[0] == '0');
        }();
        if(completedHandoffEnabled() && !impl.completedDraws.empty()) {
            auto& ready=impl.completedDraws.front();verts.swap(ready.vertices);calls.swap(ready.calls);
            request=ready.display;completedFrame=ready.frame;impl.completedDraws.pop_front();impl.haveBoundary=false;
        }
        else if (!impl.haveBoundary || !useBoundary)
        {
            // No boundary (a present the frontend did not mark): take it all.
            verts.swap(impl.vertices);
            calls.swap(impl.calls);
            impl.acquireFrameStorage(impl.vertices,impl.calls);
        }
        else
        {
            // Replay this frame only; whatever the game submitted after the
            // boundary belongs to the next one and stays recorded.
            impl.acquireFrameStorage(verts,calls);
            const size_t splitCalls = std::min<size_t>(impl.boundaryCalls, impl.calls.size());
            const size_t splitVerts = std::min<size_t>(impl.boundaryVertices, impl.vertices.size());
            calls.assign(impl.calls.begin(), impl.calls.begin() + splitCalls);
            verts.assign(impl.vertices.begin(), impl.vertices.begin() + splitVerts);
            impl.calls.erase(impl.calls.begin(), impl.calls.begin() + splitCalls);
            impl.vertices.erase(impl.vertices.begin(), impl.vertices.begin() + splitVerts);
            // The calls left behind index into the vertices left behind.
            for (Impl::DrawCall &call : impl.calls)
                call.first = call.first >= splitVerts ? call.first - splitVerts : 0u;
            impl.haveBoundary = false;
        }
    }
#ifdef _WIN32
    if(ps2xD3D12Enabled()) {
        {std::lock_guard<std::mutex> lock(impl.recordMutex);impl.retired.clear();}
        const uint64_t display=request.display2?request.display2:request.display1;
        uint32_t w=((display>>32)&0xfffu)+1u,h=((display>>44)&0x7ffu)+1u;
        w/=((display>>23)&15u)+1u;h/=((display>>27)&3u)+1u;
        if(!w||w>Impl::kNativeWidth)w=Impl::kNativeWidth;
        if(!h||h>Impl::kNativeHeight)h=Impl::kNativeHeight;
        if(NeedsPresentSnapshot(request)) {
            auto frame=m_cpu.PresentSnapshot(request.presentToken?request.presentToken:m_cpu.EnqueuePresentSnapshot(),request);
            if(!frame.pixels.empty())ps2xD3D12Movie(frame.pixels.data(),frame.width,frame.height);
            ++impl.frames;return frame;
        }
        const char* scaleSpec=std::getenv("PS2X_GS_SCALE");
        impl.scale=uint32_t(std::clamp(scaleSpec?std::atoi(scaleSpec):1,1,8));
        impl.width=Impl::kNativeWidth*impl.scale;impl.height=Impl::kNativeHeight*impl.scale;
        wotm_capture::Frame packet;packet.width=impl.width;packet.height=impl.height;packet.scale=impl.scale;
        packet.display=uint32_t(((request.pmode&2u)?request.dispfb2:request.dispfb1)&0x1ffu);
        impl.preparePacket(packet,verts,calls,true);
        g_ps2xGsPresentedGameFrame.store(completedFrame,std::memory_order_release);
        ps2xD3D12Frame(packet,w,h);++impl.frames;
        PresentationFrame frame{};frame.width=w;frame.height=h;frame.displayFbp=packet.display;frame.sourceFbp=packet.display;
        return frame;
    }
#endif
    if (impl.failed)
        return m_cpu.Present(request);
    {
        std::lock_guard<std::mutex> lock(impl.glMutex);
        if (!impl.ensure()) // creates the context and objects on first use
            return m_cpu.Present(request);
    }
    static const uint64_t dumpFrame = [] {
        const char *p = std::getenv("PS2X_GS_GPU_DUMPCALLS");
        return p ? std::strtoull(p, nullptr, 10) : ~0ull;
    }();
    if (impl.frames == dumpFrame)
    {
        for (const Impl::DrawCall &c : calls)
        {
            float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
            for (size_t i = c.first; i < c.first + c.count && i < verts.size(); ++i)
            {
                x0 = std::min<float>(x0, verts[i].x);
                x1 = std::max<float>(x1, verts[i].x);
                y0 = std::min<float>(y0, verts[i].y);
                y1 = std::max<float>(y1, verts[i].y);
            }
            std::fprintf(stderr,
                         "[gs:call] verts=%zu bbox=(%.0f,%.0f)-(%.0f,%.0f) tex=%d tbp0=0x%x psm=0x%02x cbp=0x%x "
                         "%ux%u tfx=%u tcc=%u abe=%d alpha=%llx test=%llx wrap=%d,%d v0=(%.2f,%.2f,%.2f,%.2f) stq0=(%.3f,%.3f,%.3f)\n",
                         c.count, x0, y0, x1, y1, c.texture ? 1 : 0, c.tbp0, c.texPsm, c.cbp,
                         c.texture ? c.texture->width : 0u, c.texture ? c.texture->height : 0u, c.tfx, c.tcc,
                         c.abe ? 1 : 0, (unsigned long long)c.alpha, (unsigned long long)c.test, c.wrapU, c.wrapV,
                         c.first < verts.size() ? verts[c.first].r : -1.0f,
                         c.first < verts.size() ? verts[c.first].g : -1.0f,
                         c.first < verts.size() ? verts[c.first].b : -1.0f,
                         c.first < verts.size() ? verts[c.first].a : -1.0f,
                         c.first < verts.size() ? verts[c.first].s : -1.0f,
                         c.first < verts.size() ? verts[c.first].t : -1.0f,
                         c.first < verts.size() ? verts[c.first].q : -1.0f);
        }
    }

    // Decide before touching the GPU: a frame that comes out of local memory
    // does not need the render target cleared and redrawn at all.
    if (NeedsPresentSnapshot(request))
    {
        const bool circuit2Early = (request.pmode & 0x2u) != 0u;
        const uint32_t wanted =
            static_cast<uint32_t>((circuit2Early ? request.dispfb2 : request.dispfb1) & 0x1FFu);
        ++impl.frames;
        if ((++impl.presentsFromCpu % 300u) == 0u)
            std::fprintf(stderr,
                         "[gs:gl] %llu frames presented from local memory (fbp 0x%x holds transferred pixels, not drawn ones)\n",
                         (unsigned long long)impl.presentsFromCpu, wanted);
        impl.publishedTexture.store(0u, std::memory_order_release);
        // The ordered snapshot the frontend queued while it held the state
        // lock. Without it this is the unordered copy that shows a buffer the
        // game is part way through refilling -- a half drawn screen.
        if (request.presentToken != 0u)
            return m_cpu.PresentSnapshot(request.presentToken, request);
        return m_cpu.PresentSnapshot(m_cpu.EnqueuePresentSnapshot(), request);
    }

    Impl::Scope scope(impl); // borrows the thread's context and restores it
    impl.collectRetired();
    impl.noteReplaySize(calls.size());
    impl.replayTiming.start(impl.frames,calls.size());
    impl.captureReplay(verts, calls, static_cast<uint32_t>(((request.pmode & 2u) ? request.dispfb2 : request.dispfb1) & 0x1ffu));
    impl.replayTiming.finish();

    // Present the buffer the display circuit is actually reading.
    const bool circuit2 = (request.pmode & 0x2u) != 0u;
    const uint64_t dispfb = circuit2 ? request.dispfb2 : request.dispfb1;
    uint32_t displayFbp = static_cast<uint32_t>(dispfb & 0x1FFu);
    if (impl.targets.find(displayFbp) == impl.targets.end())
        displayFbp = impl.frameFbp; // never drawn into: fall back to the last one
    Impl::RenderTarget *target = nullptr;
    {
        auto it = impl.targets.find(displayFbp);
        if (it != impl.targets.end())
            target = &it->second;
    }

    // Did anything draw into the buffer being displayed more recently than a
    // transfer wrote it? If not, its picture is not in the render target and
    // local memory has to present this frame.
    if (!target)
    {
        // Drawn into this frame, but its render target is gone (target cap).
        impl.publishedTexture.store(0u, std::memory_order_release);
        return m_cpu.PresentSnapshot(m_cpu.EnqueuePresentSnapshot(), request);
    }
    impl.presentedFbp = displayFbp;
    // PS2X_GS_GPU_TRACE=<frame>: 30 presents of "what did we draw, what did we
    // show", to see whether successive presents publish different pictures.
    {
        static const uint64_t traceFrom = [] {
            const char *p = std::getenv("PS2X_GS_GPU_TRACE");
            return p ? std::strtoull(p, nullptr, 10) : ~0ull;
        }();
        if (impl.frames >= traceFrom && impl.frames < traceFrom + 30u)
        {
            std::map<uint32_t, size_t> drawnInto;
            for (const Impl::DrawCall &c : calls)
                drawnInto[c.fbp] += c.count;
            std::string into;
            for (const auto &entry : drawnInto)
            {
                char buf[48];
                std::snprintf(buf, sizeof(buf), "0x%x:%zuv ", entry.first, entry.second);
                into += buf;
            }
            // Also the scissor extents per buffer, and the raw DISPFB, so the
            // buffer geometry is fact rather than inference.
            std::map<uint32_t, int> lowY, highY;
            for (const Impl::DrawCall &c : calls)
            {
                if (c.clear)
                    continue;
                auto lo = lowY.find(c.fbp);
                lowY[c.fbp] = lo == lowY.end() ? c.sy : std::min<int>(lo->second, c.sy);
                auto hi = highY.find(c.fbp);
                highY[c.fbp] = hi == highY.end() ? c.sy + c.sh : std::max<int>(hi->second, c.sy + c.sh);
            }
            std::string scissors;
            for (const auto &entry : lowY)
            {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "0x%x y%d..%d ", entry.first, entry.second, highY[entry.first]);
                scissors += buf;
            }
            std::string ofys;
            for (const auto &entry : impl.traceOfy)
            {
                char buf[48];
                std::snprintf(buf, sizeof(buf), "%.4f:%zu ", entry.first / 16.0, entry.second);
                ofys += buf;
            }
            std::fprintf(stderr,
                         "[gs:trace] present %llu show fbp=0x%x dispfb=%llx calls=%zu drew {%s} scissor {%s} "
                         "ofy {%s} vy %.4f..%.4f\n",
                         (unsigned long long)impl.frames, displayFbp, (unsigned long long)dispfb, calls.size(),
                         into.c_str(), scissors.c_str(), ofys.c_str(), impl.traceMinY, impl.traceMaxY);
        }
        impl.traceOfy.clear();
        impl.traceAnyY = false;
    }

    // DISPLAY1/2: DX/DY (bits 0..22), MAGH/MAGV, DW/DH give the visible size.
    const uint64_t display = (request.display2 != 0u) ? request.display2 : request.display1;
    const uint32_t magh = static_cast<uint32_t>((display >> 23) & 0xFu) + 1u;
    const uint32_t magv = static_cast<uint32_t>((display >> 27) & 0x3u) + 1u;
    // In PS2 pixels; the render target holds `scale` times as many.
    uint32_t outWidth = (static_cast<uint32_t>((display >> 32) & 0xFFFu) + 1u) / magh;
    uint32_t outHeight = (static_cast<uint32_t>((display >> 44) & 0x7FFu) + 1u) / magv;
    if (outWidth == 0u || outWidth > Impl::kNativeWidth)
        outWidth = Impl::kNativeWidth;
    if (outHeight == 0u || outHeight > Impl::kNativeHeight)
        outHeight = Impl::kNativeHeight;
    const auto passHeight = impl.passHeights.find(displayFbp);
    const uint32_t sourceHeight = wotm_capture::presentationSourceHeight(outHeight,
        passHeight != impl.passHeights.end() ? passHeight->second : Impl::kNativeHeight, impl.height);

    PresentationFrame frame{};
    frame.width = outWidth;
    frame.height = outHeight;
    frame.displayFbp = impl.frameFbp;
    frame.sourceFbp = impl.frameFbp;

    if (impl.direct)
    {
        // Hand the render target itself to the host: no readback, and the image
        // stays at whatever resolution the target has.
        impl.gl.BindFramebuffer(kFramebuffer, target->fbo);

        // PS2X_GS_SHOT=<file.png>,<frame>[,<count>] reads back that frame, or
        // that many consecutive frames (named file.000.png, file.001.png, ...)
        // so successive presented frames can be compared for flicker.
        static const uint64_t shotFrame = [] {
            const char *spec = std::getenv("PS2X_GS_SHOT");
            if (!spec)
                return ~0ull;
            const char *comma = std::strchr(spec, ',');
            return comma ? std::strtoull(comma + 1, nullptr, 10) : 600ull;
        }();
        static const uint64_t shotCount = [] {
            const char *spec = std::getenv("PS2X_GS_SHOT");
            const char *comma = spec ? std::strchr(spec, ',') : nullptr;
            const char *second = comma ? std::strchr(comma + 1, ',') : nullptr;
            return second ? std::max<uint64_t>(std::strtoull(second + 1, nullptr, 10), 1ull) : 1ull;
        }();
        static const uint64_t shotStep = [] {
            const char *p = std::getenv("PS2X_GS_SHOT_STEP");
            return p ? std::max<uint64_t>(std::strtoull(p, nullptr, 10), 1ull) : 1ull;
        }();
        if (impl.frames >= shotFrame && (impl.frames - shotFrame) / shotStep < shotCount &&
            (impl.frames - shotFrame) % shotStep == 0u)
        {
            impl.readback.assign(static_cast<size_t>(impl.width) * impl.height * 4u, 0u);
            glReadPixels(0, 0, static_cast<GLsizei>(impl.width), static_cast<GLsizei>(impl.height), GL_RGBA,
                         GL_UNSIGNED_BYTE, impl.readback.data());
            const uint32_t shotWidth = outWidth * impl.scale;
            const uint32_t shotHeight = outHeight * impl.scale;
            std::vector<uint8_t> topDown(static_cast<size_t>(shotWidth) * shotHeight * 4u, 255u);
            for (uint32_t y = 0; y < shotHeight && y < impl.height; ++y)
                std::memcpy(topDown.data() + static_cast<size_t>(y) * shotWidth * 4u,
                            impl.readback.data() + static_cast<size_t>(impl.height - 1u - uint64_t(y) * sourceHeight / shotHeight) * impl.width * 4u,
                            static_cast<size_t>(shotWidth) * 4u);
            for (size_t i = 3; i < topDown.size(); i += 4)
                topDown[i] = 255u;
            if (shotCount == 1u)
            {
                ps2xGsSaveFrame(topDown.data(), shotWidth, shotHeight, shotFrame);
            }
            else
            {
                const char *spec = std::getenv("PS2X_GS_SHOT");
                const char *comma = spec ? std::strchr(spec, ',') : nullptr;
                char path[512];
                const int stem = comma ? static_cast<int>(comma - spec) : 0;
                std::snprintf(path, sizeof(path), "%.*s.%03llu.png", stem, spec ? spec : "",
                              (unsigned long long)(impl.frames - shotFrame));
                ps2xGsSaveImage(path, topDown.data(), shotWidth, shotHeight);
            }

            // Diagnostic exact-frame reference.  MarkPresentBoundary queued the
            // CPU snapshot while the GS state lock still separated this frame
            // from the next, so this is a real A/B comparison rather than two
            // runs whose animation and input timing merely look similar.
            if (gpuReferenceEnabled() && impl.referencePresentToken != 0u)
            {
                PresentationFrame reference = m_cpu.PresentSnapshot(impl.referencePresentToken, request);
                impl.referencePresentToken = 0u;
                if (!reference.pixels.empty() && reference.width != 0u && reference.height != 0u)
                {
                    const char *spec = std::getenv("PS2X_GS_SHOT");
                    const char *comma = spec ? std::strchr(spec, ',') : nullptr;
                    char path[512];
                    const int stem = comma ? static_cast<int>(comma - spec) : 0;
                    std::snprintf(path, sizeof(path), "%.*s.cpu.%03llu.png", stem, spec ? spec : "",
                                  (unsigned long long)(impl.frames - shotFrame));
                    ps2xGsSaveImage(path, reference.pixels.data(), reference.width, reference.height);
                }
            }
        }
        glFlush();
        impl.publishedTexture.store(target->color, std::memory_order_release);
        impl.publishedWidth.store(outWidth, std::memory_order_relaxed);
        impl.publishedHeight.store(outHeight, std::memory_order_relaxed);
        impl.publishedSourceWidth.store(outWidth * impl.scale, std::memory_order_relaxed);
        impl.publishedSourceHeight.store(sourceHeight, std::memory_order_relaxed);
        impl.publishedTargetWidth.store(impl.width, std::memory_order_relaxed);
        impl.publishedTargetHeight.store(impl.height, std::memory_order_relaxed);
        if ((++impl.frames % 300u) == 0u)
            std::fprintf(stderr,
                         "[gs:gl] frames=%llu prims=%llu draws=%llu direct %ux%u fbp=0x%x of %zu | tex live=%zu "
                         "decoded=%llu (mipped %llu) uploaded=%llu invalidated=%llu | cpu-presented %llu rtt %llu blits %llu\n",
                         (unsigned long long)impl.frames, (unsigned long long)impl.primitives,
                         (unsigned long long)impl.drawCalls, outWidth, outHeight, displayFbp, impl.targets.size(),
                         impl.textures.size(), (unsigned long long)impl.textureDecodes,
                         (unsigned long long)impl.textureMipmapped, (unsigned long long)impl.textureUploads,
                         (unsigned long long)impl.textureInvalidations, (unsigned long long)impl.presentsFromCpu,
                         (unsigned long long)impl.renderTargetReads, (unsigned long long)impl.blits);
        return frame; // no pixels: the host draws the texture instead
    }

    // Fallback: read the target back so the existing host upload path works.
    impl.readback.resize(static_cast<size_t>(impl.width) * impl.height * 4u);
    impl.gl.BindFramebuffer(kFramebuffer, target->fbo);
    glReadPixels(0, 0, static_cast<GLsizei>(impl.width), static_cast<GLsizei>(impl.height), GL_RGBA, GL_UNSIGNED_BYTE,
                 impl.readback.data());
    impl.readbackBytes += impl.readback.size();
    // The host upload path only understands PS2-sized frames, so a scaled
    // target is point sampled back down to them.
    frame.pixels.resize(static_cast<size_t>(outWidth) * outHeight * 4u);
    for (uint32_t y = 0; y < outHeight; ++y)
    {
        const uint32_t sourceRow = std::min<uint32_t>(uint64_t(y) * sourceHeight / outHeight, impl.height - 1u);
        // glReadPixels returns bottom-up rows.
        const uint8_t *src = impl.readback.data() + static_cast<size_t>(impl.height - 1u - sourceRow) * impl.width * 4u;
        uint8_t *dst = frame.pixels.data() + static_cast<size_t>(y) * outWidth * 4u;
        if (impl.scale == 1u)
            std::memcpy(dst, src, static_cast<size_t>(outWidth) * 4u);
        else
            for (uint32_t x = 0; x < outWidth; ++x)
                std::memcpy(dst + static_cast<size_t>(x) * 4u, src + static_cast<size_t>(x) * impl.scale * 4u, 4u);
        // The display has no alpha channel (this game's FRAME is PSMCT24), and
        // the render target's alpha is a blend operand, not coverage: hand the
        // host an opaque frame or every low-alpha pixel shows as transparent.
        for (uint32_t x = 0; x < outWidth; ++x)
            dst[x * 4u + 3u] = 255u;
    }
    if ((++impl.frames % 300u) == 0u)
        std::fprintf(stderr,
                     "[gs:gl] frames=%llu prims=%llu draws=%llu readback=%llu MiB %ux%u | tex live=%zu decoded=%llu "
                     "uploaded=%llu invalidated=%llu\n",
                     (unsigned long long)impl.frames, (unsigned long long)impl.primitives,
                     (unsigned long long)impl.drawCalls, (unsigned long long)(impl.readbackBytes / 1048576u), outWidth,
                     outHeight, impl.textures.size(), (unsigned long long)impl.textureDecodes,
                     (unsigned long long)impl.textureUploads, (unsigned long long)impl.textureInvalidations);
    return frame;

}

bool GSGlBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    const bool handled = m_cpu.ClearFramebuffer(context, rgba);
    if (!handled || m_impl->failed)
        return handled;
    // Queue it in draw order: a clear that jumped ahead of the frame's draws
    // would wipe them.
    Impl::DrawCall call{};
    call.clear = true;
    call.fbp = context.frame.fbp;
    call.sx = static_cast<int>(context.scissor.x0);
    call.sy = static_cast<int>(context.scissor.y0);
    call.sw = static_cast<int>(context.scissor.x1) - call.sx + 1;
    call.sh = static_cast<int>(context.scissor.y1) - call.sy + 1;
    call.clearColor[0] = static_cast<float>(rgba & 0xFFu) / 255.0f;
    call.clearColor[1] = static_cast<float>((rgba >> 8) & 0xFFu) / 255.0f;
    call.clearColor[2] = static_cast<float>((rgba >> 16) & 0xFFu) / 255.0f;
    call.clearColor[3] = static_cast<float>((rgba >> 24) & 0xFFu) / 128.0f;
    std::lock_guard<std::mutex> lock(m_impl->recordMutex);
    m_impl->preparedValid = false;
    Impl::KnownTarget &known = m_impl->knownTargets[context.frame.fbp];
    known.blocks = GSInternal::framePageBaseToBlock(context.frame.fbp);
    known.psm = context.frame.psm;
    known.fbw = context.frame.fbw;
    m_impl->calls.push_back(call);
    return handled;
}

uint32_t GSGlBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) { return m_cpu.ConsumeLocalToHostBytes(dst, maxBytes); }
uint32_t GSGlBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const { return m_cpu.ReadVram(psm, base, bw, x, y); }
void GSGlBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) { m_cpu.WriteVram(psm, base, bw, x, y, value); }
void GSGlBackend::SnapshotVram(std::vector<uint8_t> &out) const { m_cpu.SnapshotVram(out); }
GSTransferSnapshot GSGlBackend::GetTransferSnapshot() const { return m_cpu.GetTransferSnapshot(); }

#else  // !_WIN32: no GL backend yet, everything falls through to the CPU rasterizer.

struct GSGlBackend::Impl
{
};
bool ps2xGsGpuPresentTexture(Ps2xGpuFrame *) { return false; }
GSGlBackend::GSGlBackend() : m_impl(std::make_unique<Impl>()) {Impl::s_activeImpl=m_impl.get();}
GSGlBackend::~GSGlBackend() = default;
bool GSGlBackend::Available() const { return false; }
void GSGlBackend::Initialize(uint8_t *vram, uint32_t vramSize) { m_cpu.Initialize(vram, vramSize); }
void GSGlBackend::Reset() { m_cpu.Reset(); }
void GSGlBackend::Submit(const GSPrimitiveBatch &batch) { m_cpu.Submit(batch); }
void GSGlBackend::BeginTransfer(const GSTransferCommand &command) { m_cpu.BeginTransfer(command); }
void GSGlBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes) { m_cpu.UploadImage(data, sizeBytes); }
void GSGlBackend::Flush() { m_cpu.Flush(); }
void GSGlBackend::TextureFlush() { m_cpu.TextureFlush(); }
void GSGlBackend::Sync(GSSyncReason reason) { m_cpu.Sync(reason); }
PresentationFrame GSGlBackend::Present(const GSPresentationRequest &request) { return m_cpu.Present(request); }
bool GSGlBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba) { return m_cpu.ClearFramebuffer(context, rgba); }
uint32_t GSGlBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) { return m_cpu.ConsumeLocalToHostBytes(dst, maxBytes); }
uint32_t GSGlBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const { return m_cpu.ReadVram(psm, base, bw, x, y); }
void GSGlBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) { m_cpu.WriteVram(psm, base, bw, x, y, value); }
void GSGlBackend::SnapshotVram(std::vector<uint8_t> &out) const { m_cpu.SnapshotVram(out); }
GSTransferSnapshot GSGlBackend::GetTransferSnapshot() const { return m_cpu.GetTransferSnapshot(); }

#endif




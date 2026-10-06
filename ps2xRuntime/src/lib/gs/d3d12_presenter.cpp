// Experimental D3D12 presentation foundation. Not selected by the game yet.
// All methods are called on the owning window/render thread. No guest headers.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdio>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <vector>
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "user32.lib")

namespace wotm::d3d12 {
using Microsoft::WRL::ComPtr;
static void check(HRESULT hr, const char* operation) {
    if (FAILED(hr)) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s failed (0x%08lx)", operation, (unsigned long)hr);
        throw std::runtime_error(message);
    }
}
class Presenter {
    static constexpr UINT count = 2;
    struct Frame {
        ComPtr<ID3D12Resource> color;
        ComPtr<ID3D12CommandAllocator> allocator;
        UINT64 fence = 0;
    } frames[count];
    ComPtr<IDXGIFactory6> factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain3> swap;
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    UINT64 nextFence = 0;
    UINT stride = 0, width = 0, height = 0;
    bool tearing = false;
    void wait(UINT64 value) {
        UINT64 completed = fence->GetCompletedValue();
        if (completed == UINT64_MAX) check(device->GetDeviceRemovedReason(), "Device removed");
        if (completed < value) {
            check(fence->SetEventOnCompletion(value, event), "SetEventOnCompletion");
            DWORD result = WaitForSingleObject(event, 10000);
            if (result != WAIT_OBJECT_0) throw std::runtime_error("GPU fence wait failed or timed out");
        }
    }
    UINT64 signal() {
        const UINT64 value = ++nextFence;
        check(queue->Signal(fence.Get(), value), "Signal");
        return value;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE rtv(UINT i) const {
        auto handle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += SIZE_T(i) * stride;
        return handle;
    }
    void targets() {
        for (UINT i = 0; i < count; ++i) {
            check(swap->GetBuffer(i, IID_PPV_ARGS(&frames[i].color)), "GetBuffer");
            device->CreateRenderTargetView(frames[i].color.Get(), nullptr, rtv(i));
        }
    }
    void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        commands->ResourceBarrier(1, &barrier);
    }
public:
    Presenter() = default;
    Presenter(const Presenter&) = delete;
    Presenter& operator=(const Presenter&) = delete;
    ~Presenter() {
        try { if (queue && fence && event) wait(signal()); } catch (...) {}
        if (event) CloseHandle(event);
    }
    void initialize(HWND window, UINT w, UINT h, bool debug) {
        if (device || !window || !w || !h) throw std::runtime_error("Invalid presenter initialization");
        if (debug) {
            ComPtr<ID3D12Debug> layer;
            check(D3D12GetDebugInterface(IID_PPV_ARGS(&layer)), "D3D12 debug layer (install Graphics Tools)");
            layer->EnableDebugLayer();
        }
        check(CreateDXGIFactory2(debug ? DXGI_CREATE_FACTORY_DEBUG : 0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
        for (UINT i = 0;; ++i) {
            ComPtr<IDXGIAdapter1> adapter;
            HRESULT hr = factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
            if (hr == DXGI_ERROR_NOT_FOUND) break;
            check(hr, "EnumAdapterByGpuPreference");
            DXGI_ADAPTER_DESC1 desc{}; check(adapter->GetDesc1(&desc), "GetDesc1");
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
                std::printf("D3D12 adapter: %ls\n", desc.Description); break;
            }
        }
        if (!device) throw std::runtime_error("No hardware D3D12 adapter available");
        D3D12_COMMAND_QUEUE_DESC q{}; q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) throw std::runtime_error("CreateEvent failed");
        BOOL allowed = FALSE;
        tearing = SUCCEEDED(factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowed, sizeof(allowed))) && allowed;
        DXGI_SWAP_CHAIN_DESC1 desc{};
        desc.Width = w; desc.Height = h; desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1; desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.BufferCount = count; desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc.Flags = tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
        ComPtr<IDXGISwapChain1> chain;
        check(factory->CreateSwapChainForHwnd(queue.Get(), window, &desc, nullptr, nullptr, &chain), "CreateSwapChainForHwnd");
        check(chain.As(&swap), "Query swap chain");
        check(factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER), "MakeWindowAssociation");
        D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV; heap.NumDescriptors = count;
        check(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&rtvHeap)), "CreateDescriptorHeap");
        stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        for (auto& frame : frames) check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&frame.allocator)), "CreateCommandAllocator");
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, frames[0].allocator.Get(), nullptr, IID_PPV_ARGS(&commands)), "CreateCommandList");
        check(commands->Close(), "Initial Close");
        width = w; height = h; targets();
    }
    void validateDebugMessages() {
        wait(signal());
        ComPtr<ID3D12InfoQueue> info;
        if (FAILED(device.As(&info))) return;
        bool errors = false;
        for (UINT64 i = 0; i < info->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
            SIZE_T bytes = 0; check(info->GetMessage(i, nullptr, &bytes), "Debug message size");
            std::vector<uint8_t> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            check(info->GetMessage(i, message, &bytes), "Debug message");
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
                std::fprintf(stderr, "D3D12 validation: %s\n", message->pDescription);
                errors = true;
            }
        }
        if (errors) throw std::runtime_error("D3D12 debug layer reported warnings/errors");
    }
    // A zero client size means minimized: retain resources, do not ResizeBuffers.
    void resize(UINT w, UINT h) {
        if (!w || !h || (w == width && h == height)) return;
        wait(signal());
        for (auto& frame : frames) { frame.color.Reset(); frame.fence = 0; }
        check(swap->ResizeBuffers(count, w, h, DXGI_FORMAT_R8G8B8A8_UNORM,
              tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0), "ResizeBuffers");
        width = w; height = h; targets();
    }
    // Readback is strictly for validation. Normal presentation stays on the GPU.
    void clearAndPresent(const float color[4], bool vsync, bool verify = false) {
        const UINT index = swap->GetCurrentBackBufferIndex();
        auto& frame = frames[index]; wait(frame.fence);
        check(frame.allocator->Reset(), "Allocator Reset");
        check(commands->Reset(frame.allocator.Get(), nullptr), "Command list Reset");
        transition(frame.color.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        commands->ClearRenderTargetView(rtv(index), color, 0, nullptr);
        ComPtr<ID3D12Resource> readback;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        if (verify) {
            auto desc = frame.color->GetDesc(); UINT64 bytes = 0;
            device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
            D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC buffer{}; buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width = bytes; buffer.Height = 1; buffer.DepthOrArraySize = 1;
            buffer.MipLevels = 1; buffer.SampleDesc.Count = 1; buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                  D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)), "Create readback");
            transition(frame.color.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = readback.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = footprint;
            D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = frame.color.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            commands->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        transition(frame.color.Get(), verify ? D3D12_RESOURCE_STATE_COPY_SOURCE : D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        check(commands->Close(), "Close");
        ID3D12CommandList* lists[] = {commands.Get()}; queue->ExecuteCommandLists(1, lists);
        frame.fence = signal();
        check(swap->Present(vsync ? 1 : 0, !vsync && tearing ? DXGI_PRESENT_ALLOW_TEARING : 0), "Present");
        if (verify) {
            wait(frame.fence);
            uint8_t* data = nullptr; const SIZE_T size = SIZE_T(footprint.Offset) + SIZE_T(footprint.Footprint.RowPitch) * (height - 1) + SIZE_T(width) * 4;
            D3D12_RANGE range{0, size}; check(readback->Map(0, &range, reinterpret_cast<void**>(&data)), "Readback Map");
            bool matches = true;
            for (UINT y = 0; y < height; ++y) for (UINT x = 0; x < width; ++x) for (UINT c = 0; c < 4; ++c) {
                int actual = data[footprint.Offset + SIZE_T(y) * footprint.Footprint.RowPitch + x * 4 + c];
                int expected = int(color[c] * 255.0f + 0.5f);
                if (actual < expected - 1 || actual > expected + 1) matches = false;
            }
            D3D12_RANGE written{0, 0}; readback->Unmap(0, &written);
            if (!matches) throw std::runtime_error("GPU clear readback mismatch");
        }
    }
};
} // namespace wotm::d3d12

#ifdef WOTM_D3D12_SMOKE
int main(int argc, char** argv) {
    HWND window = nullptr;
    try {
        WNDCLASSW cls{}; cls.lpfnWndProc = DefWindowProcW;
        cls.hInstance = GetModuleHandleW(nullptr); cls.lpszClassName = L"WotmD3D12Smoke";
        if (!RegisterClassW(&cls)) throw std::runtime_error("RegisterClass failed");
        window = CreateWindowW(cls.lpszClassName, L"War of the Monsters - D3D12 validation", WS_OVERLAPPEDWINDOW,
                              100, 100, 640, 480, nullptr, nullptr, cls.hInstance, nullptr);
        if (!window) throw std::runtime_error("CreateWindow failed");
        {
            wotm::d3d12::Presenter renderer;
            renderer.initialize(window, 640, 480, argc > 1 && std::strcmp(argv[1], "--debug") == 0);
            for (UINT frame = 0; frame < 120; ++frame) {
                MSG msg{}; while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
                if (frame == 30) renderer.resize(0, 0);
                if (frame == 40) renderer.resize(321, 241);
                if (frame == 80) renderer.resize(1280, 720);
                float color[] = {float(frame % 3 == 0), float(frame % 3 == 1), float(frame % 3 == 2), 1.0f};
                renderer.clearAndPresent(color, frame % 2 == 0, frame % 10 == 0 || frame == 119);
            }
            renderer.validateDebugMessages();
        }
        DestroyWindow(window); window = nullptr;
        std::puts("PASS: 120 presents; 13 full-image readbacks; zero-size and odd-size resize; VSync on/off; clean shutdown");
        return 0;
    } catch (const std::exception& error) {
        if (window) DestroyWindow(window);
        std::fprintf(stderr, "D3D12 validation failed: %s\n", error.what()); return 1;
    }
}
#endif

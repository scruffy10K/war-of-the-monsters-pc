// Private temporal-rendering diagnostic contract; no generated runner dependencies.
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <atomic>
#include <map>
namespace MotionProvenance {
uint64_t resetSerial();
void resetHistory();
struct Object {
    uint64_t epoch=resetSerial();
    uint64_t frame=0;
    uint32_t node=0, view=0, slot=0, source=0;
    bool candidate=false;
    std::shared_ptr<std::atomic<bool>> valid;
    std::array<float,64> transforms{}; // current/previous object, current/previous camera
};
struct Span {
    uint32_t begin=0,end=0,tag=0;
    std::shared_ptr<const Object> object;
    uint64_t signature=0;
    uint32_t rootTag=0; // Outer DMA CALL, captured before CALL/RET changes ASP.
};
bool vectorsEnabled();
bool enabled();
void registerPacket(uint32_t begin,uint32_t end,const Object& object);
std::shared_ptr<const Object> lookup(uint32_t tag);
void publish(const uint8_t* data,std::vector<Span>&& spans);
std::vector<Span> take(const uint8_t* data);
struct Run {
    Run(const std::vector<Span>& spans,uint32_t offset,uint32_t opcode,uint32_t pc);
    ~Run();
};
void issue(const void* interpreter);
void kick(const void* interpreter,uint64_t serial,uint32_t bytes,uint32_t pc,uint64_t image=0);
struct Point {float x,y,z,q;};
struct Mesh {
    uint64_t epoch=resetSerial();
    bool shadow=false;
    uint64_t frame=0;
    std::vector<std::pair<std::array<uint32_t,4>,std::array<float,4>>> previous;
    std::shared_ptr<std::atomic<bool>> unique,sourceValid,previousValid;
    bool usable() const {return epoch==resetSerial() && unique && unique->load() && sourceValid && sourceValid->load() && previousValid && previousValid->load();}
};
std::shared_ptr<Mesh> prepareMesh(const std::vector<Point>& points,const uint8_t* adc,uint64_t tag);
std::shared_ptr<Mesh> activeMesh();
struct MeshScope {
    std::shared_ptr<Mesh> saved;
    explicit MeshScope(std::shared_ptr<Mesh> mesh);
    ~MeshScope();
};
std::array<float,4> previousVertex(const Mesh& mesh,Point point);
}

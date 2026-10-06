#pragma once
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <utility>

// Caller serializes access. Buffers in this pool have no live elements and no
// frame identity: immutable queued frames and GPU uploads own separate storage.
template<class VertexVector,class CallVector> struct GsFrameStoragePool {
    std::array<VertexVector,2> vertices;
    std::array<CallVector,2> calls;
    uint64_t acquires=0,retains=0;

    template<class Vector> static bool acquireOne(Vector& out,std::array<Vector,2>& pool) {
        if(!out.empty())std::abort();
        const size_t best=pool[1].capacity()>pool[0].capacity()?1u:0u;
        if(pool[best].capacity()<=out.capacity())return false;
        out.swap(pool[best]);return true;
    }
    template<class Vector> static bool recycleOne(Vector& out,std::array<Vector,2>& pool) {
        out.clear();
        const size_t smallest=pool[1].capacity()<pool[0].capacity()?1u:0u;
        if(out.capacity()<=pool[smallest].capacity())return false;
        out.swap(pool[smallest]);return true;
    }
    void acquire(VertexVector& v,CallVector& c) {
        acquires+=acquireOne(v,vertices);acquires+=acquireOne(c,calls);
    }
    void recycle(VertexVector& v,CallVector& c) {
        retains+=recycleOne(v,vertices);retains+=recycleOne(c,calls);
    }
    size_t vertexBytes() const {
        return (vertices[0].capacity()+vertices[1].capacity())*sizeof(typename VertexVector::value_type);
    }
};

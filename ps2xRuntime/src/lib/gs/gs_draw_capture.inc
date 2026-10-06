// Private diagnostic interchange, included only by the GL recorder and D3D12
// replay. Versioned local captures contain game assets: never distribute them.
#include <cstdint>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <memory>
#include <map>
#include <algorithm>
#include <cstring>
namespace wotm_capture {
struct Vertex {
  float x, y, z, r, g, b, a, s, t, q, fog;
};
static_assert(sizeof(Vertex) == 44);
struct Draw {
  uint32_t first = 0, count = 0, fbp = 0, zbp = 0, texture = ~0u,
           textureFbp = ~0u;
  uint64_t alpha = 0, test = 0;
  uint32_t zmsk = 0, abe = 0, clear = 0, fbmsk = 0;
  int32_t sx = 0, sy = 0, sw = 0, sh = 0;
  uint32_t tcc = 0, tfx = 0, iip = 0, alphaTest = 0, atst = 0, afail = 0,
           pabe = 0, dither = 0, fog = 0;
  float aref = 0, region[4]{}, fogColor[3]{}, dimx[16]{}, clearColor[4]{},
        lodBias = 0;
  int32_t wrapU = 0, wrapV = 0;
  uint32_t repeatU = 0, repeatV = 0, mmin = 0, mmag = 0, transient = 0;
};
struct Image {
  uint32_t id = 0, width = 0, height = 0;
  std::vector<uint8_t> bytes;
  // Live packets borrow immutable pixels until synchronous GPU submission completes.
  const uint8_t* borrowed = nullptr;
  const uint8_t* data() const { return borrowed ? borrowed : bytes.data(); }
};
struct Texture {
  std::vector<Image> levels;
  std::shared_ptr<const void> owner; // Live cache identity/lifetime; never serialized.
};
struct TargetFormat {uint32_t fbp=0,psm=~0u;};
struct TargetHeight { uint32_t fbp = 0, height = 0; };

// Passes are normalized vertically into the host target. DISPLAY dimensions
// describe the output aspect, not an unscaled crop of that normalized texture.
// Keep intentional partial displays, but do not crop 368-line movies after
// expanding them to 448 lines (or crop line-doubled 224-line gameplay fields).
inline uint32_t presentationSourceHeight(uint32_t displayHeight, uint32_t passHeight, uint32_t targetHeight) {
  if (!passHeight) return targetHeight;
  return uint32_t(std::min<uint64_t>(targetHeight,
      (uint64_t(displayHeight) * targetHeight + passHeight - 1u) / passHeight));
}

// A cropped scissor limits writes; it does not resize the render target.
// Preserve the last full-origin pass extent during partial updates. A later
// full-origin pass can shrink again (e.g. 448-line movie to 224-line field).
template<class Draws>
std::map<uint32_t, uint32_t> resolvePassHeights(
    const Draws& draws, std::map<uint32_t, uint32_t>& previous) {
  std::map<uint32_t, uint32_t> extents, full;
  for (const auto& d : draws) {
    const uint32_t end = uint32_t(std::max(0, d.sy + d.sh));
    extents[d.fbp] = std::max(extents[d.fbp], end);
    if (d.sy == 0 && d.sh > 0) full[d.fbp] = std::max(full[d.fbp], end);
  }
  for (auto& e : extents) {
    auto f = full.find(e.first), old = previous.find(e.first);
    if (f != full.end()) e.second = std::max(e.second, f->second);
    else if (old != previous.end()) e.second = std::max(e.second, old->second);
    e.second = std::max(e.second, 1u);
    previous[e.first] = e.second;
  }
  return extents;
}

struct Frame {
  uint32_t width = 0, height = 0, scale = 1, display = 0;
  std::vector<TargetFormat> formats;
  std::vector<TargetHeight> heights;
  std::vector<Vertex> vertices;
  // Live submission is synchronous: borrow its immutable vertex storage until
  // upload completes. Captures and atlas UV edits materialize their own copy.
  const Vertex* borrowedVertices = nullptr;
  size_t borrowedVertexCount = 0;
  const Vertex* vertexData() const { return borrowedVertices ? borrowedVertices : vertices.data(); }
  size_t vertexCount() const { return borrowedVertices ? borrowedVertexCount : vertices.size(); }
  void ownVertices() {
    if (!borrowedVertices) return;
    vertices.assign(borrowedVertices,borrowedVertices+borrowedVertexCount);
    borrowedVertices=nullptr;borrowedVertexCount=0;
  }
  // Live experimental motion stream: previous XY times W ratio, ratio, validity.
  // v4+ serializes this stream; older captures contain no motion history.
  std::vector<std::array<float,4>> motion;
  std::vector<Draw> draws;
  // v5: source engine generation per draw; zero denotes an untracked overlay.
  // Empty (legacy) streams keep conservative per-packet invalidation.
  std::vector<uint64_t> motionGenerations;
  uint64_t motionResetSerial=0;
  std::vector<uint8_t> motionKinds; // v6: 1=verified blocker source, 0=ordinary/unknown
  std::vector<Texture> textures;
  std::vector<Image> colors, depths, expected;
};
class File {
  FILE *f = nullptr;
  bool reading;
  size_t remaining = 512u * 1024u * 1024u;

public:
  File(const char *path, bool read) : reading(read) {
    f = std::fopen(path, read ? "rb" : "wb");
    if (!f)
      throw std::runtime_error("Capture open failed");
  }
  ~File() {
    if (f)
      std::fclose(f);
  }
  void bytes(void *data, size_t n) {
    if (n > remaining)
      throw std::runtime_error("Capture exceeds size limit");
    remaining -= n;
    if ((reading ? std::fread(data, 1, n, f) : std::fwrite(data, 1, n, f)) != n)
      throw std::runtime_error("Capture I/O failed");
  }
  template <class T> void scalar(T &v) {
    static_assert(std::is_trivially_copyable_v<T>);
    bytes(&v, sizeof(v));
  }
  template <class T> void array(std::vector<T> &v) {
    static_assert(std::is_trivially_copyable_v<T>);
    uint32_t n = uint32_t(v.size());
    scalar(n);
    if (n > remaining / sizeof(T))
      throw std::runtime_error("Capture array too large");
    if (reading)
      v.resize(n);
    bytes(v.data(), size_t(n) * sizeof(T));
  }
  template <class T, class F> void list(std::vector<T> &v, F fn) {
    uint32_t n = uint32_t(v.size());
    scalar(n);
    if (n > 65536)
      throw std::runtime_error("Capture list too large");
    if (reading)
      v.resize(n);
    for (auto &x : v)
      fn(x);
  }
  void image(Image &x) {
    scalar(x.id);
    scalar(x.width);
    scalar(x.height);
    array(x.bytes);
    if (!x.width || !x.height || x.width > 8192 || x.height > 8192 ||
        x.bytes.size() != uint64_t(x.width) * x.height * 4)
      throw std::runtime_error("Invalid capture image");
  }
  void frame(Frame &x) {
    if (!reading) x.ownVertices();
    else { x.borrowedVertices=nullptr;x.borrowedVertexCount=0; }
    uint32_t magic = 0x31524457, version = 6;
    scalar(magic);
    scalar(version);
    if (magic != 0x31524457 || (version < 1 || version > 6))
      throw std::runtime_error("Wrong capture format");
    scalar(x.width);
    scalar(x.height);
    scalar(x.scale);
    scalar(x.display);
    if(version>=2)array(x.formats);
    if(version>=3)array(x.heights);
    if (!x.width || !x.height || x.width > 8192 || x.height > 8192 ||
        x.scale < 1 || x.scale > 8)
      throw std::runtime_error("Invalid capture dimensions");
    array(x.vertices);
    if(version>=4)array(x.motion);
    if(!x.motion.empty() && x.motion.size()!=x.vertices.size())throw std::runtime_error("Capture motion size mismatch");
    array(x.draws);
    if(version>=5)array(x.motionGenerations);
    if(version>=6){scalar(x.motionResetSerial);array(x.motionKinds);}
    if(!x.motionKinds.empty() && x.motionKinds.size()!=x.draws.size())throw std::runtime_error("Capture motion kind count mismatch");
    if(!x.motionGenerations.empty() && x.motionGenerations.size()!=x.draws.size())throw std::runtime_error("Capture motion generation count mismatch");
    list(x.textures, [&](Texture &t) {
      list(t.levels, [&](Image &i) { image(i); });
      if (t.levels.empty() || t.levels.size() > 14)
        throw std::runtime_error("Invalid mip count");
    });
    list(x.colors, [&](Image &i) { image(i); });
    list(x.depths, [&](Image &i) { image(i); });
    list(x.expected, [&](Image &i) { image(i); });
    for (auto &d : x.draws)
      if (d.first > x.vertices.size() ||
          d.count > x.vertices.size() - d.first || d.count % 3 ||
          (d.texture != ~0u && d.texture >= x.textures.size()))
        throw std::runtime_error("Invalid capture draw");
  }
};
} // namespace wotm_capture

// Pack short-lived transfer tiles without changing draw order or sampling at
// tile edges. One replicated texel around each tile isolates bilinear filtering.
namespace wotm_capture {
inline void packTransferTiles(Frame& frame) {
  struct Placement {uint32_t texture=~0u,x=0,y=0,w=0,h=0;};
  std::vector<Placement> positions(frame.textures.size());
  std::vector<bool> eligible(frame.textures.size(),true), seen(frame.textures.size(),false);
  for(const auto& d:frame.draws) if(d.texture<eligible.size()) {
    seen[d.texture]=true;
    eligible[d.texture]=eligible[d.texture]&&d.transient&&!d.repeatU&&!d.repeatV&&d.wrapU<2&&d.wrapV<2;
  }
  std::vector<Texture> packed;
  uint32_t atlas=~0u,x=0,y=0,row=0;constexpr uint32_t edge=1024;
  for(uint32_t i=0;i<frame.textures.size();++i) {
    auto& input=frame.textures[i];
    if(!seen[i]||!eligible[i]||input.levels.size()!=1||input.levels[0].width+2>edge||input.levels[0].height+2>edge) {
      positions[i].texture=uint32_t(packed.size());packed.push_back(std::move(input));continue;
    }
    const auto& image=input.levels[0];uint32_t tw=image.width+2,th=image.height+2;
    if(x+tw>edge){x=0;y+=row;row=0;}
    if(atlas==~0u||y+th>edge){Texture t;t.levels.push_back({0,edge,edge,{}});t.levels[0].bytes.resize(size_t(edge)*edge*4);
      atlas=uint32_t(packed.size());packed.push_back(std::move(t));x=y=row=0;}
    positions[i]={atlas,x+1,y+1,image.width,image.height};auto& pixels=packed[atlas].levels[0].bytes;
    for(uint32_t py=0;py<th;++py)for(uint32_t px=0;px<tw;++px){
      uint32_t sx=px?std::min(px-1,image.width-1):0,sy=py?std::min(py-1,image.height-1):0;
      std::memcpy(pixels.data()+((size_t(y+py)*edge)+x+px)*4,image.data()+(size_t(sy)*image.width+sx)*4,4);
    }
    x+=tw;row=std::max(row,th);
  }
  for(auto& d:frame.draws)if(d.texture<positions.size()) {
    const auto& p=positions[d.texture];d.texture=p.texture;
    if(p.w)frame.ownVertices();
    if(p.w)for(size_t i=d.first;i<size_t(d.first)+d.count;++i){auto& v=frame.vertices.at(i);
      v.s=(v.s*p.w+p.x*v.q)/edge;v.t=(v.t*p.h+p.y*v.q)/edge;}
  }
  frame.textures=std::move(packed);
}
}

// Extracted disc reader. Compiled once in ps2_runtime.cpp. Index format and
// installer live in launcher/GameImage.cs; all integers are little endian.
#include "installed_disc_api.inc"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace wotm_disc {
namespace {
namespace fs = std::filesystem;
struct Range {
    uint64_t offset, length, fileOffset;
    fs::path file;
    bool asset;
};
struct CachedFile {
    std::ifstream stream;
    uint64_t used;
    CachedFile(const fs::path& path, uint64_t tick) : stream(path, std::ios::binary), used(tick) {}
};
class Disc {
    std::mutex mutex;
    std::unordered_map<fs::path, CachedFile> cache;
    uint64_t tick = 0;
public:
    fs::path index;
    uint64_t size = 0;
    std::vector<Range> ranges;
    explicit Disc(const fs::path& path) : index(fs::canonical(path)) {
        const auto indexSize = fs::file_size(index);
        if (indexSize < 24 || indexSize > 8*1024*1024)
            throw std::runtime_error("Invalid installed disc index size. Reinstall from your ISO.");
        std::ifstream input(index, std::ios::binary);
        std::vector<unsigned char> bytes(static_cast<size_t>(indexSize));
        if (!input.read(reinterpret_cast<char*>(bytes.data()), bytes.size()) ||
            std::memcmp(bytes.data(), "WOTMDI02", 8))
            throw std::runtime_error("Invalid installed disc index. Reinstall from your ISO.");
        size_t cursor = 8;
        auto number = [&](size_t count) {
            if (cursor > bytes.size() || count > bytes.size()-cursor)
                throw std::runtime_error("Truncated installed disc index.");
            uint64_t value = 0;
            for (size_t i=0; i<count; ++i) value |= uint64_t(bytes[cursor++]) << (8*i);
            return value;
        };
        size = number(8);
        const auto count = number(4), reserved = number(4);
        if (size < 32768 || size > (uint64_t(8)<<30) || size%2048 ||
            count == 0 || count > 100000 || reserved != 0)
            throw std::runtime_error("Unsupported installed disc layout.");
        const auto root = index.parent_path();
        uint64_t end = 0;
        for (uint64_t i=0; i<count; ++i) {
            Range range{};
            range.offset=number(8); range.length=number(8); range.fileOffset=number(8);
            const auto nameSize=number(4);
            if (nameSize == 0 || nameSize > 1024 || nameSize > bytes.size()-cursor)
                throw std::runtime_error("Invalid installed disc filename.");
            std::string name(reinterpret_cast<char*>(bytes.data()+cursor), static_cast<size_t>(nameSize));
            cursor += static_cast<size_t>(nameSize);
            range.asset = name.rfind("files/", 0) == 0;
            if ((!range.asset && name != "disc.meta") || name.back()=='/')
                throw std::runtime_error("Unsafe installed disc path.");
            size_t component=0;
            for (size_t n=0; n<=name.size(); ++n) {
                if (n==name.size() || name[n]=='/') {
                    const auto part=name.substr(component,n-component);
                    if (part.empty() || part=="." || part==".." || part.back()=='.' || part.back()==' ')
                        throw std::runtime_error("Unsafe installed disc path component.");
                    component=n+1;
                } else {
                    const unsigned char ch=name[n];
                    if (ch<32 || ch>126 || std::strchr("\\:*?\"<>|",ch))
                        throw std::runtime_error("Unsafe installed disc path character.");
                }
            }
            if (range.length==0 || range.offset<end || range.offset>size || range.length>size-range.offset)
                throw std::runtime_error("Overlapping or out-of-bounds installed disc range.");
            range.file=fs::canonical(root/fs::path(name));
            // Canonical component comparison also rejects symlinks/junctions
            // that point outside this installation, not just textual '..'.
            auto a=root.begin(), b=range.file.begin();
            for (; a!=root.end() && b!=range.file.end() && *a==*b; ++a,++b) {}
            if (a!=root.end() || b==range.file.end() || !fs::is_regular_file(range.file))
                throw std::runtime_error("Installed disc file escapes its installation.");
            const uint64_t fileSize=fs::file_size(range.file);
            if (range.fileOffset>fileSize || range.length>fileSize-range.fileOffset ||
                (range.asset && (range.fileOffset!=0 || range.length!=fileSize || range.offset%2048)))
                throw std::runtime_error("Installed game file is truncated or modified. Reinstall from your ISO.");
            end=range.offset+range.length;
            ranges.push_back(std::move(range));
        }
        if (cursor!=bytes.size()) throw std::runtime_error("Trailing data in installed disc index.");
    }
    bool read(uint64_t offset, void* data, size_t length, size_t& got) {
        got=0;
        if ((!data && length) || offset>size) return false;
        const size_t available=static_cast<size_t>(std::min<uint64_t>(length,size-offset));
        if (!available) return true;
        std::lock_guard<std::mutex> lock(mutex);
        auto* out=static_cast<unsigned char*>(data);
        const uint64_t stop=offset+available;
        auto it=std::lower_bound(ranges.begin(),ranges.end(),offset,
            [](const Range& r,uint64_t value) { return r.offset+r.length<=value; });
        while (offset<stop) {
            if (it==ranges.end() || offset<it->offset) {
                const auto end=it==ranges.end()?stop:std::min(stop,it->offset);
                const auto n=static_cast<size_t>(end-offset);
                std::memset(out+got,0,n); got+=n; offset=end;
                continue;
            }
            auto found=cache.find(it->file);
            if (found==cache.end()) {
                // Bound CRT/OS handles even when a match touches hundreds of assets.
                if (cache.size()>=32) {
                    auto oldest=std::min_element(cache.begin(),cache.end(),
                        [](const auto& a,const auto& b) { return a.second.used<b.second.used; });
                    cache.erase(oldest);
                }
                found=cache.try_emplace(it->file,it->file,++tick).first;
            }
            auto& cached=found->second; cached.used=++tick;
            auto& stream=cached.stream;
            if (!stream.is_open()) { cache.erase(found); return false; }
            const auto n=static_cast<size_t>(std::min(stop,it->offset+it->length)-offset);
            stream.clear();
            stream.seekg(static_cast<std::streamoff>(it->fileOffset+offset-it->offset));
            if (!stream.read(reinterpret_cast<char*>(out+got),static_cast<std::streamsize>(n))) return false;
            got+=n; offset+=n;
            if (offset==it->offset+it->length) ++it;
        }
        return true;
    }
};
struct State {
    std::mutex mutex;
    std::shared_ptr<Disc> mounted;
    // Each open keeps its own disc alive if the runtime changes its mount.
    std::unordered_map<std::FILE*,std::shared_ptr<Disc>> handles;
};
State& state() { static State value; return value; }
std::shared_ptr<Disc> current() {
    auto& s=state(); std::lock_guard<std::mutex> lock(s.mutex); return s.mounted;
}
std::FILE* hostOpen(const fs::path& path) {
#ifdef _WIN32
    return ::_wfopen(path.c_str(),L"rb");
#else
    return std::fopen(path.c_str(),"rb");
#endif
}
}
bool mount(const std::filesystem::path& path) {
    std::shared_ptr<Disc> next;
    if (path.filename()=="disc.index") next=std::make_shared<Disc>(path);
    auto& s=state(); std::lock_guard<std::mutex> lock(s.mutex);
    s.mounted=next;
    if (next) std::fprintf(stderr,"[disc-install] mounted extracted files: %zu ranges, %llu virtual bytes\n",
        next->ranges.size(),static_cast<unsigned long long>(next->size));
    return bool(next);
}
uint64_t mountedSize() { auto disc=current(); return disc?disc->size:0; }
bool readMounted(uint64_t offset,void* data,size_t length,size_t& got) {
    got=0; auto disc=current(); return disc && disc->read(offset,data,length,got);
}
bool location(const std::filesystem::path& asset,uint32_t& lbn) {
    auto disc=current(); if (!disc) return false;
    std::error_code error; auto canonical=std::filesystem::canonical(asset,error);
    if (error) return false;
    for (const auto& r:disc->ranges) if (r.asset && r.file==canonical) {
        lbn=static_cast<uint32_t>(r.offset/2048); return true;
    }
    return false;
}
std::FILE* open(const std::filesystem::path& path,uint64_t* length) {
    auto disc=current();
    std::error_code error;
    bool installed=disc && std::filesystem::equivalent(path,disc->index,error) && !error;
    error.clear();
    uint64_t size=installed?disc->size:std::filesystem::file_size(path,error);
    if (error) return nullptr;
    auto* file=hostOpen(path); if (!file) return nullptr;
    if (installed) {
        auto& s=state(); std::lock_guard<std::mutex> lock(s.mutex); s.handles.emplace(file,disc);
    }
    if (length) *length=size;
    return file;
}
bool read(std::FILE* file,uint64_t offset,void* data,size_t length,size_t& got) {
    got=0;
    if (!file || (!data && length)) return false;
    std::shared_ptr<Disc> disc;
    { auto& s=state(); std::lock_guard<std::mutex> lock(s.mutex);
      auto it=s.handles.find(file); if (it!=s.handles.end()) disc=it->second; }
    if (disc) return disc->read(offset,data,length,got);
    if (offset>static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return false;
#ifdef _WIN32
    if (::_fseeki64(file,static_cast<int64_t>(offset),SEEK_SET)!=0) return false;
#else
    if (::fseeko(file,static_cast<off_t>(offset),SEEK_SET)!=0) return false;
#endif
    got=length?std::fread(data,1,length,file):0;
    if (got<length && std::ferror(file)) { std::clearerr(file); return false; }
    return true;
}
void close(std::FILE* file) {
    if (!file) return;
    { auto& s=state(); std::lock_guard<std::mutex> lock(s.mutex); s.handles.erase(file); }
    std::fclose(file);
}
}

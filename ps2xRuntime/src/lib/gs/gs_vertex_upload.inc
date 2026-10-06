// Write directly to a disjoint destination. Never read upload-heap memory:
// it can be write-combined. Rotate GL's last provoking vertex to D3D's first.
namespace wotm_capture {
template<class T> void copyProvokingVertices(void* output,const T* input,size_t count) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (!count) return;
  auto* bytes=static_cast<uint8_t*>(output);
  if (!input) { std::memset(bytes,0,count*sizeof(T));return; }
  size_t i=0;
  for(;i+2<count;i+=3) {
    std::memcpy(bytes+i*sizeof(T),input+i+2,sizeof(T));
    std::memcpy(bytes+(i+1)*sizeof(T),input+i,sizeof(T));
    std::memcpy(bytes+(i+2)*sizeof(T),input+i+1,sizeof(T));
  }
  if(i<count)std::memcpy(bytes+i*sizeof(T),input+i,(count-i)*sizeof(T));
}
}

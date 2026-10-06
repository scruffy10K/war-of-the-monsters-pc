// Private shared recorder conversion. Only disjoint world STQ streams use it.
// UI, fixed UV, tracing and sparse indexed streams retain the general converter.
#include <immintrin.h>
namespace wotm_vertex {
template<class Out, class In>
inline void storeWorldVertex(Out& out, const In& v, float z, float ofx, float ofy) {
    out.x = v.x - ofx;
    out.y = v.y - ofy;
    out.z = z;
    out.r = static_cast<float>(v.r) / 255.0f;
    out.g = static_cast<float>(v.g) / 255.0f;
    out.b = static_cast<float>(v.b) / 255.0f;
    out.a = static_cast<float>(v.a) / 128.0f;
    out.s = v.s;
    out.t = v.t;
    out.q = v.q;
    out.fog = static_cast<float>(v.fog);
}
template<class Out, class In>
inline void convertWorldVertices(Out* __restrict output,
    const In* __restrict input, size_t count, float ofx, float ofy) {
    size_t i = 0;
    for (; i + 1 < count; i += 2) {
        const __m128d z = _mm_set_pd(input[i+1].z, input[i].z);
        const __m128 depth = _mm_cvtpd_ps(_mm_div_pd(z, _mm_set1_pd(16777215.0)));
        storeWorldVertex(output[i], input[i], _mm_cvtss_f32(depth), ofx, ofy);
        storeWorldVertex(output[i+1], input[i+1],
            _mm_cvtss_f32(_mm_shuffle_ps(depth, depth, _MM_SHUFFLE(1,1,1,1))), ofx, ofy);
    }
    if (i < count)
        storeWorldVertex(output[i], input[i], static_cast<float>(input[i].z / 16777215.0), ofx, ofy);
}
}

#version 330
in vec2 fragTexCoord;
out vec4 finalColor;
uniform sampler2D texture0;
uniform vec2 pixelStep;
uniform vec4 sampleBounds;
vec3 colorAt(vec2 p) { return texture(texture0, clamp(p, sampleBounds.xy, sampleBounds.zw)).rgb; }
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
void main() {
    vec2 p = fragTexCoord;
    vec3 c = colorAt(p);
    float m = luma(c);
    float nw = luma(colorAt(p + vec2(-1,-1)*pixelStep));
    float ne = luma(colorAt(p + vec2( 1,-1)*pixelStep));
    float sw = luma(colorAt(p + vec2(-1, 1)*pixelStep));
    float se = luma(colorAt(p + vec2( 1, 1)*pixelStep));
    float lo = min(m,min(min(nw,ne),min(sw,se)));
    float hi = max(m,max(max(nw,ne),max(sw,se)));
    if (hi-lo < max(0.0312,hi*0.125)) { finalColor=vec4(c,1); return; }
    vec2 dir = vec2(-((nw+ne)-(sw+se)), (nw+sw)-(ne+se));
    float reduce = max((nw+ne+sw+se)*0.03125, 0.0078125);
    dir = clamp(dir/(min(abs(dir.x),abs(dir.y))+reduce),vec2(-8),vec2(8))*pixelStep;
    vec3 a = 0.5*(colorAt(p-dir/6.0)+colorAt(p+dir/6.0));
    vec3 b = a*0.5+0.25*(colorAt(p-dir*0.5)+colorAt(p+dir*0.5));
    float lb=luma(b);
    finalColor=vec4((lb<lo || lb>hi)?a:b,1);
}

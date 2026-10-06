// Mirrors the established GL GS shading contract; input vertices are screen-space.
cbuffer DrawConstants : register(b0) {
 float2 targetSize; float2 texSize;
 uint4 flags; // textured, TCC, TFX, IIP
 uint4 tests; // ATE, ATST, inverted AFAIL pass, PABE
 float4 alphaFog; // AREF raw, fog enabled, reserved, render target height
 float4 fogColor;
 int4 wrapMode; // U,V, render-target texture (flip V), dither
 float4 region;
 float4 dimx[4];
 float4 motionInfo; // output dimensions, opaque motion allowed, reserved
};
Texture2D<float4> sourceTexture : register(t0);
SamplerState sourceSampler : register(s0);
struct Input {float3 pos:POSITION;float4 color:COLOR;float3 stq:TEXCOORD;float fog:FOG;
#ifdef MOTION_OUTPUT
 float4 previous:MOTION;
#endif
};
struct Varying {float4 pos:SV_Position;float4 color:COLOR0;nointerpolation float4 flatColor:COLOR1;float3 stq:TEXCOORD;float fog:FOG;
#ifdef MOTION_OUTPUT
 float4 previous:MOTION;
#endif
};
Varying VSMain(Input v) {
 Varying o;o.pos=float4(v.pos.x/targetSize.x*2-1,1-v.pos.y/targetSize.y*2,v.pos.z,1);
 o.color=v.color;o.flatColor=v.color;o.stq=v.stq;o.fog=v.fog;
#ifdef MOTION_OUTPUT
 o.previous=v.previous;
#endif
 return o;
}
float wrapAxis(float x,int mode,float lo,float hi){if(mode==2)return clamp(x,lo,hi);if(mode==3)return float((int(x)&int(lo))|int(hi));return x;}
#ifdef MOTION_OUTPUT
struct PixelOutput {float4 color:SV_Target0;float4 motion:SV_Target1;};
PixelOutput PSMain(Varying v) {
#else
float4 PSMain(Varying v):SV_Target {
#endif
 float4 shade=flags.w!=0?v.color:v.flatColor;float originalAlpha=shade.a;
 if(flags.x!=0){
  float2 st=v.stq.xy/abs(v.stq.z)*texSize;
  // Compare absolute IEEE bit patterns: rejects NaN, infinity and magnitudes
  // above 1e6 without the FXC combined isnan/isinf/float-comparison lowering
  // that collapsed valid coordinates in the captured GS scene.
  if(any((asuint(st) & 0x7fffffffu) > asuint(1000000.0f)))st=0;
  st=float2(wrapAxis(st.x,wrapMode.x,region.x,region.y),wrapAxis(st.y,wrapMode.y,region.z,region.w));
  float2 uv=st/texSize;
  // Live D3D scene targets are top-down, unlike GL framebuffer textures.
  if(alphaFog.z!=0)uv=st/float2(640,224);
  else if(wrapMode.z!=0)uv.y=1-uv.y;
  float4 texel=sourceTexture.Sample(sourceSampler,uv);float a=texel.a*(255.0/128.0);
  if(flags.z==1){shade.rgb=texel.rgb;if(flags.y!=0)shade.a=a;}
  else{
   shade.rgb=texel.rgb*shade.rgb*(255.0/128.0);if(flags.y!=0)shade.a*=a;
   if(flags.z>=2){shade.rgb+=originalAlpha*(128.0/255.0);if(flags.y!=0)shade.a=flags.z==2?a+originalAlpha:a;}
  }
 }
 if(alphaFog.y!=0)shade.rgb=shade.rgb*(v.fog/256.0)+fogColor.rgb*((255-v.fog)/256.0);
 if(tests.x!=0){
  float a=shade.a*128.0;bool accepted=true;
  if(tests.y==0)accepted=false;else if(tests.y==2)accepted=a<alphaFog.x;else if(tests.y==3)accepted=a<=alphaFog.x;
  else if(tests.y==4)accepted=abs(a-alphaFog.x)<0.5;else if(tests.y==5)accepted=a>=alphaFog.x;
  else if(tests.y==6)accepted=a>alphaFog.x;else if(tests.y==7)accepted=abs(a-alphaFog.x)>=0.5;
  if(tests.z!=0)accepted=!accepted;if(!accepted)discard;
 }
 if(wrapMode.w!=0){int2 cell=int2(v.pos.x,alphaFog.w-v.pos.y)&3;shade.rgb+=dimx[cell.y][cell.x]/255.0;}
 if(tests.w!=0&&shade.a<1)shade.a=1;
#ifdef MOTION_OUTPUT
 PixelOutput output;output.color=saturate(shade);output.motion=float4(0,0,0,1);
 if(motionInfo.w==2 || (motionInfo.w!=0 && shade.a<=0))output.motion=0;
 if((motionInfo.z==1 || (motionInfo.z==2 && shade.a>=1)) && v.previous.w>0.9999 && v.previous.z>0.000001) {
  float2 previous=v.previous.xy/v.previous.z;
  float2 current=v.pos.xy*targetSize/motionInfo.xy;
  output.motion=float4((previous-current)*motionInfo.xy/targetSize,1,1);
 }
 return output;
#else
 return saturate(shade);
#endif
}

// Presentation-only FXAA. region.xy is the visible upper UV bound in the
// padded GS target; neighboring samples must never read the unused padding.
float3 presentColor(float2 uv) {
 float2 halfPixel=0.5/texSize;
 return sourceTexture.Sample(sourceSampler,clamp(uv,halfPixel,region.xy-halfPixel)).rgb;
}
float presentLuma(float3 color) {return dot(color,float3(0.299,0.587,0.114));}
float4 PSFxAA(Varying v):SV_Target {
 float2 p=v.stq.xy/v.stq.z;
 float2 step=1.0/texSize;
 float3 center=presentColor(p);
 float m=presentLuma(center);
 float nw=presentLuma(presentColor(p+float2(-1,-1)*step));
 float ne=presentLuma(presentColor(p+float2( 1,-1)*step));
 float sw=presentLuma(presentColor(p+float2(-1, 1)*step));
 float se=presentLuma(presentColor(p+float2( 1, 1)*step));
 float lo=min(m,min(min(nw,ne),min(sw,se)));
 float hi=max(m,max(max(nw,ne),max(sw,se)));
 if(hi-lo<max(0.0312,hi*0.125))return float4(center,1);
 float2 dir=float2(-((nw+ne)-(sw+se)),(nw+sw)-(ne+se));
 float reduce=max((nw+ne+sw+se)*0.03125,0.0078125);
 dir=clamp(dir/(min(abs(dir.x),abs(dir.y))+reduce),-8.0,8.0)*step;
 float3 a=0.5*(presentColor(p-dir/6.0)+presentColor(p+dir/6.0));
 float3 b=a*0.5+0.25*(presentColor(p-dir*0.5)+presentColor(p+dir*0.5));
 float lb=presentLuma(b);
 return float4((lb<lo||lb>hi)?a:b,1);
}

// t0=current finished frame, t1=previous, t2=current-to-previous motion in
// internal-render pixels. Unknown/HUD/fade surfaces retain the current picture.
Texture2D<float4> previousFrame : register(t1);
Texture2D<float4> interpolationMotion : register(t2);
float3 interpolatedColor(float2 p) {
 float3 fallback=presentColor(p);
 float alpha=saturate(motionInfo.x);
 if(alpha>=0.999)return fallback;
 float2 at=p;
 float4 flow=interpolationMotion.SampleLevel(sourceSampler,at,0);
 if(flow.z<0.999 || any(abs(flow.xy)>motionInfo.z))return fallback;
 [unroll]for(int i=0;i<2;++i) {
  at=p-(1-alpha)*flow.xy/texSize;
  if(any(at<0) || any(at>region.xy))return fallback;
  flow=interpolationMotion.SampleLevel(sourceSampler,at,0);
  if(flow.z<0.999 || any(abs(flow.xy)>motionInfo.z))return fallback;
 }
 float2 old=at+flow.xy/texSize;
 if(any(old<0) || any(old>region.xy))return fallback;
 float3 current=presentColor(at),previous=previousFrame.SampleLevel(sourceSampler,old,0).rgb;
 // Large colour changes indicate disocclusion or an animated effect whose
 // geometry motion does not describe its pixels. Keep current in those cases.
 if(max(max(abs(current.r-previous.r),abs(current.g-previous.g)),abs(current.b-previous.b))>0.25)return fallback;
 return lerp(previous,current,alpha);
}
float4 PSInterpolation(Varying v):SV_Target {
 float2 p=v.stq.xy/v.stq.z;float3 center=interpolatedColor(p);
 if(motionInfo.y==0)return float4(center,1);
 float2 step=1.0/texSize;float m=presentLuma(center);
 float nw=presentLuma(interpolatedColor(p+float2(-1,-1)*step));
 float ne=presentLuma(interpolatedColor(p+float2(1,-1)*step));
 float sw=presentLuma(interpolatedColor(p+float2(-1,1)*step));
 float se=presentLuma(interpolatedColor(p+float2(1,1)*step));
 float lo=min(m,min(min(nw,ne),min(sw,se))),hi=max(m,max(max(nw,ne),max(sw,se)));
 if(hi-lo<max(0.0312,hi*0.125))return float4(center,1);
 float2 dir=float2(-((nw+ne)-(sw+se)),(nw+sw)-(ne+se));
 float reduce=max((nw+ne+sw+se)*0.03125,0.0078125);
 dir=clamp(dir/(min(abs(dir.x),abs(dir.y))+reduce),-8.0,8.0)*step;
 float3 a=0.5*(interpolatedColor(p-dir/6.0)+interpolatedColor(p+dir/6.0));
 float3 b=a*0.5+0.25*(interpolatedColor(p-dir*0.5)+interpolatedColor(p+dir*0.5));
 float lb=presentLuma(b);return float4((lb<lo||lb>hi)?a:b,1);
}

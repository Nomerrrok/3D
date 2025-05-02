cbuffer global : register(b5)
{
    float4 gConst[32];
};

cbuffer frame : register(b4)
{
    float4 time;
    float4 aspect;
};

cbuffer camera : register(b3)
{
    float4x4 world[2];
    float4x4 view[2];
    float4x4 proj[2];
};
struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float4 vpos : POSITION0;
    float4 wpos : POSITION1;
    float4 vnorm : NORMAL1;
    float2 uv : TEXCOORD0;
};
Texture2D ObjTexture;
SamplerState ObjSamplerState;

float4 PS(VS_OUTPUT input) : SV_Target
{
    float pi = 3.141519;

float3 nrml = input.vnorm.xyz;

float f = dot(float3(1, 0, 0), nrml);

//return float4(input.uv.x, 0, 0, 1);

float c = 0;
for (int i = 1; i < 3; i++)
{
    float2 uv = 2 * ((input.uv) - .5) * pi;
    uv += float2(sin(time.x * .13 * sin(i * .4)), sin(time.x * .12 * sin(i * .5)));
    c += sin((atan2(uv.x, uv.y) * 12 - time.x * .3)) * (sin(1 / length(uv * 2) + 5)) * saturate(1 / pow(length(uv),3)) * 2;
}
return ObjTexture.Sample(ObjSamplerState, input.uv);
return float4(c, c, c, 1.);

}
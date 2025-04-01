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

cbuffer drawMat : register(b2)
{
    float4x4 model;
    float hilight;
};

cbuffer params : register(b1)
{
    float r, g, b;
};

struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float3 normal : NORMAL0;
    float2 uv : TEXCOORD0;
    uint faceID : COLOR0;
};

// Цвета для 20 граней
static const float3 faceColors[20] = {
    {1.0, 0.2, 0.2}, {0.2, 1.0, 0.2}, {0.2, 0.2, 1.0}, {1.0, 1.0, 0.2},
    {1.0, 0.2, 1.0}, {0.2, 1.0, 1.0}, {0.5, 0.5, 0.5}, {1.0, 0.5, 0.2},
    {0.5, 1.0, 0.2}, {0.2, 1.0, 0.5}, {0.2, 0.5, 1.0}, {0.5, 0.2, 1.0},
    {1.0, 0.2, 0.5}, {1.0, 0.5, 0.5}, {0.5, 1.0, 0.5}, {0.5, 0.5, 1.0},
    {0.75, 0.25, 0.5}, {0.25, 0.75, 0.5}, {0.5, 0.25, 0.75}, {0.25, 0.5, 0.75}
};

float4 PS(VS_OUTPUT input) : SV_Target
{

    float3 baseColor = faceColors[input.faceID % 20];

    return float4(baseColor, 1.0);
}
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

cbuffer objParams : register(b0)
{
    float gx;
    float gy;
};

struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float4 vpos : POSITION0;
    float4 wpos : POSITION1;
    float4 vnorm : NORMAL1;
    float2 uv : TEXCOORD0;
};

float3 rotY(float3 pos, float a)
{
    float3x3 m =
    {
        cos(a), 0, sin(a),
        0, 1, 0,
        -sin(a), 0, cos(a)
    };
    pos = mul(pos, m);
    return pos;
}
float3 rotX(float3 pos, float a)
{
    float3x3 m =
    {
        1, 0, 0,
        0, cos(a), -sin(a),
        0, sin(a), cos(a)
    };
    pos = mul(pos, m);
    return pos;
}

float3 rotZ(float3 pos, float a)
{
    float3x3 m =
    {
        cos(a), sin(a), 0,
        -sin(a), cos(a), 0,
        0,0, 1
    };
    pos = mul(pos, m);
    return pos;
}
#define PI 3.1415926535897932384626433832795

VS_OUTPUT VS(uint vID : SV_VertexID)
{
    VS_OUTPUT output = (VS_OUTPUT)0;

    float2 quad[6] = { -1,-1,1,-1,-1,1,1,-1,1,1,-1,1 };

    float2 p = quad[vID % 6];

    int qID = vID / 6;



    float R = 20;
    float r = 0.2;

    float x = (qID % (uint)gx + p.x / 2.) / gx +.5;
    float y = (qID / (uint)gy + p.y / 2.) / gy +.5 ;
    float2 a = float2(x, y) * PI * 2;
    a.x *= -1;

    float3 pos = float3((2 + cos(a.x * 3)) * cos(a.x * 2), -sin(a.x * 3), (2 + cos(3 * a.x)) * sin(2 * a.x));

    float3 dpos = normalize(float3(-2 * (R + r * cos(3 * a.x)) * sin(2 * a.x) - 3 * r * sin(3 * a.x) * cos(2 * a.x),
        3 * r * cos(3 * a.x),
        2 * (R + r * cos(3 * a.x)) * cos(2 * a.x) - 3 * r * sin(3 * a.x) * sin(2 * a.x)));

    float3 B = normalize(cross(dpos, float3(0, 1, 0)));
    float3 N = normalize(cross(B, dpos)); 

    pos = pos + r * (cos(a.y) * N + sin(a.y) * B);
    output.pos = mul(float4(pos, 1), mul(view[0], proj[0]));
    output.uv = float2(1, -1) * p;
    return output;
}

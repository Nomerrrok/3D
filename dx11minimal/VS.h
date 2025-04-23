cbuffer global : register(b5)
{
    float4 gConst[32];
};

cbuffer frame : register(b4)
{
    float4 time;
    float4 aspect;
    float2 iResolution;
    float2 pad;
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
    float4 wnorm : NORMAL2;
    float4 bnorm : NORMAL3;
    float2 uv : TEXCOORD0;
    float4 singlePos : POSITION2;
};

float3 rotY(float3 pos, float a)
{
    float3x3 m = float3x3(
        cos(a), 0, sin(a),
        0, 1, 0,
        -sin(a), 0, cos(a)
    );
    return mul(pos, m);
}

float3 rotX(float3 pos, float a)
{
    float3x3 m = float3x3(
        1, 0, 0,
        0, cos(a), -sin(a),
        0, sin(a), cos(a)
    );
    return mul(pos, m);
}

float3 rotZ(float3 pos, float a)
{
    float3x3 m = float3x3(
        cos(a), sin(a), 0,
        -sin(a), cos(a), 0,
        0, 0, 1
    );
    return mul(pos, m);
}

#define PI 3.1415926535897932384626433832795
float length(float3 c)
{
    float x = c.x;
    float y = c.y;
    float z = c.z;
    float l = sqrt(x * x + y * y + z * z);
    return l;
}

float3 calcGeom(float2 a)
{
    float R = 1.0;  

    float3 pos = float3(
        R * cos(a.y/2) * cos(a.x),  
        R * cos(a.y/2) * sin(a.x),  
        R * sin(a.y/2)              
    );


    float3 norm = normalize(pos);


    //pos = rotY(pos, time.x * 0.01);

    return pos;
}

VS_OUTPUT VS(uint vID : SV_VertexID, uint iID : SV_InstanceID)
{
    VS_OUTPUT output = (VS_OUTPUT)0;

    float2 quad[6] = {
        float2(-1, -1), float2(1, -1), float2(-1, 1),
        float2(1, -1), float2(1, 1), float2(-1, 1)
    };

    float2 p = quad[vID % 6];
    int qID = vID / 6;

    float x = (qID % (uint)gx + p.x * 0.5) / gx + 0.5;
    float y = (qID / (uint)gx + p.y * 0.5) / gy + 0.5;

    float stepX = 1.0 / gx;
    float stepY = 1.0 / gy;

    float2 a = float2(x, y) * PI * 2.0;
    a.x *= -1.0;
    float2 a1 = a + float2(stepX * PI * 2.0, 0);
    float2 a2 = a + float2(0, stepY * PI * 2.0);
    float3 pos = calcGeom(a);
    float3 pos1 = calcGeom(a1);
    float3 pos2 = calcGeom(a2);
    float3 binormal = normalize(pos2 - pos);
    float3 tangent = normalize(pos1 - pos);
    float3 norm = normalize(pos);
    norm = mul(norm, transpose(view[0]));

    float s = iID % 5;
    float t = iID % 3;
    pos.x += s * 3.0f - 6.0f;
    pos.y += t * 3.0f - 3.f;
    pos *= 0.2f;

    output.singlePos = float4(s+1, t+1, 0, 1);
    output.pos = mul(mul(float4(pos, 1.0), view[0]), proj[0]);
    output.wpos = output.pos;
    float2 uv = float2(x, y);
    output.uv = uv * float2(3, 2);
    output.vnorm = float4(norm, 1.0);
    output.bnorm = float4(binormal, 1.0);
    output.wnorm = float4(tangent, 1.0);
    return output;
}
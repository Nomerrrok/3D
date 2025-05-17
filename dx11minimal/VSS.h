cbuffer camera : register(b3)
{
    float4x4 world[2];
    float4x4 view[2];  // view[1] Ч дл€ света
    float4x4 proj[2];  // proj[1] Ч дл€ света
};

cbuffer drawMat : register(b2)
{
    float4x4 model;
};

cbuffer frame : register(b4)
{
    float4 time;       // time.x Ч текущее врем€
    float4 aspect;
    float2 iResolution;
    float2 pad;
};

cbuffer objParams : register(b0)
{
    float gx;
    float gy;
};

struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float3 cubeToSphere(float3 p)
{
    return normalize(p);
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

float3 rotY(float3 pos, float a)
{
    float3x3 m = float3x3(
        cos(a), 0, sin(a),
        0, 1, 0,
        -sin(a), 0, cos(a)
    );
    return mul(pos, m);
}

float3 calcGeom(float2 uv, int faceID, float time)
{
    float2 p = uv * 2.0 - 1.0;
    float3 cubePos;
    if (faceID == 0)      cubePos = float3(-1, p.y, p.x);
    else if (faceID == 1) cubePos = float3(1, p.y, -p.x);
    else if (faceID == 2) cubePos = float3(-p.x, -1, -p.y);
    else if (faceID == 3) cubePos = float3(-p.x, 1, p.y);
    else if (faceID == 4) cubePos = float3(-p.x, p.y, -1);
    else if (faceID == 5) cubePos = float3(p.x, p.y, 1);
    else                  cubePos = float3(0, 0, 0);

    cubePos = normalize(cubePos);
    cubePos = rotX(cubeToSphere(cubePos), time * 0.05);
    return rotY(cubePos, time * 0.05);
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
    int vg = (int)(gx * gy);
    int localID = qID % vg;
    int faceID = qID / vg;

    int px = localID % (int)gx;
    int py = localID / (int)gx;
    float2 uv = float2(px + 0.5 + p.x * 0.5, py + 0.5 + p.y * 0.5) / float2(gx, gy);
    float3 pos = calcGeom(uv, faceID, time.x);

    float4 worldPos = mul(float4(pos, 1.0), model);
    float4 lightSpacePos = mul(worldPos, mul(view[1], proj[1]));
    float2 projUV = lightSpacePos.xy / lightSpacePos.w * 0.5 + 0.5;

    output.uv = projUV;
    output.pos = lightSpacePos;

    return output;
}

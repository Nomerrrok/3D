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

VS_OUTPUT VS(uint vID : SV_VertexID)
{
    VS_OUTPUT output = (VS_OUTPUT)0;

    float3 quad[36] = {
        // ѕередн€€ грань
-1, -1,  1,
 1, -1,  1,
-1,  1,  1,

 1, -1,  1,
 1,  1,  1,
-1,  1,  1,
//задн€€ грань
-1, -1,-1,
 -1, 1,-1,
  1, -1,-1,

 1, -1,-1,
 -1,  1,-1,
 1,  1,-1,

 //права€ грань
1,-1,-1,
1,1,-1,
1,-1,1,

1,1,-1,
1,1,1,
1,-1,1,

//лева€ грвнь
-1,1,1,
-1,1,-1,
-1,-1,1,

-1,1,-1,
-1,-1,-1,
-1,-1,1,

//верхн€€ грань
1,  1, 1,
 1,  1, -1,
-1,  1,  1,

 -1,  1,  -1,
-1,  1,  1,
 1,  1, -1,

 //нижн€€ грань
 -1,  -1, -1,
 1,  -1, -1,
-1, -1,  1,

 1,  -1,  1,
-1,  -1,  1,
 1,  -1, -1
    };

    float3 p = quad[vID];
    float4 pos = float4(quad[vID], 1);
    
    float4 rPos = mul(float4(0,0,1,1), view[0]);
    output.vnorm = rPos;

    output.pos = mul(pos, mul(view[0], proj[0]));
    float2 pUV = float2(atan2(p.x,p.z)/4,p.y/2+.5);
    output.uv = float2(1, -1) * pUV / 2. + .5;
    return output;
}

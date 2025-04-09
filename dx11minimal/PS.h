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
    float4 vpos : POSITION0;
    float4 wpos : POSITION1;
    float4 vnorm : NORMAL1;
    float2 uv : TEXCOORD0;
};

float nrand(float2 n)
{
    return frac(sin(dot(n.xy, float2(12.9898, 78.233))) * 43758.5453);
}

float sincosbundle(float val)
{
    return sin(cos(2. * val) + sin(4. * val) - cos(5. * val) + sin(3. * val)) * 0.05;
}

float3 color(float2 uv)
{
    float2 coord = floor(uv);
    float2 gv = frac(uv);

    float movingValue = -sincosbundle(coord.y) * 2.;

    float offset = floor(fmod(uv.y, 2.0)) * (1.5);
    float verticalEdge = abs(cos(uv.x + offset));

    float3 brick = float3(0.45, 0.29, 0.23) - movingValue;

    bool vrtEdge = step(1. - 0.01, verticalEdge) == 1.0;
    bool hrtEdge = gv.y > (0.9) || gv.y < (0.1);

    if (hrtEdge || vrtEdge)
        return float3(0.845, 0.845, 0.845);

    return brick;
}

float lum(float2 uv)
{
    float3 rgb = color(uv);
    return 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
}

float3 normal(float2 uv)
{
    float r = 0.03;

    float x0 = lum(uv + float2(r, 0.0));
    float x1 = lum(uv - float2(r, 0.0));
    float y0 = lum(uv + float2(0.0, r));
    float y1 = lum(uv - float2(0.0, r));

    float s = 1.0;
    float3 n = normalize(float3(x1 - x0, y1 - y0, s));

    float3 p = float3(uv * 2.0 - 1.0, 0.0);
    float3 v = float3(0.0, 0.0, 1.0);

    float3 l = v - p;
    float d_sqr = dot(l, l);
    l *= (1.0 / sqrt(d_sqr));

    float3 h = normalize(l + v);

    float dot_nl = clamp(dot(n, l), 0.0, 1.0);
    float dot_nh = clamp(dot(n, h), 0.0, 1.0);

    float color = lum(uv) * pow(dot_nh, 14.0) * dot_nl * (1.0 / d_sqr);
    color = pow(color, 1.0 / 2.2);

    return (n * 0.5 + 0.5);
}

float4 PS(VS_OUTPUT input) : SV_Target
{
    float3 nrml = normalize(input.vnorm.xyz);
    float3 pos = input.wpos.xyz;

    float3 cameraPos = float3(3.5, 0, 0);
    float3 lightPos = float3(1, 0, 0);

    float3 L = normalize(lightPos - pos);
    float3 V = normalize(cameraPos - pos);
    float3 H = normalize(L + V);

    float spec = pow(saturate(dot(nrml, H)), 32);

    float f = saturate(dot(float3(1, 0, 0), nrml)) + spec;
    float xScale = 14.0;
    float yScale = 2.0; 

    float2 uv = input.uv * float2(xScale, yScale); 
    float3 fragColor = color(uv);

   // if ((input.pos.x - input.pos.y) > (aspect.x - aspect.y) / 2.0)
    {
       // fragColor = normal(uv);
    }

    return float4(fragColor, 1.0);
}

cbuffer InstanceData : register(b6)
{
    int index;
};

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
#define PI 3.1415926535897932384626433832795
struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float4 vpos : POSITION0;
    float4 wpos : POSITION1;
    float4 vnorm : NORMAL1;
    float4 wnorm : NORMAL2;
    float4 bnorm : NORMAL3;
    float2 uv : TEXCOORD0;
    float3 singlePos : POSITION2;

};

float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotV, roughness);
    float ggx2 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float nrand(float2 n)
{
    return frac(sin(dot(n.xy, float2(19.9898, 78.233))) * 43758.5453);
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


float3 sfMap(float3 v)
{
    float3 c = sign(saturate(sin(v.x * 33) / sin(v.z * 33)));
    float3 a = c;
    if (v.y > 0) c *= float3(1, 0, 0);
    if (v.y < -0) c *= float3(0, 0, 1);

    if (v.x > 0.5) c = a * float3(0, 1, 0);
    if (v.x < -0.5) c = a * float3(0, 1, 1);

    if (abs(v.x) < .5 && v.z > 0.5) c = a * float3(1, 1, 0);
    if (abs(v.x) < .5 && v.z < -0.5) c = a * float3(1, 0, 1);

    float3 lp = float3(0, 1, 0);

    c += 4 * pow(1 / (distance(lp, v) + .75), 12);

    return c;
}


float4 PS(VS_OUTPUT input) : SV_Target
{
    float3 T = input.wnorm.xyz;
    float3 B = input.bnorm.xyz;
    float3 N = input.vnorm.xyz;
    float3 baseColor = float3(0.5, 0.5, 0.5);
    //float2 brickUV = input.uv * float2(10, 10);
   // float2 uv = input.uv;

    //float3 texNormal = normal(brickUV) * 2.0 - 1.0;

    //float3x3 TBN = float3x3(T, B, N);
    //float3 finalNormal = mul(texNormal, TBN);
    //finalNormal = N;
   // float3x3 vm = (float3x3)view[0];
    //finalNormal = mul(finalNormal,vm);

    //float3 N_color = N * 0.5 + 0.5;
    //float3 B_color = B * 0.5 + 0.5;
    //float3 T_color = T * 0.5 + 0.5;

    //float3 baseColor = color(brickUV);

    float3 pos = input.wpos.xyz;
    float4x4 invView = saturate(view[0]);
    float3 cameraPos = invView._m03_m13_m23.xyz;
    //cameraPos.x = cameraPos+x-6;
    //cameraPos.y = cameraPos + y-3;

    float3 lightPos = normalize(float3(1, 0, 0));
    float3 SinglePos = input.singlePos;
    float roughness = saturate(0.05 + (SinglePos.x - 1) * (1.0 - 0.05));
    float3 F0 = lerp(0.1, 0.9, saturate((SinglePos.y - 1.0)));

    float3 L = normalize(lightPos - pos);
   float3 V = normalize(cameraPos - pos);
    float3 H = normalize(L + V);
    float3 finalNormal = N;
    float NL = saturate(dot(finalNormal, L));
    float NH = saturate(dot(finalNormal, H));

    float ambient = 0.1;
    float diffuse = NL;
    float specular = pow(NH, 64.0) * 8.0;

    float lighting = ambient + diffuse + specular;
    // Итоговый цвет
    float3 fragColor = baseColor * lighting;

    float3 ref = reflect(V, N);
    float3 env = sfMap(ref);
    //float3 env = sfMap(N);

    return float4(env,1);

    return float4(fragColor, 1.0);
    return float4(finalNormal / 2 + .5, 1.0);
}
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
    float4 singlePos : POSITION2;
};

float GGX_PartialGeometry(float cosThetaN, float alpha) {
    float cosTheta_sqr = saturate(cosThetaN * cosThetaN);
    float tan2 = (1 - cosTheta_sqr) / cosTheta_sqr;
    float GP = 2 / (1 + sqrt(1 + alpha * alpha * tan2));
    return GP;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;


    return NdotV / (NdotV * (1.0 - k) + k);
}


float DistributionGGX(float3 N, float3 H, float a)
{
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float GeometrySmith(float3 N, float3 V, float3 L, float k)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotV, k);
    float ggx2 = GeometrySchlickGGX(NdotL, k);

    return ggx1 * ggx2;
}

float3 CookTorrance_GGX(float3 n, float3 l, float3 v, float roughness,float f0,float metallic) {
    n = normalize(n);
    v = normalize(v);
    l = normalize(l);
    float3 h = normalize(v + l);
    float3 Lo = float3(0,0,0);
    float albedo = (0.5, 0.5, 0.5);
    //precompute dots
    float NL = dot(n, l);
    if (NL <= 0.0) return 0.0;
    float NV = dot(n, v);
    if (NV <= 0.0) return 0.0;
    float NH = dot(n, h);
    float HV = dot(h, v);

    //precompute roughness square
    float roug_sqr = roughness * roughness;

    //calc coefficients
    float NDF = DistributionGGX(n, h, roughness);
    float G = GeometrySmith(n, v, l, roughness);
    float3 F = fresnelSchlick(max(dot(n, v), 0.0), f0);
    

    float3 kS = F;
    float3 kD = float(1.0) - kS;
    kD *= 1.0 - metallic;
    kD = 1;

    float3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(n, v), 0.0) *max(dot(n, l), 0.0);
    
    denominator = max(dot(n, l), 0.0);


    //return denominator;


    float3 specular = numerator / max(denominator, 0.001);

    // прибавл€ем результат к исход€щей энергетической €ркости Lo
    float NdotL = max(dot(n, l), 0.0);
    Lo += (kD * albedo / PI + specular) *  NdotL;
    //mix
    float3 specK = G  * F * 0.25 / NV;
    //return specular;
    return max(0.0, specK);
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
    if (v.y <- 0) c *= float3(0, 0, 1);

    if (v.x>0.5) c = a*float3(0,1,0);
    if (v.x < -0.5) c = a * float3(0, 1, 1);

    if (abs (v.x) <.5 && v.z > 0.5) c = a * float3(1, 1, 0);
    if (abs(v.x) < .5 && v.z < -0.5) c = a * float3(1, 0, 1);

    float3 lp = float3(0, 1, 0);

    c += 4*pow(1 / (distance(lp, v) + .75),12);

    return c;
}


float4 PS(VS_OUTPUT input) : SV_Target
{
    float3 SinglePos = input.singlePos;
    float albedo = 1/5* SinglePos.x;
    float3 T = input.wnorm.xyz;
    float3 B = input.bnorm.xyz;
    float3 N = input.vnorm.xyz;

    //return float4(normalize(N.xyz), 1);

    float3 lightDir = normalize(float3(0, 0, -1));
    float cosTheta = dot(lightDir, N);
    float F0 = (0.04, 0.04, 0.04);
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
    float3 lightColor=(1, 1, 1);
    //float3 lightPos = normalize(float3(0, 1, 0));
    float distance = length(lightDir - pos);
    float attenuation = 1.0 / (distance * distance);
    float roughness =  1- SinglePos.y/10;
    float3 radiance = lightColor * attenuation;

    float3 L = normalize(lightDir - pos);
    float3 V = normalize(cameraPos - pos);

    float3 H = normalize(L + V);
   // float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
    float metallic = SinglePos%2;

    float3 ref = reflect(V, N);
    float3 env = sfMap(ref);
    //float3 env = sfMap(N);
    float roug_sqr = roughness * roughness;
    //float3 G = CookTorrance_GGX(N, L, V ,roughness,F0, metallic);
    float3 G = CookTorrance_GGX(N, lightDir, V, 0, 1, 1);
    float3 OutColor =  G;
   // float3 p = CookTorrance_GGX(N, L, V, roughness, F0);
    

    //OutColor = dot(N, lightDir);

    
    //return float4(frac(input.uv * 8), 0, 1);
    
    return float4(OutColor,1);



  //  return float4(p,p,p, 1.0);
    return float4(N / 2 + .5, 1.0);
}
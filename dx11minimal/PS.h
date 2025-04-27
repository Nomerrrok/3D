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
    float4 normal : NORMAL1;
    float4 tangent : NORMAL2;
    float4 binormal : NORMAL3;
    float2 uv : TEXCOORD0;
    float4 singlePos : POSITION2;
};

float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotV, roughness);
    float ggx2 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

float4 PS(VS_OUTPUT input) : SV_Target
{

    float s = input.singlePos.x;
    float t = input.singlePos.y;

    float3 N = normalize(input.normal.xyz);

    float3 fragPos = input.wpos.xyz;
    float3 cameraPos = float3(view[0]._m30, view[0]._m31, view[0]._m32);
    float3 V = normalize(cameraPos - fragPos);
    float3 L = normalize(float3(1, 0, 0)); 
    float3 H = normalize(V + L); 
    float3 T = normalize(input.tangent.xyz);
    float3 B = normalize(input.binormal.xyz);

    float3 albedo;
    float metallic;
    float roughness;

    if (s == 1.0 && t == 1.0) {
        // Золото
        albedo = float3(1.00, 0.71, 0.29);
        metallic = 1.0;
        roughness = 0.3;
    }
    else if (s == 2.0 && t == 1.0) {
        // Железо
        albedo = float3(0.56, 0.57, 0.58);
        metallic = 1.0;
        roughness = 0.2;
    }
    else if (s == 3.0 && t == 1.0) {
        // Пластик (красный)
        albedo = float3(0.8, 0.1, 0.1);
        metallic = 0.0;
        roughness = 0.4;
    }
    else if (s == 4.0 && t == 1.0) {
        // Резина
        albedo = float3(0.05, 0.05, 0.05);
        metallic = 0.0;
        roughness = 0.9;
    }
    else if (s == 5.0 && t == 1.0) {
        // Хром
        albedo = float3(0.95, 0.95, 0.95);
        metallic = 1.0;
        roughness = 0.1;
    }
    else if (t == 2.0) {
        // Для второго ряда - градация roughness
        albedo = float3(0.5, 0.5, 0.8);
        metallic = 1.0;
        roughness = s / 5.0; // От 0.2 до 1.0
    }
    else if (t == 3.0) {
        // Для третьего ряда - градация metallic
        albedo = float3(0.7, 0.7, 0.7);
        metallic = s / 5.0; // От 0.2 до 1.0
        roughness = 0.4;
    }
    else {
        // По умолчанию
        albedo = float3(0.8, 0.8, 0.8);
        metallic = 0.5;
        roughness = 0.5;
    }

   
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    float cosTheta = max(dot(V, N), 0.0);
    float3 fresnel = FresnelSchlick(cosTheta, F0);
    return float4(cosTheta.xxx, 1.0);

  //  float3 kS = fresnel;           // Сколько отражает
  //  float3 kD = 1.0 - kS;           // Сколько рассеивает
  //  kD *= 1.0 - metallic;           // Металлы НЕ дают диффуза

   // float3 diffuse = albedo / PI;   // Ламбертовская модель диффуза

  //  float3 finalColor = kD * diffuse + kS;

    // float3 N_color = H * 0.5 + 0.5;
   //return float4(N_color, 1);
    // float ao = 1.0; // Ambient occlusion
 //   // F0 - базовый коэффициент отражения при нулевом угле
 //   float3 F0 = float3(0.04, 0.04, 0.04);
 //   F0 = lerp(F0, albedo, metallic);
 //
 //   float3 radiance = float3(1.0, 1.0, 1.0); // Интенсивность света
 //
 //   // BRDF (Bidirectional Reflectance Distribution Function)
 //   float NDF = DistributionGGX(N, H, roughness);
 //   float G = GeometrySmith(N, V, L, roughness);
 //   float edgeFactor = 1.5; 
 //   float3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0) * edgeFactor;
 //
 //   float3 kS = F; // Коэффициент зеркального отражения
 //   float3 kD = float3(1.0, 1.0, 1.0) - kS; // Коэффициент диффузного отражения
 //   kD *= 1.0 - metallic;
 //
 //   float3 numerator = NDF * G * F;
 //   float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
 //   float3 specular = numerator / denominator;
 //
 //   // Угол между нормалью и светом
 //   float NdotL = max(dot(N, L), 0.0);
 //
 //   // Итоговый цвет
 //   float3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;
 //
 //   // Ambient
 //   float3 ambient = float3(0.09, 0.09, 0.09) * albedo * ao;
 //
 //   float3 color = ambient + Lo;
 //
 //   // Тоновая коррекция (tone mapping)
 //   color = color / (color + float3(1.0, 1.0, 1.0));
 //   // Гамма-коррекция
 //   float gamma = 1.6;
 //   color.rgb = pow(color.rgb, float3(1.0 / gamma, 1.0 / gamma, 1.0 / gamma));
//    return float4(color, 1.0);
}


   // float3 fragPos = input.wpos.xyz;
   // float3 lightDir = float3(0, 1, 0);
   // float3  lightColor = float3(23.47, 21.31, 20.79);
   // float3  Wi = normalize(lightDir - fragPos);
   // float cosTheta = max(dot(N, Wi), 0.0);
   // float attenuation = calculateAttenuation(fragPos, lightDir);
   // float3 radiance = lightColor * attenuation * cosTheta;
   // float3 L = normalize(lightDir - WorldPos);
   // float3 H = normalize(V + L);
   // float3 N_color = T * 0.5 + 0.5;
   // //return float4(N_color , 1.0);
   // float3 metalness = float3(0, 0, 0);
   // float3 F0 = float3(0.04, 0.04, 0.04);
   // float3 surfaceColor=(1,0,0)
   // F0 = mix(F0, surfaceColor.rgb, metalness);
   // float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
  //  //return float4(normalize(N.xyz), 1);
  //
  //  float3 lightDir = normalize(float3(0, 0, -1));
  //  float cosTheta = dot(lightDir, N);
  //  float F0 = (0.04, 0.04, 0.04);
  //  float3 baseColor = float3(0.5, 0.5, 0.5);
  //  //float2 brickUV = input.uv * float2(10, 10);
  // // float2 uv = input.uv;
  //
  //  //float3 texNormal = normal(brickUV) * 2.0 - 1.0;
  //
  //  //float3x3 TBN = float3x3(T, B, N);
  //  //float3 finalNormal = mul(texNormal, TBN);
  //  //finalNormal = N;
  // // float3x3 vm = (float3x3)view[0];
  //  //finalNormal = mul(finalNormal,vm);
  //
  //  float3 N_color = B * 0.5 + 0.5;
  //  //float3 B_color = B * 0.5 + 0.5;
  //  //float3 T_color = T * 0.5 + 0.5;
  //
  //  //float3 baseColor = color(brickUV);
  //
  //  float3 pos = input.wpos.xyz;
  //  float4x4 invView = saturate(view[0]);
  //  float3 cameraPos = invView._m03_m13_m23.xyz;
  //  //cameraPos.x = cameraPos+x-6;
  //  //cameraPos.y = cameraPos + y-3;
  //  float3 lightColor=(1, 1, 1);
  //  //float3 lightPos = normalize(float3(0, 1, 0));
  //  float distance = length(lightDir - pos);
  //  float attenuation = 1.0 / (distance * distance);
  //  float roughness =  1- SinglePos.y/10;
  //  float3 radiance = lightColor * attenuation;
  //
  //  float3 L = normalize(lightDir - pos);
  //  float3 V = normalize(cameraPos - pos);
  //
  //  float3 H = normalize(L + V);
  // // float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
  //  float metallic = SinglePos%2;
  //
  //  float3 ref = reflect(V, N);
  //  float3 env = sfMap(ref);
  //  //float3 env = sfMap(N);
  //  float roug_sqr = roughness * roughness;
  //  //float3 G = CookTorrance_GGX(N, L, V ,roughness,F0, metallic);
  //  float3 G = CookTorrance_GGX(N, lightDir, V, 0, 1, 1);
  //  float3 OutColor =  G;
  // // float3 p = CookTorrance_GGX(N, L, V, roughness, F0);
  //  
  //
  //  OutColor = dot(N, lightDir);
  //
  //  
  //  //return float4(frac(input.uv * 8), 0, 1);
  //  
  //  //return float4(N_color,1);
  //
  //
  //
  ////  return float4(p,p,p, 1.0);
  //  return float4(N / 2 + .5, 1.0);

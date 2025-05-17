Texture2D shadowMap : register(t0);
SamplerState samp : register(s0);

struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float4 PS(VS_OUTPUT input) : SV_Target
{
    float depth = shadowMap.Sample(samp, input.uv).r;

// Визуализируем глубину в оттенках серого
return float4(depth, depth, depth, 1.0);
}
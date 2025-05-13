struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

Texture2D renderTexture : register(t0);
SamplerState samplerState : register(s0);

float4 PS(VS_OUTPUT input) : SV_Target
{
    //return float4(1,1,0,1);
    return float4(renderTexture.SampleLevel(samplerState, input.uv,0).rgb,1);
}
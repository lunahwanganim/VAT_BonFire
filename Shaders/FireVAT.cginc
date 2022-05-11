

#include "../Libraries/Quaternion.cginc"
#include "../Libraries/header.cginc"
void FireVAT_float(uint ResX, uint ResY, uint NumTriangle, 
                UnityTexture2D VAT0, UnityTexture2D VAT1, 
                UnitySamplerState SS, float FPS, float3 CamPosition,
                float2 uv, float2 uv2, float Time, uint NumFrame,
                out float3 OutPosition, out float OutHeat, out float OutEmission, out float OutAlpha, out float3 OutNormal)
{
    OutPosition = float3(0, 0, 0);
    OutHeat = 0;
    OutEmission = 0;
    OutAlpha = 0;
    OutNormal = float3(0, 0, 0);

    //uint     triangle_id = (uv.x + .00005) * 10000;
    uint     triangle_id = floor(uv.x * 100) * 100 + floor(uv.y  * 100);
    uint     vtx_id = floor((uv2.x + .05) * 2);

    uint     frame = Time * FPS;
    frame = frame % NumFrame;

    //frame = 100;


    uint     pixel_id = frame * NumTriangle * 3 + triangle_id * 3 + vtx_id;

    uint     i_x = pixel_id % ResX;
    uint     i_y = pixel_id / ResX;
    float   u = (i_x + .5) / (ResX);
    float   v = (i_y + .5) / (ResY);


    float2 uv_vat = float2(u, v);
    //float4  vat_0 = SAMPLE_TEXTURE2D_LOD(VAT0, SS, uv_vat, 0.0);
    
    float4  vat_0 =  VAT0.Load(uint3(i_x, i_y, 0));

    float3  pos_vat = float3(-vat_0.x, vat_0.y, vat_0.z);



    float4  vat_1 = VAT1.Load(uint3(i_x, i_y, 0));
    float   alpha = vat_1.w;
    float   normal_packed = vat_1.x;
    float   emission = vat_1.y;
    float   heat = vat_1.z;


    // decode float to float2 
    normal_packed *= 1023.0;
    float2 f2;
    f2.x = floor(normal_packed * 0.03125) * 0.03226; // 0.03125 <- 1/32.0,  0.03226 <- 1/31.0 
    f2.y = (normal_packed - (floor(normal_packed * 0.03125)) * 32.0) * 0.03175;   //  0.03125 <- 1/32.0,  0.03175 <- 1/31.5   
    // decode float2 to float3 
    float3  f3;
    f2 *= 4.0;
    f2 -= 2.0;
    float f2dot = dot(f2, f2);
    f3 = float3(float2( sqrt(1 - f2dot * .25) * f2), 0);
    f3.z = 1 - (f2dot * .5);
    f3 = clamp(f3, -1, 1); // I don't know why we need this.  But this makes the output better. 
    float3 normal = normalize(f3);


    float3 view_dir = normalize(CamPosition - pos_vat);

    float   fresnel = abs(dot(view_dir, normal));
    fresnel *= fresnel;
    //fresnel = Fit(fresnel, 0, 1, )



    OutNormal = normal;
    OutEmission = emission;
    OutAlpha = alpha * fresnel;
    OutPosition = pos_vat;
    OutHeat = heat;

}  

#ifndef __NNS_INCLUDED__
#define __NNS_INCLUDED__

float3  ClosestPointOnTriangle(float3 P, float3 A, float3 B, float3 C){ // triangle ABC
    float3  AB = B - A;
    float3  AC = C - A;
    // Check if P in vertex region outside A 
    float3  AP = P - A;
    float   d1 = dot(AB, AP);
    float   d2 = dot(AC, AP);  
    if (d1 <= 0.0 && d2 <= 0.0) return A; // barycentric coordinates (1,0,0)
    // Check if P in vertex region outside B 
    float3 BP = P - B; 
    float d3 = dot(AB, BP); 
    float d4 = dot(AC, BP); 
    if (d3 >= 0.0f && d4 <= d3) return B; // barycentric coordinates (0,1,0)
    // Check if P in edge region of AB, if so return projection of P onto AB
    float VC = d1 * d4 - d3 * d2; 
    if (VC <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { 
        float v = d1 / (d1 - d3); 
        return A + v * AB; // barycentric coordinates (1 - v, v, 0) 
    }
    // Check if P in vertex region outside C
    float3 CP = P - C; 
    float d5 = dot(AB, CP); 
    float d6 = dot(AC, CP); 
    if (d6 >= 0.0 && d5 <= d6) return C; // barycentric coordinates (0,0,1)
    // Check if P in edge region of AC, if so return projection of P onto AC
    float VB = d5 * d2 - d1 * d6; 
    if (VB <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { 
        float w = d2 / (d2 - d6); 
        return A + w * AC; // barycentric coordinates (1 - w, 0, w) 
    }
    // Check if P in edge region of BC, if so return projection of P onto BC
    float VA = d3 * d6 - d5 * d4; 
    if (VA <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) { 
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); 
        return B + w * (C - B); // barycentric coordinates (0, 1 - w, w) 
    }  
    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    float denom = 1.0 / (VA + VB + VC);
    float v = VB * denom;
    float w = VC * denom;
    return A + AB * v + AC * w; // = u*a + v*b + w*c, u = va * denom = 1.0f-v-w
}
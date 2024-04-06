#pragma once
#include "common.h"
#include "Texture.h"

class Material : public BaseClass {
public:
    Vector3 Kd;
    Vector3 Ks;
    Vector3 Tr;
    float Ns{ 0.0f };
    float Ni{ 0.0f };
    float F0{ 0.04f };
    Texture* map_Kd{ nullptr };

    bool isEmissive{ false };
    Vector3 Ke;

    Material() = default;
    ~Material() = default;

    CUDA_DEVICE Vector3 brdf(const Vector3& wi, const Vector3& wo, const Vector3& normal, const Vector2& uv) const;

    CUDA_DEVICE void sampleBRDF(const Vector3& wi, const Vector3& normal, const Vector2& uv, Vector3& brdf, Vector3& wo, float& pdf, CURAND_STATE_T_PTR rand_state) const;

    CUDA_DEVICE Vector3 getEmission(const Vector2& uv = Vector2(0.0f, 0.0f)) const
    {
        // TODO: may have texture
        return Ke;
    }

    CUDA_DEVICE Vector3 getSpecular(const Vector2& uv = Vector2(0.0f, 0.0f)) const
    {
        // TODO: may have texture
        return Ks;
    }

    CUDA_DEVICE Vector3 getDiffuse(const Vector2& uv = Vector2(0.0f, 0.0f)) const
    {
        // TODO: may have texture
        return Vector3(1.0f) - Ks;
    }

    CUDA_DEVICE float getRoughness(const Vector2& uv = Vector2(0.0f, 0.0f)) const
    {
        // TODO: may have texture
        return sqrtf(2.0f/(Ns+2.0f));
    }

    CUDA_DEVICE Vector3 getAlbedo(const Vector2& uv = Vector2(0.0f, 0.0f)) const
    {
        // TODO: may have texture
        return Kd;
    }

    CUDA_DEVICE Vector3 getF0(const Vector2& uv = Vector2(0.0f, 0.0f)) const
    {
        // TODO: may have texture
        if (Tr.x < 1.0f || Tr.y < 1.0f || Tr.z < 1.0f)
        {
            Vector3 ior = Vector3(Ni);
            return (ior - 1) / (ior + 1);
        }
        else
        {
            return Vector3(1.0f);
        }
    }

    CUDA_DEVICE float D_GGX(float cosNH, float roughness) const
    {
        float a2 = roughness * roughness;
        float f = (cosNH * cosNH) * (a2 - 1) + 1;
        return a2 / (PI * f * f);
    }

    CUDA_DEVICE float G_GGX(float cosNV, float k) const
    {
        return cosNV / (cosNV * (1 - k) + k);
    }

    CUDA_DEVICE float G_Schlicksmith(float cosNV, float cosNL, float roughness) const
    {
        float k = (roughness + 1) * (roughness + 1) / 8;
        return G_GGX(cosNV, k) * G_GGX(cosNL, k);
    }

    CUDA_DEVICE Vector3 F_Schlick(float cosVH, Vector3 F0) const
    {
        return F0 + (1.0f - F0) * powf(1 - cosVH, 5);
    }
};

CUDA_DEVICE Vector3 Material::brdf(const Vector3& wi, const Vector3& wo, const Vector3& normal, const Vector2& uv) const
{
    // TODO: transparency, metallic, etc.
    Vector3 Kd = getDiffuse(uv);
    Vector3 Ks = getSpecular(uv);
    float roughness = getRoughness(uv);
    Vector3 albedo = getAlbedo(uv);
    Vector3 F0 = getF0(uv);

    // compute brdf
    Vector3 wh = (wi+wo).normalized();
    float cosNL = wo.dot(normal);
    float cosNV = wi.dot(normal);
    float cosVH = wi.dot(wh);
    float cosNH = wh.dot(normal);
    float D = D_GGX(cosNH, roughness);
    float G = G_Schlicksmith(cosNV, cosNL, roughness);
    Vector3 F = F_Schlick(cosVH, F0);

    Vector3 brdf_diffuse = albedo / PI;
    Vector3 brdf_specular = F * D * G / (4 * cosNL * cosNV);
    return Kd * brdf_diffuse + Ks * brdf_specular;
    //return Kd * brdf_diffuse;
    //return Ks * brdf_specular;
}

CUDA_DEVICE void Material::sampleBRDF(const Vector3& wi, const Vector3& normal, const Vector2& uv, Vector3& brdf, Vector3& wo, float& pdf, CURAND_STATE_T_PTR rand_state) const
{
    // TODO: transparency, metallic, etc.
    float roughness = getRoughness(uv);

    // sample wh by GGX distribution
    float rand1 = curand_uniform(rand_state);
    float theta = acosf(sqrtf((1 - rand1) / (rand1 * (roughness * roughness - 1) + 1)));
    float phi = 2 * PI * curand_uniform(rand_state);
    Vector3 wh = Vector3::fromSpherical(theta, phi).toWorld(normal, wi);

    // compute wo
    wo = wi.reflect(wh);
    if (wo.dot(normal) < 0)
    {
        pdf = 1.0f;
        brdf = Vector3(0.0f);
        return;
    }

    // compute pdf
    float cosNH = wh.dot(normal);
    float sinNH  = sqrtf(1 - cosNH * cosNH);
    float a2 = roughness * roughness;
    float f = (a2 - 1.0f)*cosNH*cosNH + 1.0f;
    pdf = (a2 * cosNH * sinNH) / (f * f) / (4.0f * wi.dot(wh));

    // compute brdf
    brdf = this->brdf(wi, wo, normal, uv);
}


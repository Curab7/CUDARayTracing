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
    Texture* map_Kd{ nullptr };

    bool isEmissive{ false };
    Vector3 Ke;

    Material() = default;
    ~Material() = default;

    CUDA_DEVICE Vector3 getEmission(const Vector2& uv = Vector2(0.0f, 0.0f)) const
    {
        // TODO: may have texture
        return Ke;
    }

    CUDA_DEVICE Vector3 brdf(const Vector3& wi, const Vector3& wo, const Vector3& normal, const Vector2& uv) const;
};

CUDA_DEVICE Vector3 Material::brdf(const Vector3& wi, const Vector3& wo, const Vector3& normal, const Vector2& uv) const
{
    return Kd /(2 * PI);
}

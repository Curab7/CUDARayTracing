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
};
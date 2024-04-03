#pragma once

#include "common.h"

class Vector3;
class Vector2;
class SolidObject;

class Intersection
{
public:
    bool hit{ false };
    bool sense{ false };
    float t{ 0.0f };
    Vector2 uv;
    Vector3 point;
    Vector3 normal;
    SolidObject* object{ nullptr };

    CUDA_CALLABLE Intersection() {}
};
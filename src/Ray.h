#pragma once
#include "common.h"
#include "Math.h"

class Ray
{
public:
    Vector3 origin;
    Vector3 direction;
    Vector3 invDirection;
    float tMin{ 0.0f };
    float tMax{ INF };

    CUDA_CALLABLE Ray(const Vector3& origin, const Vector3& direction, float tMin = 0.0f, float tMax = INF)
        : origin(origin), direction(direction), tMin(tMin), tMax(tMax)
    {
        for(int i = 0; i < 3; i++)
        {
            if(direction[i] == 0.0f)
            {
                invDirection[i] = 0.0f;
            }
            else
            {
                invDirection[i] = 1.0f / direction[i];
            }
        }
    }

    CUDA_CALLABLE Ray(const Ray& ray)
        : origin(ray.origin), direction(ray.direction), invDirection(ray.invDirection), tMin(ray.tMin), tMax(ray.tMax)
    {}
};
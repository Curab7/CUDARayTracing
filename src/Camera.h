#pragma once
#include "Math.h"
#include "Ray.h"

class Camera : public BaseClass
{
public:
    Vector3 position;
    Vector3 direction;
    Vector3 up;
    float fov;  //radius
    float aspectRatio;

    int width{ 0 };
    int height{ 0 };

    Camera(const Vector3& position, const Vector3& direction, const Vector3& up, float fov, int width, int height)
        : position(position), direction(direction), up(up),
        fov(DEG2RAD(fov)), aspectRatio((float)width / (float)height), width(width), height(height)
    {
    };

    // [0, 1] x [0, 1]
    CUDA_CALLABLE Ray generateRay(float x, float y) const
    {
        CUDA_LOG("start generating ray at (%f, %f)\n", x, y);
        float scale = tan(fov / 2.0f);
        float xOffset = (x * 2.0f - 1.0f) * scale * this->aspectRatio;
        float yOffset = (1.0f - y * 2.0f) * scale;
        Vector3 right = (direction.cross(up)).normalized();
        Vector3 target = position + direction + right * xOffset + up * yOffset;
        Vector3 lightDir = (target - position).normalized();
        float tmin = 0.001f;
        float tmax = 100000.0f;
        CUDA_LOG("prepared ray at (%f, %f)\n", x, y);
        return Ray(position, lightDir, tmin, tmax);
    }
};
#pragma once
#include "Math.h"

class AABB : public BaseClass {
public:
    Vector3 pointMin;
    Vector3 pointMax;

    AABB() {}
    AABB(const Vector3& center) : pointMin(center), pointMax(center) {}
    AABB(const Vector3& pointMin, const Vector3& pointMax) : pointMin(pointMin), pointMax(pointMax) {}

    Vector3 getCenter() const { return (pointMin + pointMax) * 0.5f; }

    int longestAxis() const
    {
        int axis = 0;
        float maxLen = (pointMax.x - pointMin.x);
        if ((pointMax.y - pointMin.y) > maxLen)
        {
            axis = 1;
            maxLen = (pointMax.y - pointMin.y);
        }
        if ((pointMax.z - pointMin.z) > maxLen)
        {
            axis = 2;
        }
        return axis;
    }

    AABB unionWith(const AABB& other)
    {
        float minX = std::min(pointMin.x, other.pointMin.x);
        float minY = std::min(pointMin.y, other.pointMin.y);
        float minZ = std::min(pointMin.z, other.pointMin.z);
        float maxX = std::max(pointMax.x, other.pointMax.x);
        float maxY = std::max(pointMax.y, other.pointMax.y);
        float maxZ = std::max(pointMax.z, other.pointMax.z);
        return AABB(Vector3(minX, minY, minZ), Vector3(maxX, maxY, maxZ));
    }

    CUDA_CALLABLE bool intersect(const Ray& ray, float& tin, float& tout) const;
};

CUDA_CALLABLE bool AABB::intersect(const Ray& ray, float& tin, float& tout) const
{
    //CUDA_LOG("AABB::intersect\n");
    float tmin[3];
    float tmax[3];
    for (int i = 0; i < 3; i++)
    {
        if (ray.direction[i] == 0)
        {
            if (ray.origin[i] > pointMin[i] && ray.origin[i] < pointMax[i])
            {
                tmin[i] = -INF;
                tmax[i] = INF;
            }
            else
            {
                return false;
            }
        }
        else
        {
            tmin[i] = (pointMin[i] - ray.origin[i]) * ray.invDirection[i];
            tmax[i] = (pointMax[i] - ray.origin[i]) * ray.invDirection[i];
            if (tmin[i] > tmax[i])
            {
                float t = tmin[i];
                tmin[i] = tmax[i];
                tmax[i] = t;
            }
        }
    }
    tin = MAX(MAX(tmin[0], tmin[1]), tmin[2]);
    tout = MIN(MIN(tmax[0], tmax[1]), tmax[2]);
    return tin <= tout + TOLERANCE && tout > ray.tMin && (tin < ray.tMax || tout < ray.tMax);
}
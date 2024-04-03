#pragma once
#include "Object.h"

class Triangle : public SolidObject
{
public:
    float* verticesPool;
    float* normalPool;
    float* uvPool;
    int idxV0, idxV1, idxV2;
    int idxN0, idxN1, idxN2;
    int idxUV0, idxUV1, idxUV2;

    Triangle(): SolidObject(ObjectType::Triangle) {}

    Triangle(float* verticesPool, int idxV0, int idxV1, int idxV2,
             float* normalPool, int idxN0, int idxN1, int idxN2,
             float* uvPool, int idxUV0, int idxUV1, int idxUV2, const Material* material=nullptr)
        : SolidObject(ObjectType::Triangle),
          verticesPool(verticesPool), normalPool(normalPool), uvPool(uvPool),
          idxV0(idxV0), idxV1(idxV1), idxV2(idxV2),
          idxN0(idxN0), idxN1(idxN1), idxN2(idxN2),
          idxUV0(idxUV0), idxUV1(idxUV1), idxUV2(idxUV2)
    {
        this->material = material;
    }

    AABB getAABB() const override
    {
        AABB aabb;
        Vector3 v0 = Vector3(verticesPool, idxV0);
        Vector3 v1 = Vector3(verticesPool, idxV1);
        Vector3 v2 = Vector3(verticesPool, idxV2);
        aabb.pointMin = v0.min(v1).min(v2);
        aabb.pointMax = v0.max(v1).max(v2);
        return aabb;
    }

    CUDA_CALLABLE bool intersect(const Ray& ray, Intersection& intersection) const;
};



CUDA_CALLABLE bool Triangle::intersect(const Ray& ray, Intersection& intersection) const
{
    CUDA_LOG("Triangle::intersect\n");
    Vector3 v0 = Vector3(verticesPool, idxV0);
    Vector3 v1 = Vector3(verticesPool, idxV1);
    Vector3 v2 = Vector3(verticesPool, idxV2);
    Vector3 e1 = v1 - v0;
    Vector3 e2 = v2 - v0;
    Vector3 s = ray.origin - v0;                  // S = O - P0
    Vector3 s1 = ray.direction.cross(e2);         // S1 = D cross E2
    Vector3 s2 = s.cross(e1);                     // S2 = S cross E1
    float denominator = s1.dot(e1);               // S1 dot E1

    if (ISZERO(denominator))
    {
        return false;
    }

    float invDenominator = 1.0f / denominator;

    float b1 = s1.dot(s) * invDenominator;
    if (b1 < 0.0f || b1 > 1.0f) {
        return false;
    }

    float b2 = s2.dot(ray.direction) * invDenominator;
    if (b2 < 0.0f || b2 > 1.0f) {
        return false;
    }

    float b0 = 1 - b1 - b2;
    if (b0 < 0.0f || b0 > 1.0f) {
        return false;
    }

    float t = s2.dot(e2) * invDenominator;
    if (t <ray.tMin || t > ray.tMax || (intersection.hit && t > TOLERANCE + intersection.t))
    {
        return false;
    }

    Vector3 n0 = Vector3(normalPool, idxN0);
    Vector3 n1 = Vector3(normalPool, idxN1);
    Vector3 n2 = Vector3(normalPool, idxN2);

    intersection.hit = true;
    intersection.t = t;
    intersection.uv = Vector2(b1, b2);
    intersection.point = ray.origin + ray.direction * t;
    intersection.normal = (n0 * b0 + n1 * b1 + n2 * b2).normalized();
    intersection.sense = ray.direction.dot(intersection.normal) > 0.0f;
    intersection.object = (SolidObject*)this;

    return true;
}
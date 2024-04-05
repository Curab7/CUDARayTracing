#pragma once

#include "Math.h"
#include "Object.h"
#include "BVH.h"
#include <vector>

class Scene : public BaseClass
{
public:
    std::vector<Object*> objects;

#if USE_CUDA
    Object** notBVH{ nullptr };
    int numObjects{ 0 };
#endif

    BVH* bvh{ nullptr };


    int maxDepth{ 4 };
    float RussianRoulette { 0.8 };
    Vector3 background{ 0.1f, 0.1f, 0.1f };

    Scene() {}

    ~Scene();

    void addObject(Object* object) { objects.push_back(object); }

    void buildBVH();

    CUDA_CALLABLE bool intersect(const Ray& ray, Intersection& isec) const;

    // 'state' is useless if not use CUDA
    CUDA_DEVICE float getRandomFloat(CURAND_STATE_T_PTR state) const;
    CUDA_DEVICE Vector3 castRay(const Ray& ray, int depth, CURAND_STATE_T_PTR state) const;
    CUDA_DEVICE void sampleMirror(const Intersection& isec, const Vector3& wi, Vector3& wo, float& invPdf, CURAND_STATE_T_PTR state) const;
    CUDA_DEVICE void sampleUniform(const Intersection& isec, const Vector3& wi, Vector3& wo, float& invPdf, CURAND_STATE_T_PTR state) const;
};

Scene::~Scene() {
#if USE_CUDA
    if (!USE_BVH)
    {
        CUDA_FREE(notBVH);
    }
#endif
    delete bvh;
}

void Scene::buildBVH() {
    if (USE_BVH)
    {
        std::cout << "Building scene BVH..." << std::endl;
        bvh = new BVH(objects);
    }
    else
    {
#if USE_CUDA
        numObjects = objects.size();
        CUDA_MALLOC(notBVH, numObjects * sizeof(Object*), Object*);
        for (int i = 0; i < numObjects; i++)
        {
            notBVH[i] = objects[i];
        };
#endif
    }
}

CUDA_CALLABLE bool Scene::intersect(const Ray& ray, Intersection& isec) const
{
    if (!bvh)
    {
#if USE_CUDA
        CUDA_LOG("Inserting scene without BVH...\n");
        for (int i = 0; i < numObjects; i++)
        {
            notBVH[i]->intersect(ray, isec);
        }
#else
        for (Object* object : objects)
        {
            object->intersect(ray, isec);
        }
#endif

    }
    else
    {
        CUDA_LOG("Inserting scene with BVH...\n");
        bvh->intersect(ray, isec);
    }
    return isec.hit;
}

CUDA_DEVICE Vector3 Scene::castRay(const Ray& ray, int depth, CURAND_STATE_T_PTR state) const
{
    Vector3 result(0.0f, 0.0f, 0.0f);

    if (depth > maxDepth)
        return result;

    Intersection isec;

    if (!intersect(ray, isec))
    {
        return background;
    }

    CUDA_LOG("Dealing with intersection at %f %f %f\n", isec.point.x, isec.point.y, isec.point.z);

    SolidObject* hitObject = isec.object;
    Vector3 point = isec.point;
    Vector3 normal = isec.normal;
    Vector2 uv = isec.uv;
    const Material* mtr = hitObject->material;

    if (mtr->isEmissive)
    {
        result += mtr->getEmission(uv);
    }

    // return normal
    result += normal * 0.5f + 0.5f;

    Vector3 wi = -ray.direction;
    Vector3 wo;
    float invPdf;
    sampleMirror(isec, wi, wo, invPdf, state);

    Vector3 brdf = mtr->brdf(wi, wo, normal, uv);
    Vector3 radiance = castRay(Ray(point, wo), depth + 1, state);
    //Vector3 radiance = Vector3(0.5f, 0.5f, 0.5f);
    float cosTheta = wo.dot(normal);

    result += brdf * radiance * cosTheta * invPdf;

    return result;
}

#if USE_CUDA
CUDA_DEVICE float Scene::getRandomFloat(CURAND_STATE_T_PTR state) const
{
    return curand_uniform(state);
}
#else
CUDA_DEVICE float Scene::getRandomFloat(CURAND_STATE_T_PTR state) const
{
    static std::mt19937 rng(0);
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}
#endif

CUDA_DEVICE void Scene::sampleMirror(const Intersection& isec, const Vector3& wi, Vector3& wo, float& invPdf, CURAND_STATE_T_PTR state) const
{
    Vector3 vecZ = isec.normal;
    Vector3 vecY = -wi.cross(vecZ).normalized();
    Vector3 vecX = vecY.cross(vecZ).normalized();
    wo = -wi.dot(vecX) * vecX + wi.dot(vecZ) * vecZ;
    invPdf = 1.0f;
}

CUDA_DEVICE void Scene::sampleUniform(const Intersection& isec, const Vector3& wi, Vector3& wo, float& invPdf, CURAND_STATE_T_PTR state) const
{
    Vector3 vecZ = isec.normal;
    Vector3 vecY = -wi.cross(vecZ).normalized();
    Vector3 vecX = vecY.cross(vecZ).normalized();
    double phi = acosf(1 - getRandomFloat(state));
    double theta = 2 * PI * getRandomFloat(state);
    Vector3 localOut = Vector3(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta), cosf(phi));
    wo = localOut.x * vecX + localOut.y * vecY + localOut.z * vecZ;
    invPdf = 2 * PI;
}
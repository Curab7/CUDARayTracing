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
    float RussianRoulette { 0.8f };
    Vector3 background{ 0.0f, 0.0f, 0.0f };

    Scene() {}

    ~Scene();

    void addObject(Object* object) { objects.push_back(object); }

    void buildBVH();

    CUDA_CALLABLE bool intersect(const Ray& ray, Intersection& isec) const;

    // 'state' is useless if not use CUDA
    CUDA_DEVICE float getRandomFloat(CURAND_STATE_T_PTR state) const;
    CUDA_DEVICE Vector3 castRay(const Ray& ray, int depth, CURAND_STATE_T_PTR state) const;
    CUDA_DEVICE void sampleMirror(const Intersection& isec, const Vector3& wi, Vector3& wo, float& pdf, CURAND_STATE_T_PTR state) const;
    CUDA_DEVICE void sampleUniform(const Intersection& isec, const Vector3& wi, Vector3& wo, float& pdf, CURAND_STATE_T_PTR state) const;
    CUDA_DEVICE void sampleCos(const Intersection& isec, const Vector3& wi, Vector3& wo, float& pdf, CURAND_STATE_T_PTR state) const;
    CUDA_DEVICE void sampleBRDF(const Intersection& isec, const Vector3& wi, Vector3& wo, float& pdf, CURAND_STATE_T_PTR state) const;
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

    if (depth > 1 && getRandomFloat(state) > RussianRoulette)
    {
        return result;
    }

    Intersection isec;

    if (!intersect(ray, isec))
    {
        return background;
    }

    CUDA_LOG("Dealing with intersection at %f %f %f\n", isec.point.x, isec.point.y, isec.point.z);

    Vector3 point = isec.point;
    Vector3 normal = isec.normal;
    Vector2 uv = isec.uv;
    const MaterialBase* mtr = isec.object->material;

    result += mtr->emission;

    Vector3 wi = -ray.direction;
    Vector3 wo;
    float pdf;
    Vector3 brdf;

    //sampleCos(isec, wi, wo, pdf, state);
    sampleBRDF(isec, wi, wo, pdf, state);
    brdf = mtr->brdf(wi, wo, normal, uv);

    //return brdf;
    //return wo * 0.5f + 0.5f;
    //return (wi + wo).normalized() * 0.5f + 0.5f;

    if (brdf.x == 0.0f && brdf.y == 0.0f && brdf.z == 0.0f)
    {
        return result;
    }

    Vector3 radiance = castRay(Ray(point, wo), depth + 1, state);

    //if (depth == 2)
    //{
    //    float theta = wo.dot(normal);
    //    //printf("theta: %f\n", theta);
    //    //Vector3 light = brdf * radiance * wo.dot(normal);
    //    //printf("light: %f %f %f\n", light.x, light.y, light.z);
    //    printf("brdf: %f %f %f, radiance: %f %f %f, theta: %f\n", brdf.x, brdf.y, brdf.z, radiance.x, radiance.y, radiance.z, theta);
    //}

    result += brdf * radiance * wo.dot(normal) / pdf / RussianRoulette;

    //printf("result: %f %f %f\n", result.x, result.y, result.z);

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

CUDA_DEVICE void Scene::sampleMirror(const Intersection& isec, const Vector3& wi, Vector3& wo, float& pdf, CURAND_STATE_T_PTR state) const
{
    wo = wi.reflect(isec.normal);
    pdf = 1.0f;
}

CUDA_DEVICE void Scene::sampleUniform(const Intersection& isec, const Vector3& wi, Vector3& wo, float& pdf, CURAND_STATE_T_PTR state) const
{
    double theta = acosf(1 - getRandomFloat(state));
    double phi = 2 * PI * getRandomFloat(state);
    wo = Vector3::fromSpherical(theta, phi).toWorld(isec.normal, wi);
    pdf = 1.0f / (2 * PI);
}

CUDA_DEVICE void Scene::sampleCos(const Intersection& isec, const Vector3& wi, Vector3& wo, float& pdf, CURAND_STATE_T_PTR state) const
{
    double theta = acosf(sqrtf(1 - getRandomFloat(state)));
    double phi = 2 * PI * getRandomFloat(state);
    wo = Vector3::fromSpherical(theta, phi).toWorld(isec.normal, wi);
    pdf = cosf(theta) / PI;
}

CUDA_DEVICE void Scene::sampleBRDF(const Intersection& isec, const Vector3& wi, Vector3& wo, float& pdf, CURAND_STATE_T_PTR state) const
{
    const MaterialBase* mtr = isec.object->material;
    mtr->sampleBRDF(wi, isec.normal, isec.uv, wo, pdf, state);
}

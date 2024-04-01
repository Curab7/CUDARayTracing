#pragma once

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
    double RussianRoulette { 0.8 };
    Vector3 background{ 0.0f, 0.0f, 0.0f };

    Scene() {}

    ~Scene() {
#if USE_CUDA
        CUDA_FREE(notBVH);
#endif
        delete bvh;
    }

    void addObject(Object* object) { objects.push_back(object); }

    void buildBVH() {
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

    CUDA_CALLABLE Vector3 castRay(const Ray& ray, int depth) const;
};


CUDA_CALLABLE Vector3 Scene::castRay(const Ray& ray, int depth) const
{
    Vector3 result(0.0f, 0.0f, 0.0f);

    if (depth > maxDepth)
        return result;

    Intersection isec;

    if (!bvh)
    {
#if USE_CUDA
        CUDA_LOG("Traversing scene without BVH...\n");
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
        CUDA_LOG("Traversing scene with BVH...\n");
        bvh->intersect(ray, isec);
    }

    if (!isec.hit)
    {
        return background;
    }

    CUDA_LOG("Dealing with intersection at %f %f %f\n", isec.point.x, isec.point.y, isec.point.z);

    SolidObject* hitObject = isec.object;
    Vector3 point = isec.point;
    Vector3 normal = isec.normal;
    const Material* mtr = hitObject->material;

    result = normal * 0.5f + 0.5f;

    return result;
}
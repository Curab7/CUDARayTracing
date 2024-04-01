#pragma once
#include "Object.h"
#include "Triangle.h"
#include "BVH.h"
#include <vector>

class Mesh : public Object
{
public:
    std::vector<Triangle*> triangles;
#if USE_CUDA
    Triangle** notBVH{ nullptr };
    int numTriangles{ 0 };
#endif

    BVH *bvh{nullptr};

    Mesh(std::vector<Triangle*> triangles) :Object(ObjectType::Mesh), triangles(triangles) {

#if USE_CUDA
        numTriangles = triangles.size();
        CUDA_MALLOC(notBVH, numTriangles * sizeof(Triangle*), Triangle*);
        for (int i = 0; i < numTriangles; i++)
        {
            this->notBVH[i] = triangles[i];
        }
#endif

        if (USE_BVH)
        {
            std::vector<Object*> objects;
            for (Triangle* triangle : triangles) {
                objects.push_back(triangle);
            }
            bvh = new BVH(objects);
        }
    }

    ~Mesh() {
#if USE_CUDA
        CUDA_FREE(notBVH);
#endif
        delete bvh;
    }

    AABB getAABB() const override { return bvh->getAABB(); };

    CUDA_CALLABLE bool intersect(const Ray& ray, Intersection& intersection) const;
};

CUDA_CALLABLE bool Mesh::intersect(const Ray& ray, Intersection& intersection) const
{
    CUDA_LOG("Mesh::intersect\n");
    if (!bvh)
    {
        bool result = false;
#if USE_CUDA
        for (int i=0; i<numTriangles; i++)
        {
            result |= notBVH[i]->intersect(ray, intersection);
        }
#else
        for (Triangle* triangle : triangles) {
            result |= triangle->intersect(ray, intersection);
        }
#endif
        return result;
    }
    else
    {
        return bvh->intersect(ray, intersection);
    }
}

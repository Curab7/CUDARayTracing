#include "Loader.h"
#include "Render.h"

int main(int argc, char* argv[])
{
    size_t stackSize;
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, STACK_SIZE));
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    printf("Stack size: %zu\n", stackSize);

    //std::string name = "cornell-box";
    std::string name = "veach-mis";
    //std::string name = "stairscase";

    Loader loader("../scenes/"+name);
    loader.loadCoursePack(name);

    Scene* scene = loader.getScene();
    Camera* camera = loader.getCamera();

    Render render(scene, camera, 64, 6);
    render.render();

    render.saveImage("../output/" + name + ".png");

    return 0;
}

CUDA_CALLABLE bool Object::intersect(const Ray& ray, Intersection& intersection)
{
    //CUDA_LOG("Object::intersect, type = %d\n", type);
    switch (type)
    {
    case ObjectType::Mesh:
    {
        const Mesh* mesh = static_cast<const Mesh*>(this);
        return mesh->intersect(ray, intersection);
    }
    case ObjectType::Triangle:
    {
        const Triangle* triangle = static_cast<const Triangle*>(this);
        return triangle->intersect(ray, intersection);
    }
    case ObjectType::BVHNode:
    {
        const BVHNode* bvhNode = static_cast<const BVHNode*>(this);
        return bvhNode->intersect(ray, intersection);
    }
    case ObjectType::BVH:
    {
        const BVH* bvh = static_cast<const BVH*>(this);
        return bvh->intersect(ray, intersection);
    }
    default:
        return false;
    }
}
#include "Loader.h"
#include "Render.h"

int main()
{
    //std::string name = "cornell-box";
    std::string name = "veach-mis";
    //std::string name = "stairscase";

    Loader loader("D:/Code/MyProjects/RayTracing/RayTracing/scenes/"+name +"/");
    loader.loadCoursePack(name);

    Scene* scene = loader.getScene();
    Camera* camera = loader.getCamera();

    Render render(scene, camera, 1, 1);
    render.render();

    render.saveImage(name + ".png");

    return 0;
}

#include "Object.h"
#include "Mesh.h"
#include "Triangle.h"
#include "BVH.h"


CUDA_CALLABLE bool Object::intersect(const Ray& ray, Intersection& intersection)
{
    CUDA_LOG("Object::intersect, type = %d\n", type);
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
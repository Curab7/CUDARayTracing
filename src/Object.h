#pragma once
#include "common.h"
#include "Math.h"
#include "Ray.h"
#include "Intersection.h"
#include "AABB.h"
#include "Material.h"

class Object;

enum class ObjectType
{
    None,
    Mesh,
    BVHNode,
    BVH,

    Triangle,
};

class Object : public BaseClass
{
public:
    ObjectType type;

    Object(ObjectType type = ObjectType::None) : type(type) {}
    virtual ~Object() {}

    CUDA_CALLABLE bool intersect(const Ray& ray, Intersection& intersection);
    virtual AABB getAABB() const = 0;
};

class SolidObject : public Object
{
public:
    const MaterialBase* material{ nullptr };

    SolidObject(ObjectType type = ObjectType::None) : Object(type) {}
    virtual ~SolidObject() {}

    virtual AABB getAABB() const = 0;
};
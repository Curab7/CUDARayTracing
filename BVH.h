#pragma once
#include "common.h"
#include "Object.h"
#include "Ray.h"
#include <vector>
#include <algorithm>


class BVHNode : public Object
{
public:
    Object* object{ nullptr };
    AABB aabb;
    BVHNode* left{ nullptr };
    BVHNode* right{ nullptr };

    BVHNode(Object* object, AABB aabb) : Object(ObjectType::BVHNode), object(object), aabb(aabb) {};
    ~BVHNode() { delete left; delete right; }

    AABB getAABB() const override { return aabb; }

    CUDA_CALLABLE bool intersect(const Ray& ray, Intersection& intersection) const;
};

class BVH : public Object
{
public:
    BVHNode* root{ nullptr };

    BVH(const std::vector<Object*>& objects);

    ~BVH();

    BVHNode* buildRecursively(std::vector<BVHNode*> nodes);

    AABB getAABB() const override { return root->aabb; }

    CUDA_CALLABLE bool intersect(const Ray& ray, Intersection& isec) const
    {
        CUDA_LOG("BVH::intersect\n");
        return root ? root->intersect(ray, isec) : isec.hit;
    };
};


BVH::BVH(const std::vector<Object*>& objects) : Object(ObjectType::BVH)
{
    std::vector<BVHNode*> nodes;

    for (Object* object : objects)
    {
        nodes.push_back(new BVHNode(object, object->getAABB()));
    }
    root = buildRecursively(nodes);
}

BVH::~BVH()
{
    delete root;
}

BVHNode* BVH::buildRecursively(std::vector<BVHNode*> nodes)
{
    if (nodes.empty())
    {
        return nullptr;
    }

    if (nodes.size() == 1)
    {
        return nodes[0];
    }
    else if (nodes.size() == 2)
    {
        BVHNode* node = new BVHNode(nullptr, nodes[0]->aabb.unionWith(nodes[1]->aabb));
        node->left = nodes[0];
        node->right = nodes[1];
        return node;
    }

    // 按最长轴分割
    AABB centerAABB(nodes[0]->aabb.getCenter());
    for (int i = 1; i < nodes.size(); i++)
    {
        centerAABB = centerAABB.unionWith(nodes[i]->aabb.getCenter());
    }
    switch (centerAABB.longestAxis())
    {
    case 0:
        std::sort(nodes.begin(), nodes.end(), [](auto f1, auto f2) {
            return f1->aabb.getCenter().x < f2->aabb.getCenter().x; });
        break;
    case 1:
        std::sort(nodes.begin(), nodes.end(), [](auto f1, auto f2) {
            return f1->aabb.getCenter().y < f2->aabb.getCenter().y; });
        break;
    case 2:
        std::sort(nodes.begin(), nodes.end(), [](auto f1, auto f2) {
            return f1->aabb.getCenter().z < f2->aabb.getCenter().z; });
        break;
    }

    auto mid = nodes.begin() + nodes.size() / 2;
    BVHNode* left = buildRecursively(std::vector<BVHNode*>(nodes.begin(), mid));
    BVHNode* right = buildRecursively(std::vector<BVHNode*>(mid, nodes.end()));

    BVHNode* node = new BVHNode(nullptr, left->aabb.unionWith(right->aabb));
    node->left = left;
    node->right = right;

    return node;
}

CUDA_CALLABLE bool BVHNode::intersect(const Ray& ray, Intersection& isec) const
{
    CUDA_LOG("BVHNode::intersect\n");
    float tmin = 0.0f;
    float tmax = INF;

    if (!aabb.intersect(ray, tmin, tmax))
    {
        return isec.hit;
    }

    if (isec.hit && isec.t + TOLERANCE < tmin)
    {
        return isec.hit;
    }

    if (object)
    {
        return object->intersect(ray, isec);
    }

    if (left)
    {
        printf("BVHNode::intersect left\n");
        left->intersect(ray, isec);
    }

    if (right)
    {
        printf("BVHNode::intersect right\n");
        right->intersect(ray, isec);
    }

    return isec.hit;
}
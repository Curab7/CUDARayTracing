#pragma once

#include "common.h"
#include "Scene.h"
#include "Camera.h"

#include <string>

#include "external/stb_image_write.h"


CUDA_CALLABLE void drawPixel(int x, int y, int width, int height, int SSAA, Scene* scene, Camera* camera, Vector3* frameBuffer)
{
    Vector3 color(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < SSAA; i++)
    {
        for (int j = 0; j < SSAA; j++)
        {
            float sx = (x + (i + 1.0f) / (SSAA + 1.0f)) / width;
            float sy = (y + (j + 1.0f) / (SSAA + 1.0f)) / height;

            CUDA_LOG("CUDA: Generating ray for pixel (%d, %d)\n", x, y);
            //Ray ray = camera->generateRay(sx, sy);
            Ray ray = camera->generateRay(sx, sy);
            CUDA_LOG("CUDA: Casting ray for pixel (%d, %d)\n", x, y);
            color = color + scene->castRay(ray, 0) / (float)(SSAA * SSAA);
            CUDA_LOG("CUDA: Got pixel (%d, %d)\n", x, y);

        }
    }

    frameBuffer[y * width + x] = color;
}

#if USE_CUDA
CUDA_GLOBAL void cudaRender(int width, int height, int SSAA, Scene* scene, Camera* camera, Vector3* frameBuffer)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    drawPixel(x, y, width, height, SSAA, scene, camera, frameBuffer);
}
#endif


class Render
{
public:
    Scene* scene;
    Camera* camera;

    int width{ 512 };
    int height{ 512 };

    int SSAA{ 2 };

    Vector3* frameBuffer;

    Render(Scene* scene, Camera* camera, int SSAA = 2, int maxDepth = 4, double RussianRoulette = 0.8);
    ~Render();

    void render();

    void saveImage(const std::string& filename);
};


Render::Render(Scene* scene, Camera* camera, int SSAA, int maxDepth, double RussianRoulette)
{
    this->scene = scene;
    this->camera = camera;
    this->width = (int)camera->width;
    this->height = (int)camera->height;
    this->SSAA = SSAA;
    this->scene->maxDepth = maxDepth;
    this->scene->RussianRoulette = RussianRoulette;

    CUDA_MALLOC(frameBuffer, width * height * sizeof(Vector3), Vector3);
}


Render::~Render()
{
    CUDA_FREE(frameBuffer);
}


void Render::render()
{
    std::cout << "Rendering..." << std::endl;
    // 清空帧缓存
    CUDA_MEMSET(frameBuffer, 0, width * height * sizeof(Vector3));

#if USE_CUDA
    dim3 blockDim(16, 16);
    dim3 gridDim(width / blockDim.x + 1, height / blockDim.y + 1);
    cudaRender<<<gridDim, blockDim>>>(width, height, SSAA, scene, camera, frameBuffer);


    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
#else
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            drawPixel(x, y, width, height, SSAA, scene, camera, frameBuffer);
        }
    }
#endif
}


void Render::saveImage(const std::string& filename)
{
    std::cout << "Saving image to " << filename << std::endl;
    // 将颜色值从线性空间转换为sRGB空间，并存储到unsigned char数组中
    std::vector<unsigned char> pixels(width * height * 3);
    for (size_t i = 0; i < width * height; ++i) {
        Vector3 color = frameBuffer[i].pow(1.0f / 2.2f).clamp(0.0f, 1.0f);
        pixels[i * 3] = (unsigned char)(color.x * 255);
        pixels[i * 3 + 1] = (unsigned char)(color.y * 255);
        pixels[i * 3 + 2] = (unsigned char)(color.z * 255);
    }

    stbi_write_png(filename.c_str(), width, height, 3, pixels.data(), width * 3);
}

#pragma once

#include "common.h"
#include "Scene.h"
#include "Camera.h"

#include <string>

#include "external/stb_image_write.h"

#if USE_CUDA
CUDA_DEVICE void cudaDrawPixel(int x, int y, int width, int height, int sqrtSPP, Scene* scene, Camera* camera, Vector3* frameBuffer, curandState_t* curandStates)
{
    curand_init(114154, y * width + x, 0, &curandStates[y * width + x]);
    Vector3 color(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < sqrtSPP; i++)
    {
        for (int j = 0; j < sqrtSPP; j++)
        {
            float sx = (x + (i + 1.0f) / (sqrtSPP + 1.0f)) / width;
            float sy = (y + (j + 1.0f) / (sqrtSPP + 1.0f)) / height;

            CUDA_LOG("CUDA: Generating ray for pixel (%d, %d)\n", x, y);
            //Ray ray = camera->generateRay(sx, sy);
            Ray ray = camera->generateRay(sx, sy);
            CUDA_LOG("CUDA: Casting ray for pixel (%d, %d)\n", x, y);
            color = color + scene->castRay(ray, 0, &curandStates[y * width + x]) / (float)(sqrtSPP * sqrtSPP);
            CUDA_LOG("CUDA: Got pixel (%d, %d)\n", x, y);

        }
    }

    frameBuffer[y * width + x] = color;
}
CUDA_GLOBAL void cudaRender(int width, int height, int sqrtSPP, Scene* scene, Camera* camera, Vector3* frameBuffer, curandState_t* curandStates)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    cudaDrawPixel(x, y, width, height, sqrtSPP, scene, camera, frameBuffer, curandStates);
}
#else
void drawPixel(int x, int y, int width, int height, int sqrtSPP, Scene* scene, Camera* camera, Vector3* frameBuffer)
{
    Vector3 color(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < sqrtSPP; i++)
    {
        for (int j = 0; j < sqrtSPP; j++)
        {
            float sx = (x + (i + 1.0f) / (sqrtSPP + 1.0f)) / width;
            float sy = (y + (j + 1.0f) / (sqrtSPP + 1.0f)) / height;

            Ray ray = camera->generateRay(sx, sy);
            color = color + scene->castRay(ray, 0) / (float)(sqrtSPP * sqrtSPP);
        }
    }

    frameBuffer[y * width + x] = color;
}
#endif


class Render
{
public:
    Scene* scene;
    Camera* camera;

    int width{ 512 };
    int height{ 512 };

    int sqrtSPP{ 2 };

    Vector3* frameBuffer;

#if USE_CUDA
    curandState_t* curandStates;
#endif

    Render(Scene* scene, Camera* camera, int SPP = 2, int maxDepth = 4, float RussianRoulette = 0.8);
    ~Render();

    void render();

    void saveImage(const std::string& filename);
};


Render::Render(Scene* scene, Camera* camera, int SPP, int maxDepth, float RussianRoulette)
{
    this->scene = scene;
    this->camera = camera;
    this->width = (int)camera->width;
    this->height = (int)camera->height;
    this->sqrtSPP = (int)sqrt(SPP);
    this->scene->maxDepth = std::min(maxDepth, MAX_DEPTH);
    this->scene->RussianRoulette = RussianRoulette;

    CUDA_MALLOC(frameBuffer, width * height * sizeof(Vector3), Vector3);

#if USE_CUDA
    CUDA_MALLOC(curandStates, width * height * sizeof(curandState_t), curandState_t);
#endif
}


Render::~Render()
{
    CUDA_FREE(frameBuffer);

#if USE_CUDA
    CUDA_FREE(curandStates);
#endif
}


void Render::render()
{
    std::cout << "Rendering..." << std::endl;
    // 清空帧缓存
    CUDA_MEMSET(frameBuffer, 0, width * height * sizeof(Vector3));

#if USE_CUDA
    dim3 blockDim(16, 16);
    dim3 gridDim(width / blockDim.x + 1, height / blockDim.y + 1);
    cudaRender<<<gridDim, blockDim>>>(width, height, sqrtSPP, scene, camera, frameBuffer, curandStates);


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
            drawPixel(x, y, width, height, sqrtSPP, scene, camera, frameBuffer);
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

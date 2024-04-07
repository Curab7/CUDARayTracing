#pragma once
#include "common.h"
#include "Math.h"
#include <string>
#include <vector>
#include <iostream>

#include "external/stb_image.h"

class Texture : public BaseClass
{
public:
    Vector3* pixels;
    int width, height;

    ~Texture() { CUDA_FREE(pixels); }

    static Texture* createFrom(const std::string& filename);

    CUDA_CALLABLE Vector3 getColor(const Vector2& uv) const
    {
        float x = uv.x * width;
        float y = uv.y * height;
        int x1 = CLAMP((int)std::floorf(x), 0, width - 1);
        int x2 = CLAMP((int)std::ceilf(x), 0, width - 1);
        int y1 = CLAMP((int)std::floorf(y), 0, height - 1);
        int y2 = CLAMP((int)std::ceilf(y), 0, height - 1);
        float dx = x - x1;
        float dy = y - y1;
        int index1 = y1 * width + x1;
        int index2 = y1 * width + x2;
        int index3 = y2 * width + x1;
        int index4 = y2 * width + x2;
        Vector3 color1 = pixels[index1].lerp(pixels[index2], dx);
        Vector3 color2 = pixels[index3].lerp(pixels[index4], dx);
        return color1.lerp(color2, dy);
    }

private:
    Texture(int width, int height) :width(width), height(height)
    {
        CUDA_MALLOC(pixels, width * height * sizeof(Vector3), Vector3);
    }
};

Texture* Texture::createFrom(const std::string& filename)
{
    int width, height, channels;
    unsigned char* imgData = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!imgData) {
        std::cout << "Failed to load texture: " << filename << std::endl;
        return nullptr;
    }
    if (channels != 3) {
        std::cerr << "Channels of texture is not 3." << std::endl;
    }
    Texture* texture = new Texture(width, height);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = i * width + j;
            texture->pixels[index] = Vector3((float)imgData[3*index] / 255.0f,
                                             (float)imgData[3*index+1] / 255.0f,
                                             (float)imgData[3*index+2] / 255.0f);
        }
    }
    return texture;
}
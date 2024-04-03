#pragma once
#include "common.h"
#include <cmath>
#include <algorithm>

#define TOLERANCE 0.000001f
#define ISZERO(x) (fabs(x) < TOLERANCE)

#define INF 1e30f
#define PI 3.14159265358979323846f
#define TWOPI 6.28318530717958647693f
#define HALFPI 1.57079632679489661923f
#define DEG2RAD(x) (x * PI / 180.0f)
#define RAD2DEG(x) (x * 180.0f / PI)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, minVal, maxVal) (MAX(MIN(x, maxVal), minVal))

class Vector3
{
public:
	union
	{
		struct
		{
			float x, y, z;
		};
		float v[3];
	};

	CUDA_CALLABLE Vector3() : x(0), y(0), z(0) {}
	CUDA_CALLABLE Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
	CUDA_CALLABLE Vector3(const Vector3 &other) : x(other.x), y(other.y), z(other.z) {}
	CUDA_CALLABLE Vector3(const float *pool, int index)
		: x(index < 0 ? 0 : pool[index * 3 + 0]), y(index < 0 ? 0 : pool[index * 3 + 1]), z(index < 0 ? 0 : pool[index * 3 + 2]) {}

	CUDA_CALLABLE Vector3 operator+(const Vector3 &other) const
	{
		return Vector3(x + other.x, y + other.y, z + other.z);
	}

	CUDA_CALLABLE Vector3 operator+=(const Vector3 &other)
	{
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}

	CUDA_CALLABLE Vector3 operator+(const float &scalar) const
	{
		return Vector3(x + scalar, y + scalar, z + scalar);
	}

	CUDA_CALLABLE Vector3 operator-(const Vector3 &other) const
	{
		return Vector3(x - other.x, y - other.y, z - other.z);
	}

	CUDA_CALLABLE Vector3 operator-=(const Vector3 &other)
	{
		x -= other.x;
		y -= other.y;
		z -= other.z;
		return *this;
	}

	CUDA_CALLABLE Vector3 operator-(const float &scalar) const
	{
		return Vector3(x - scalar, y - scalar, z - scalar);
	}

	CUDA_CALLABLE Vector3 operator*(float scalar) const
	{
		return Vector3(x * scalar, y * scalar, z * scalar);
	}

	CUDA_CALLABLE Vector3 operator*=(float scalar)
	{
		x *= scalar;
		y *= scalar;
		z *= scalar;
		return *this;
	}

	CUDA_CALLABLE Vector3 operator/(float scalar) const
	{
		return Vector3(x / scalar, y / scalar, z / scalar);
	}

	CUDA_CALLABLE Vector3 operator/=(float scalar)
	{
		x /= scalar;
		y /= scalar;
		z /= scalar;
		return *this;
	}

	CUDA_CALLABLE float &operator[](int index)
	{
		return v[index];
	}

	CUDA_CALLABLE const float &operator[](int index) const
	{
		return v[index];
	}

	CUDA_CALLABLE float dot(const Vector3 &other) const
	{
		return x * other.x + y * other.y + z * other.z;
	}

	CUDA_CALLABLE Vector3 cross(const Vector3 &other) const
	{
		return Vector3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
	}

	CUDA_CALLABLE float normSquared() const
	{
		return x * x + y * y + z * z;
	}

	CUDA_CALLABLE float norm() const
	{
		return sqrt(x * x + y * y + z * z);
	}

	CUDA_CALLABLE Vector3 normalized() const
	{
		float length = this->norm();
		return Vector3(x / length, y / length, z / length);
	}

	CUDA_CALLABLE Vector3 min(const Vector3 &other) const
	{
		return Vector3(MIN(x, other.x), MIN(y, other.y), MIN(z, other.z));
	}

	CUDA_CALLABLE Vector3 max(const Vector3 &other) const
	{
		return Vector3(MAX(x, other.x), MAX(y, other.y), MAX(z, other.z));
	}

	CUDA_CALLABLE Vector3 lerp(const Vector3 &other, float t) const
	{
		return Vector3(x + (other.x - x) * t, y + (other.y - y) * t, z + (other.z - z) * t);
	}

	CUDA_CALLABLE Vector3 pow(float exp) const
	{
		return Vector3(powf(x, exp), powf(y, exp), powf(z, exp));
	}

	CUDA_CALLABLE Vector3 clamp(float minVal, float maxVal) const
	{
		return Vector3(CLAMP(x, minVal, maxVal), CLAMP(y, minVal, maxVal), CLAMP(z, minVal, maxVal));
	}
};

CUDA_CALLABLE inline Vector3 operator-(const Vector3 &vec)
{
	return Vector3(-vec.x, -vec.y, -vec.z);
}

class Vector2
{
public:
	union
	{
		struct
		{
			float x, y;
		};
		float v[2];
	};

	CUDA_CALLABLE Vector2() : x(0), y(0) {}
	CUDA_CALLABLE Vector2(float x, float y) : x(x), y(y) {}
	CUDA_CALLABLE Vector2(const Vector2 &other) : x(other.x), y(other.y) {}
	CUDA_CALLABLE Vector2(const float *pool, int index)
		: x(index < 0 ? 0 : pool[index * 2 + 0]), y(index < 0 ? 0 : pool[index * 2 + 1]) {}

	CUDA_CALLABLE Vector2 operator+(const Vector2 &other) const
	{
		return Vector2(x + other.x, y + other.y);
	}

	CUDA_CALLABLE Vector2 operator-(const Vector2 &other) const
	{
		return Vector2(x - other.x, y - other.y);
	}

	CUDA_CALLABLE Vector2 operator*(float scalar) const
	{
		return Vector2(x * scalar, y * scalar);
	}

	CUDA_CALLABLE Vector2 operator/(float scalar) const
	{
		return Vector2(x / scalar, y / scalar);
	}

	CUDA_CALLABLE float &operator[](int index)
	{
		return v[index];
	}

	CUDA_CALLABLE const float &operator[](int index) const
	{
		return v[index];
	}

	CUDA_CALLABLE float dot(const Vector2 &other) const
	{
		return x * other.x + y * other.y;
	}

	CUDA_CALLABLE float normSquared() const
	{
		return x * x + y * y;
	}

	CUDA_CALLABLE float norm() const
	{
		return sqrt(x * x + y * y);
	}

	CUDA_CALLABLE Vector2 normalized() const
	{
		float length = this->norm();
		return Vector2(x / length, y / length);
	}

	CUDA_CALLABLE Vector2 lerp(const Vector2 &other, float t) const
	{
		return Vector2(x + (other.x - x) * t, y + (other.y - y) * t);
	}
};
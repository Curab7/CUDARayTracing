#pragma once
#include "common.h"
#include "Mesh.h"
#include "Material.h"
#include "Scene.h"
#include "Camera.h"

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>

#include <tinyxml2.h>

class Loader
{
public:
	std::string folderName;

	std::vector<Triangle*> allTriangleGroups;
	std::vector<Mesh*> allMeshes;
	std::map<std::string, std::map<std::string, Material*>> allMaterials; // path -> name -> material
	std::map<std::string, Texture*> allTextures;

	Scene* scene;
	Camera* camera;

	Loader(std::string folderName) : folderName(folderName) {}
	~Loader()
	{
		for (auto triangle : allTriangleGroups)
			delete[] triangle;
		for (auto mesh : allMeshes)
			delete mesh;
		for (auto pathPair : allMaterials)
			for(auto namePair : pathPair.second)
				delete namePair.second;
		for (auto texture : allTextures)
			delete texture.second;
	}

	bool loadCoursePack(const std::string& caseName);
	Scene* getScene() { return scene; }
	Camera* getCamera() { return camera; }

private:
	Mesh* loadOBJ(const std::string& filename);
	bool loadXML(const std::string& filename, Camera** camera, std::map<std::string, Vector3>* lightRadiance);
	std::map<std::string, Material*>* loadMTL(const std::string& filename);
	Texture* loadTexture(const std::string& filename);
};

bool Loader::loadCoursePack(const std::string& caseName)
{
	std::cout << "Loading " + caseName + "..." << std::endl;

	scene = new Scene();
	scene->addObject(loadOBJ(caseName +".obj"));
	scene->buildBVH();

	std::map<std::string, Vector3> lightRadiance;
	bool success = loadXML(caseName + ".xml", &camera, &lightRadiance);
	if (success)
	{
		auto& sceneMaterials = allMaterials[caseName + ".mtl"];
		for (auto light : lightRadiance)
		{
			sceneMaterials[light.first]->Ke = light.second;
		}
	}
	return success && scene!=nullptr && camera!=nullptr;
}

Mesh* Loader::loadOBJ(const std::string& filename)
{
	std::ifstream file(folderName + "/" + filename);
	if (!file.is_open())
	{
		std::cout << "Failed to open file: " << folderName + "/" + filename << std::endl;
		return false;
	}

	std::string line;

	std::vector<Vector3> vertices;
	std::vector<Vector3> normals;
	std::vector<Vector2> uvs;
	std::map<std::string, Material*>* matlib{nullptr};

	std::vector<int> idxVertices, idxNormals, idxUvs;
	std::vector<std::string> nameMaterials;
	std::string curMaterial;

	while (std::getline(file, line))
	{
		if (line.empty())
			continue;

		std::stringstream ss(line);
		std::string type;
		ss >> type;

		if (type == "v")
		{
			float x, y, z;
			ss >> x >> y >> z;
			vertices.push_back(Vector3(x, y, z));
		}
		else if (type == "vn")
		{
			float x, y, z;
			ss >> x >> y >> z;
			normals.push_back(Vector3(x, y, z));
		}
		else if (type == "vt")
		{
			float u, v;
			ss >> u >> v;
			uvs.push_back(Vector2(u, v));
		}
		else if (type == "f")
		{
			std::string vertex1, vertex2, vertex3;
			ss >> vertex1 >> vertex2 >> vertex3;

			int idxVertex[3], idxUV[3], idxNormal[3];
			sscanf(vertex1.c_str(), "%d/%d/%d", &idxVertex[0], &idxUV[0], &idxNormal[0]);
			sscanf(vertex2.c_str(), "%d/%d/%d", &idxVertex[1], &idxUV[1], &idxNormal[1]);
			sscanf(vertex3.c_str(), "%d/%d/%d", &idxVertex[2], &idxUV[2], &idxNormal[2]);

			idxVertices.push_back(idxVertex[0] - 1);
			idxVertices.push_back(idxVertex[1] - 1);
			idxVertices.push_back(idxVertex[2] - 1);

			idxUvs.push_back(idxUV[0] - 1);
			idxUvs.push_back(idxUV[1] - 1);
			idxUvs.push_back(idxUV[2] - 1);

			idxNormals.push_back(idxNormal[0] - 1);
			idxNormals.push_back(idxNormal[1] - 1);
			idxNormals.push_back(idxNormal[2] - 1);

			nameMaterials.push_back(curMaterial);
		}
		else if (type == "mtllib")
		{
			std::string mtlFilename;
			ss >> mtlFilename;
			matlib = loadMTL(mtlFilename);
			if (matlib == nullptr)
			{
				std::cout << "Failed to load material library: " << mtlFilename << std::endl;
				return false;
			}
		}
		else if (type == "usemtl")
		{
			ss >> curMaterial;
		}
	}

	int numTriangles = idxVertices.size() / 3;
	Triangle* triangleGroup = new Triangle[numTriangles];
	allTriangleGroups.push_back(triangleGroup);
	for (int i = 0; i < idxVertices.size(); i += 3)
	{
		Vector3 v1 = vertices[idxVertices[i]];
		Vector3 v2 = vertices[idxVertices[i + 1]];
		Vector3 v3 = vertices[idxVertices[i + 2]];

		Vector3 n1 = idxNormals[i] >= 0 ? normals[idxNormals[i]] : Vector3(0, 0, 0);
		Vector3 n2 = idxNormals[i + 1] >= 0 ? normals[idxNormals[i + 1]] : Vector3(0, 0, 0);
		Vector3 n3 = idxNormals[i + 2] >= 0 ? normals[idxNormals[i + 2]] : Vector3(0, 0, 0);

		Vector2 uv1 = idxUvs[i] >= 0 ? uvs[idxUvs[i]] : Vector2(0, 0);
		Vector2 uv2 = idxUvs[i + 1] >= 0 ? uvs[idxUvs[i + 1]] : Vector2(0, 0);
		Vector2 uv3 = idxUvs[i + 2] >= 0 ? uvs[idxUvs[i + 2]] : Vector2(0, 0);

		Material* material = nullptr;
		std::string materialName = nameMaterials[i / 3];
		if (matlib != nullptr && !materialName.empty())
		{
			auto pair = matlib->find(materialName);
			if (pair != matlib->end())
				material = pair->second;
		}

		triangleGroup[i/3] = Triangle(v1, v2, v3, n1, n2, n3, uv1, uv2, uv3, material);
	};

	std::vector<Triangle*> triangles(numTriangles);
	for (int i = 0; i < numTriangles; i++)
	{
		triangles[i] = &triangleGroup[i];
	}

	Mesh* mesh = new Mesh(triangles);
	allMeshes.push_back(mesh);

	return mesh;
}

std::map<std::string, Material*>* Loader::loadMTL(const std::string& filename)
{
	if (allMaterials.find(filename) != allMaterials.end())
		return &allMaterials[filename];

	std::ifstream file(folderName + "/" + filename);
	if (!file.is_open())
	{
		std::cout << "Failed to open file: " << folderName + "/" + filename << std::endl;
		return nullptr;
	}

	allMaterials[filename] = std::map<std::string, Material*>();
	std::map<std::string, Material*>& materials = allMaterials[filename];

	std::string line;
	Material* material{nullptr};

	while (std::getline(file, line))
	{
		if (line.empty())
			continue;

		std::stringstream ss(line);
		std::string type;
		ss >> type;

		if (type == "newmtl")
		{
			std::string name;
			ss >> name;
			material = new Material();
			materials[name] = material;
		}
		else if (type == "Ns")
		{
			float ns;
			ss >> ns;
			material->Ns = ns;
		}
		else if (type == "Kd")
		{
			float r, g, b;
			ss >> r >> g >> b;
			material->Kd = Vector3(r, g, b);
		}
		else if (type == "Ks")
		{
			float r, g, b;
			ss >> r >> g >> b;
			material->Ks = Vector3(r, g, b);
		}
		else if (type == "Ni")
		{
			float ni;
			ss >> ni;
			material->Ni = ni;
		}
		else if (type == "Tr")
		{
			float r, g, b;
			ss >> r >> g >> b;
			material->Tr = Vector3(r, g, b);
		}
		else if (type == "map_Kd")
		{
			std::string filename;
			ss >> filename;
			material->map_Kd = loadTexture(filename);
		}
	}

	return &allMaterials[filename];
}

Texture* Loader::loadTexture(const std::string& filename)
{
	if (allTextures.find(filename) != allTextures.end())
		return allTextures[filename];

	Texture* texture = Texture::createFrom(folderName + "/" + filename);

	allTextures[filename] = texture;

	return texture;
}

bool Loader::loadXML(const std::string& filename, Camera** camera, std::map<std::string, Vector3>* lightRadiance)
{
	tinyxml2::XMLDocument doc;
	auto result = doc.LoadFile((folderName + "/" + filename).c_str());

	if (result != tinyxml2::XML_SUCCESS) {
		std::cerr << "Error loading XML file." << std::endl;
		return false;
	}

	// 解析camera信息
	float fovy;
	int width, height;
	Vector3 cameraEye, cameraLookat, cameraUp;
	auto cameraElement = doc.FirstChildElement("camera");
	cameraElement->QueryIntAttribute("width", &width);
	cameraElement->QueryIntAttribute("height", &height);
	cameraElement->QueryFloatAttribute("fovy", &fovy);

	auto eyeElement = cameraElement->FirstChildElement("eye");
	auto lookatElement = cameraElement->FirstChildElement("lookat");
	auto upElement = cameraElement->FirstChildElement("up");
	eyeElement->QueryFloatAttribute("x", &cameraEye.x);
	eyeElement->QueryFloatAttribute("y", &cameraEye.y);
	eyeElement->QueryFloatAttribute("z", &cameraEye.z);
	lookatElement->QueryFloatAttribute("x", &cameraLookat.x);
	lookatElement->QueryFloatAttribute("y", &cameraLookat.y);
	lookatElement->QueryFloatAttribute("z", &cameraLookat.z);
	upElement->QueryFloatAttribute("x", &cameraUp.x);
	upElement->QueryFloatAttribute("y", &cameraUp.y);
	upElement->QueryFloatAttribute("z", &cameraUp.z);

	*camera = new Camera(cameraEye, (cameraLookat - cameraEye).normalized(), cameraUp.normalized(), fovy, width, height);

	// 解析light信息
	auto lightElement = doc.FirstChildElement("light");
	while (lightElement) {
		const char* mtlname;
		const char* radiance;
		mtlname = lightElement->Attribute("mtlname");
		radiance = lightElement->Attribute("radiance");

		// 解析radiance属性并将其以glm::vec3形式存到map中
		std::istringstream radianceStream(radiance);
		float x, y, z;
		char separator;
		radianceStream >> x >> separator >> y >> separator >> z;
		(*lightRadiance)[mtlname] = Vector3(x, y, z);
		lightElement = lightElement->NextSiblingElement();
	}

	return true;
}
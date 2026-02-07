#pragma once

#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// ќбъ€влени€ (extern) глобальных переменных
extern std::string YOLOv3CONF;
extern std::string YOLOv3WEIGHT;
extern std::string YOLOv8n;
extern std::string YOLOv8m;
extern std::string YOLOv26n;
extern std::string YOLOv26m;
extern std::string CLASSES;
extern std::string FACE_CASCADE_FRONTAL;
extern std::string FACE_CASCADE_PROFILE;

// ќбъ€влени€ функций
bool fileExists(const std::string& path);
std::string getAbsolutePath(const std::string& relativePath);
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// Для YOLOv3
std::string YOLOv3CONF = "models/yolov3.cfg";
std::string YOLOv3WEIGHT = "models/yolov3.weights";

// Для YOLOv8
std::string YOLOv8n = "models/yolov8n.onnx";
std::string YOLOv8m = "models/yolov8m.onnx";

// Для YOLOv26
std::string YOLOv26n = "models/yolov26n.onnx";
std::string YOLOv26m = "models/yolov26m.onnx";

// Для классов
std::string CLASSES = "models/coco.names";

// Для каскадов лиц
std::string FACE_CASCADE_FRONTAL = "models/haarcascade_frontalface_default.xml";
std::string FACE_CASCADE_PROFILE = "models/haarcascade_profileface.xml";

// Вспомогательная функция для проверки существования файла
bool fileExists(const std::string& path) {
    return fs::exists(path);
}

// Функция для получения абсолютного пути
std::string getAbsolutePath(const std::string& relativePath) {
    try {
        return fs::absolute(relativePath).string();
    }
    catch (...) {
        return relativePath;
    }
}
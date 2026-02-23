#include "./global.h"
#include <cmath>

// Базовый класс для координат 2D
struct Cord2D {
    float x = 0;
    float y = 0;

    Cord2D() = default;
    Cord2D(float x, float y) : x(x), y(y) {}

    // Сложение
    Cord2D operator+(const Cord2D& other) const {
        return Cord2D(x + other.x, y + other.y);
    }

    // Вычитание
    Cord2D operator-(const Cord2D& other) const {
        return Cord2D(x - other.x, y - other.y);
    }

    // Умножение на скаляр
    Cord2D operator*(float scalar) const {
        return Cord2D(x * scalar, y * scalar);
    }

    // Деление на скаляр
    Cord2D operator/(float scalar) const {
        if (scalar != 0)
            return Cord2D(x / scalar, y / scalar);
        return Cord2D(0, 0);
    }

    // Нормализация (приведение к длине 1)
    Cord2D normalize() const {
        float length = std::sqrt(x * x + y * y);
        if (length > 0)
            return Cord2D(x / length, y / length);
        return Cord2D(0, 0);
    }

    // Длина вектора
    float length() const {
        return std::sqrt(x * x + y * y);
    }

    // Скалярное произведение
    float dot(const Cord2D& other) const {
        return x * other.x + y * other.y;
    }
};

// Базовый класс для координат 3D
struct Cord3D {
    float x = 0;
    float y = 0;
    float z = 0;

    Cord3D() = default;
    Cord3D(float x, float y, float z) : x(x), y(y), z(z) {}

    // Сложение
    Cord3D operator+(const Cord3D& other) const {
        return Cord3D(x + other.x, y + other.y, z + other.z);
    }

    // Вычитание
    Cord3D operator-(const Cord3D& other) const {
        return Cord3D(x - other.x, y - other.y, z - other.z);
    }

    // Умножение на скаляр
    Cord3D operator*(float scalar) const {
        return Cord3D(x * scalar, y * scalar, z * scalar);
    }

    // Деление на скаляр
    Cord3D operator/(float scalar) const {
        if (scalar != 0)
            return Cord3D(x / scalar, y / scalar, z / scalar);
        return Cord3D(0, 0, 0);
    }

    // Нормализация (приведение к длине 1)
    Cord3D normalize() const {
        float length = std::sqrt(x * x + y * y + z * z);
        if (length > 0)
            return Cord3D(x / length, y / length, z / length);
        return Cord3D(0, 0, 0);
    }

    // Длина вектора
    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    // Скалярное произведение
    float dot(const Cord3D& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // Векторное произведение
    Cord3D cross(const Cord3D& other) const {
        return Cord3D(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    // Преобразование в 2D (проекция на плоскость XY)
    Cord2D to2D() const {
        return Cord2D(x, y);
    }
};

// Псевдонимы для удобства
using Velocity2D = Cord2D;
using Size2D = Cord2D;
using Pos2D = Cord2D;
using Acceleration2D = Cord2D;

using Velocity3D = Cord3D;
using Size3D = Cord3D;
using Pos3D = Cord3D;
using Acceleration3D = Cord3D;

// Параметры камеры для преобразования координат
struct CameraParams {
    // Фокусное расстояние (в пикселях или мм)
    float focalLength = 0;

    // Размер сенсора (в мм)
    float sensorWidth = 0;
    float sensorHeight = 0;

    // Разрешение изображения
    int imageWidth = 0;
    int imageHeight = 0;

    // Положение камеры в 3D мире
    Pos3D cameraPosition = Pos3D(0, 0, 0);

    // Ориентация камеры (углы Эйлера в радианах)
    float yaw = 0;      // поворот вокруг оси Y
    float pitch = 0;    // поворот вокруг оси X
    float roll = 0;     // поворот вокруг оси Z

    CameraParams() = default;

    CameraParams(float focal, float sensorW, float sensorH,
        int imgW, int imgH)
        : focalLength(focal), sensorWidth(sensorW), sensorHeight(sensorH),
        imageWidth(imgW), imageHeight(imgH) {
    }

    // Получение фокусного расстояния в пикселях
    float getFocalLengthInPixelsX() const {
        if (sensorWidth > 0)
            return (focalLength * imageWidth) / sensorWidth;
        return focalLength;
    }

    float getFocalLengthInPixelsY() const {
        if (sensorHeight > 0)
            return (focalLength * imageHeight) / sensorHeight;
        return focalLength;
    }

    // Матрица вращения камеры (упрощенная версия)
    Cord3D rotatePoint(const Cord3D& point) const {
        // Поворот по оси Y (yaw)
        float cosY = std::cos(yaw);
        float sinY = std::sin(yaw);
        float x1 = point.x * cosY - point.z * sinY;
        float z1 = point.x * sinY + point.z * cosY;

        // Поворот по оси X (pitch)
        float cosP = std::cos(pitch);
        float sinP = std::sin(pitch);
        float y1 = point.y * cosP - z1 * sinP;
        float z2 = point.y * sinP + z1 * cosP;

        // Поворот по оси Z (roll) - обычно не используется для камеры
        float cosR = std::cos(roll);
        float sinR = std::sin(roll);
        float x2 = x1 * cosR - y1 * sinR;
        float y2 = x1 * sinR + y1 * cosR;

        return Cord3D(x2, y2, z2);
    }
};

// Функции преобразования координат
namespace CoordinateConverter {
    // Преобразование из 2D пиксельных координат в 3D координаты (в системе камеры)
    static Pos3D pixelToCamera3D(const Pos2D& pixelPos, float distance, const CameraParams& camera) {
        // Центр изображения
        float cx = camera.imageWidth / 2.0f;
        float cy = camera.imageHeight / 2.0f;

        // Фокусное расстояние в пикселях
        float fx = camera.getFocalLengthInPixelsX();
        float fy = camera.getFocalLengthInPixelsY();

        // Нормализованные координаты
        float x_norm = (pixelPos.x - cx) / fx;
        float y_norm = (pixelPos.y - cy) / fy;

        // Координаты в системе камеры
        return Pos3D(
            x_norm * distance,
            y_norm * distance,
            distance
        );
    }

    // Преобразование из 3D координат (в системе камеры) в 2D пиксельные координаты
    static Pos2D camera3DToPixel(const Pos3D& cameraPos, const CameraParams& camera) {
        // Фокусное расстояние в пикселях
        float fx = camera.getFocalLengthInPixelsX();
        float fy = camera.getFocalLengthInPixelsY();

        // Центр изображения
        float cx = camera.imageWidth / 2.0f;
        float cy = camera.imageHeight / 2.0f;

        // Проекция на изображение
        float u = (cameraPos.x / cameraPos.z) * fx + cx;
        float v = (cameraPos.y / cameraPos.z) * fy + cy;

        return Pos2D(u, v);
    }

    // Преобразование из мировых 3D координат в пиксельные
    static Pos2D world3DToPixel(const Pos3D& worldPos, const CameraParams& camera) {
        // Перевод в систему координат камеры
        Pos3D relativePos = worldPos - camera.cameraPosition;
        Pos3D cameraPos = camera.rotatePoint(relativePos);

        // Проекция на изображение
        return camera3DToPixel(cameraPos, camera);
    }

    // Преобразование из пиксельных координат в мировые 3D координаты
    static Pos3D pixelToWorld3D(const Pos2D& pixelPos, float distance, const CameraParams& camera) {
        // Перевод в систему координат камеры
        Pos3D cameraPos = pixelToCamera3D(pixelPos, distance, camera);

        // Обратное вращение
        CameraParams invCamera = camera;
        invCamera.yaw = -camera.yaw;
        invCamera.pitch = -camera.pitch;
        invCamera.roll = -camera.roll;

        Pos3D rotatedPos = invCamera.rotatePoint(cameraPos);

        // Перевод в мировые координаты
        return rotatedPos + camera.cameraPosition;
    }

    // Преобразование скорости из 2D пикселей/сек в 3D м/сек
    static Velocity3D pixelVelocityTo3D(const Velocity2D& pixelVel, float distance,
        const CameraParams& camera) {
        // Масштабный коэффициент (метры на пиксель)
        float scaleX = (camera.sensorWidth / camera.imageWidth) * (distance / camera.focalLength);
        float scaleY = (camera.sensorHeight / camera.imageHeight) * (distance / camera.focalLength);

        return Velocity3D(
            pixelVel.x * scaleX,
            pixelVel.y * scaleY,
            0  // Z-компоненту нельзя определить из 2D
        );
    }

    // Преобразование скорости из 3D м/сек в 2D пиксели/сек
    static Velocity2D velocity3DToPixel(const Velocity3D& worldVel, float distance,
        const CameraParams& camera) {
        // Масштабный коэффициент (пиксели на метр)
        float scaleX = (camera.imageWidth / camera.sensorWidth) * (camera.focalLength / distance);
        float scaleY = (camera.imageHeight / camera.sensorHeight) * (camera.focalLength / distance);

        return Velocity2D(
            worldVel.x * scaleX,
            worldVel.y * scaleY
        );
    }
}

struct TargetInfo {
    // Идентификатор цели
    int id = -1;

    // Тип цели
    std::string type;

    // Размеры в пикселях
    Size2D p_size;

    // Реальные размеры (в метрах)
    Size3D r_size;

    // Расстояние до объекта (в метрах)
    double distance = 0;

    // Уверенность в обнаружении (0-1)
    float confidence = 0.0f;

    // Время последнего обновления
    double lastUpdateTime = 0.0;

    // Позиция центра в пикселях
    Pos2D center;

    // Позиция в 3D пространстве (в метрах)
    Pos3D position3D;

    // Bounding box в пикселях
    Pos2D bbox_tl;  // top-left
    Pos2D bbox_br;  // bottom-right

    // Параметры камеры (используются для преобразований)
    CameraParams cameraParams;

    // Скорость в пикселях/сек
    Velocity2D velocity2D;

    // Скорость в метрах/сек
    Velocity3D velocity3D;

    TargetInfo() : p_size(0, 0), r_size(0, 0, 0), center(0, 0),
        position3D(0, 0, 0), bbox_tl(0, 0), bbox_br(0, 0),
        velocity2D(0, 0), velocity3D(0, 0, 0) {
    }

    TargetInfo(int id, const std::string& type, const Size2D& p, const Size3D& r)
        : id(id), type(type), p_size(p), r_size(r), center(0, 0),
        position3D(0, 0, 0), bbox_tl(0, 0), bbox_br(0, 0),
        velocity2D(0, 0), velocity3D(0, 0, 0) {
    }

    // Обновление 2D скорости на основе 3D скорости
    void updateVelocity2DFrom3D() {
        velocity2D = CoordinateConverter::velocity3DToPixel(
            velocity3D, static_cast<float>(distance), cameraParams
        );
    }

    // Обновление 3D скорости на основе 2D скорости
    void updateVelocity3DFrom2D() {
        velocity3D = CoordinateConverter::pixelVelocityTo3D(
            velocity2D, static_cast<float>(distance), cameraParams
        );
    }

    // Обновление 3D позиции на основе 2D позиции и расстояния
    void updatePosition3DFrom2D() {
        position3D = CoordinateConverter::pixelToWorld3D(
            center, static_cast<float>(distance), cameraParams
        );
    }

    // Обновление 2D позиции на основе 3D позиции
    void updatePosition2DFrom3D() {
        center = CoordinateConverter::world3DToPixel(position3D, cameraParams);

        // Обновление bounding box на основе нового центра и размеров
        bbox_tl = Pos2D(center.x - p_size.x / 2, center.y - p_size.y / 2);
        bbox_br = Pos2D(center.x + p_size.x / 2, center.y + p_size.y / 2);
    }

    // Получение bounding box как прямоугольника
    Size2D getBBoxSize() const {
        return Size2D(
            std::abs(bbox_br.x - bbox_tl.x),
            std::abs(bbox_br.y - bbox_tl.y)
        );
    }

    // Установка bounding box через координаты углов
    void setBBox(const Pos2D& tl, const Pos2D& br) {
        bbox_tl = tl;
        bbox_br = br;
        p_size = getBBoxSize();
        center = Pos2D((tl.x + br.x) / 2, (tl.y + br.y) / 2);
    }

    // Установка bounding box через центр и размеры
    void setBBoxFromCenter(const Pos2D& centerPos, const Size2D& size) {
        center = centerPos;
        p_size = size;
        bbox_tl = Pos2D(center.x - size.x / 2, center.y - size.y / 2);
        bbox_br = Pos2D(center.x + size.x / 2, center.y + size.y / 2);
    }
};
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <chrono>
#include <cmath>
#include "ObjInfo.h"

// Базовый класс для всех целей (объектов)
class Target {
public:
    Target() {
        m_info.id = s_nextId++;
        m_info.type = "unknown";
        m_info.lastUpdateTime = getCurrentTime();
    }

    Target(const std::string& type, float realWidth, float realHeight) {
        m_info.id = s_nextId++;
        m_info.type = type;
        m_info.r_size.x = realWidth;
        m_info.r_size.y = realHeight;
        m_info.lastUpdateTime = getCurrentTime();
    }

    virtual ~Target() = default;

    // Обновление позиции и размера в пикселях
    virtual void update(const cv::Rect& boundingBox, float focalLength, float sensorWidth = 0.0f) {
        // Обновление bounding box
        m_info.setBoundingBox(boundingBox);

        // Обновление параметров камеры
        m_info.cameraParams.focalLength = focalLength;
        m_info.cameraParams.sensorWidth = sensorWidth;

        // Обновление времени
        m_info.lastUpdateTime = getCurrentTime();

        // Пересчет расстояния
        m_info.distance = m_info.calculateDistance();

        // Увеличение уверенности
        m_info.confidence = std::min(1.0f, m_info.confidence + 0.1f);
    }

    // Расчет расстояния до объекта
    virtual float calculateDistance() const {
        return m_info.calculateDistance();
    }

    // Расчет реальных размеров на основе известного расстояния
    virtual void calculateRealSize(float knownDistance) {
        float focalLength = m_info.cameraParams.getFocalLengthInPixelsX();

        if (knownDistance > 0 && focalLength > 0) {
            if (m_info.p_size.x > 0) {
                m_info.r_size.x = (m_info.p_size.x * knownDistance) / focalLength;
            }
            if (m_info.p_size.y > 0) {
                m_info.r_size.y = (m_info.p_size.y * knownDistance) / focalLength;
            }
        }
    }

    // Получение информации об объекте
    virtual std::string getInfo() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer),
            "ID: %d | Type: %s | Size: %.1fx%.1fpx | Real: %.2fx%.2fm | Distance: %.2fm | Conf: %.1f%%",
            m_info.id, m_info.type.c_str(),
            m_info.p_size.x, m_info.p_size.y,
            m_info.r_size.x, m_info.r_size.y,
            m_info.distance, m_info.confidence * 100.0f);
        return std::string(buffer);
    }

    // Getters
    cv::Rect getBoundingBox() const {
        return m_info.getBoundingBox();
    }

    cv::Point2f getCenter() const {
        return cv::Point2f(m_info.center.x, m_info.center.y);
    }

    float getPixelWidth() const { return m_info.p_size.x; }
    float getPixelHeight() const { return m_info.p_size.y; }
    float getRealWidth() const { return m_info.r_size.x; }
    float getRealHeight() const { return m_info.r_size.y; }
    float getDistance() const { return static_cast<float>(m_info.distance); }
    float getConfidence() const { return m_info.confidence; }
    const std::string& getType() const { return m_info.type; }
    int getId() const { return m_info.id; }

    // Получение всей структуры TargetInfo
    const TargetInfo& getTargetInfo() const { return m_info; }
    TargetInfo& getTargetInfo() { return m_info; }

    // Setters
    void setRealSize(float width, float height) {
        m_info.r_size.x = width;
        m_info.r_size.y = height;
        m_info.distance = calculateDistance();
    }

    void setFocalLength(float focalLength) {
        m_info.cameraParams.focalLength = focalLength;
        m_info.distance = calculateDistance();
    }

    void setSensorWidth(float sensorWidth) {
        m_info.cameraParams.sensorWidth = sensorWidth;
    }

    void setConfidence(float confidence) {
        m_info.confidence = std::max(0.0f, std::min(1.0f, confidence));
    }

    // Расчет угловых размеров (в градусах)
    float getAngularWidth() const {
        if (m_info.distance > 0 && m_info.r_size.x > 0) {
            return 2.0f * atan(m_info.r_size.x / (2.0f * static_cast<float>(m_info.distance)))
                * 180.0f / static_cast<float>(M_PI);
        }
        return 0.0f;
    }

    float getAngularHeight() const {
        if (m_info.distance > 0 && m_info.r_size.y > 0) {
            return 2.0f * atan(m_info.r_size.y / (2.0f * static_cast<float>(m_info.distance)))
                * 180.0f / static_cast<float>(M_PI);
        }
        return 0.0f;
    }

    // Расчет скорости движения
    cv::Point2f getVelocity() const {
        return cv::Point2f(m_info.velocity2D.x, m_info.velocity2D.y);
    }

    void updateVelocity(const cv::Point2f& newPosition, float deltaTime) {
        if (deltaTime > 0) {
            cv::Point2f displacement = newPosition - getCenter();
            m_info.velocity2D.x = displacement.x / deltaTime;
            m_info.velocity2D.y = displacement.y / deltaTime;
            m_info.center.x = newPosition.x;
            m_info.center.y = newPosition.y;
        }
    }

    // Установка разрешения изображения для камеры
    void setImageResolution(int width, int height) {
        m_info.cameraParams.imageWidth = width;
        m_info.cameraParams.imageHeight = height;
    }

protected:
    // Основная структура данных
    TargetInfo m_info;

    // Статические счетчики
    static int s_nextId;

private:
    // Вспомогательная функция для получения текущего времени
    static double getCurrentTime() {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<double>(now.time_since_epoch());
        return duration.count();
    }
};

// Инициализация статической переменной
int Target::s_nextId = 0;
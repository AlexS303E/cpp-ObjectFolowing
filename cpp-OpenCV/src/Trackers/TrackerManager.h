#pragma once

#include <memory>
#include "ObjectTracker.h"
#include "FaceTracker.h"

class TrackerManager {
public:
    enum class TrackerType {
        OBJECT_TRACKER,
        FACE_TRACKER
    };

    TrackerManager();
    ~TrackerManager() = default;

    // Переключение режима трекера
    void switchTracker(TrackerType type);
    TrackerType getCurrentTrackerType() const;

    // Инициализация текущего трекера
    bool initialize(const cv::Mat& frame);

    // Обновление текущего трекера
    bool update(const cv::Mat& frame);

    // Отрисовка информации текущего трекера
    void drawTrackingInfo(cv::Mat& frame) const;

    // Сброс текущего трекера
    void reset();

    // Проверка инициализации
    bool isInitialized() const;

    // Переключение выбранной цели
    void selectPrevTrg();
    void selectNextTrg();

    // Переключение режим трекера
    void selectPrevTrkMode();
    void selectNextTrkMode();

    // Получение текущего трекера
    ObjectTracker* getObjectTracker();
    FaceTracker* getFaceTracker();

    // Установка режима отслеживания лиц для ObjectTracker
    void setFaceTrackingMode(bool enable);
    bool isFaceTrackingMode() const;

private:
    TrackerType m_currentType;
    std::unique_ptr<ObjectTracker> m_objectTracker;
    std::unique_ptr<FaceTracker> m_faceTracker;

    // Вспомогательная функция для инициализации трекеров
    void initializeTrackers();
};
#pragma once

#include <memory>
#include "ObjectTracker.h"
#include "FaceTracker.h"
#include "LineModTracker.h"

class TrackerManager {
public:
    enum class TrackerType {
        OBJECT_TRACKER,
        FACE_TRACKER,
        LINEMOD_TRACKER
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
    LineModTracker* getLineModTracker();

    // Установка режима отслеживания лиц для ObjectTracker
    void setFaceTrackingMode(bool enable);
    bool isFaceTrackingMode() const;

private:
    TrackerType m_currentType_;
    std::unique_ptr<ObjectTracker> m_objectTracker_;
    std::unique_ptr<FaceTracker> m_faceTracker_;
    std::unique_ptr<LineModTracker> m_lineModTracker_;

    // Вспомогательная функция для инициализации трекеров
    void initializeTrackers();
};

inline std::ostream& operator<<(std::ostream& os, TrackerManager::TrackerType tt) {
    switch (tt) {
    case TrackerManager::TrackerType::OBJECT_TRACKER:   os << "Combined"; break;
    case TrackerManager::TrackerType::FACE_TRACKER: os << "Face"; break;
    case TrackerManager::TrackerType::LINEMOD_TRACKER:  os << "LineMod"; break;
    default:           os << "Unknown"; break;
    }
    return os;
}
#include "TrackerManager.h"
#include <iostream>

TrackerManager::TrackerManager()
    : m_currentType(TrackerType::OBJECT_TRACKER) {
    initializeTrackers();
}

void TrackerManager::initializeTrackers() {
    m_objectTracker = std::make_unique<ObjectTracker>();
    m_faceTracker = std::make_unique<FaceTracker>();
}

void TrackerManager::switchTracker(TrackerType type) {
    if (type == m_currentType) {
        return;
    }

    m_currentType = type;

    // Сбрасываем оба трекера при переключении
    if (m_objectTracker) {
        m_objectTracker->reset();
    }

    std::cout << "Switched to "
        << (type == TrackerType::OBJECT_TRACKER ? "Object Tracker" : "Face Tracker")
        << std::endl;
}

TrackerManager::TrackerType TrackerManager::getCurrentTrackerType() const {
    return m_currentType;
}

bool TrackerManager::initialize(const cv::Mat& frame) {
    if (m_currentType == TrackerType::OBJECT_TRACKER) {
        return m_objectTracker->initialize(frame);
    }
    else {
        // Для FaceTracker используем собственную логику инициализации
        std::vector<cv::Rect> faces = m_faceTracker->detectFaces(frame);
        return !faces.empty();
    }
}

bool TrackerManager::update(const cv::Mat& frame) {
    if (m_currentType == TrackerType::OBJECT_TRACKER) {
        return m_objectTracker->update(frame);
    }
    else {
        // FaceTracker не имеет метода update, поэтому всегда возвращаем true
        // при условии, что лица обнаружены
        std::vector<cv::Rect> faces = m_faceTracker->detectFaces(frame);
        return !faces.empty();
    }
}

void TrackerManager::drawTrackingInfo(cv::Mat& frame) const {
    if (m_currentType == TrackerType::OBJECT_TRACKER) {
        m_objectTracker->drawTrackingInfo(frame);
    }
    else {
        std::vector<cv::Rect> faces = m_faceTracker->detectFaces(frame);
        m_faceTracker->drawFaces(frame, faces);

        // Добавляем информацию о режиме
        cv::putText(frame, "Face Tracker Mode", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    }
}

void TrackerManager::reset() {
    if (m_currentType == TrackerType::OBJECT_TRACKER) {
        m_objectTracker->reset();
    }
    else {
        // FaceTracker не имеет метода reset, очищаем обнаруженные лица
        // путем повторного создания объекта
        m_faceTracker = std::make_unique<FaceTracker>();
    }
}

bool TrackerManager::isInitialized() const {
    if (m_currentType == TrackerType::OBJECT_TRACKER) {
        return m_objectTracker->isInitialized();
    }

    // FaceTracker всегда считается инициализированным
    return true;
}

ObjectTracker* TrackerManager::getObjectTracker() {
    return m_objectTracker.get();
}

FaceTracker* TrackerManager::getFaceTracker() {
    return m_faceTracker.get();
}

void TrackerManager::setFaceTrackingMode(bool enable) {
    if (m_objectTracker) {
        m_objectTracker->setFaceTrackingMode(enable);
    }
}

bool TrackerManager::isFaceTrackingMode() const {
    if (m_objectTracker) {
        return m_objectTracker->isFaceTrackingMode();
    }
    return false;
}
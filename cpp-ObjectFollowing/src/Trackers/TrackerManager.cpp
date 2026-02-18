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
    if (m_faceTracker) {
        m_faceTracker->reset();
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
        // Используем публичный метод инициализации FaceTracker
        return m_faceTracker->initialize(frame);
    }
}

bool TrackerManager::update(const cv::Mat& frame) {
    if (m_currentType == TrackerType::OBJECT_TRACKER) {
        return m_objectTracker->update(frame);
    }
    else {
        // Используем публичный метод обновления FaceTracker
        return m_faceTracker->update(frame);
    }
}

void TrackerManager::drawTrackingInfo(cv::Mat& frame) const {
    if (m_currentType == TrackerType::OBJECT_TRACKER) {
        m_objectTracker->drawTrackingInfo(frame);
    }
    else {
        // Используем публичный метод отрисовки FaceTracker
        m_faceTracker->drawTrackingInfo(frame);
    }
}

void TrackerManager::reset() {
    if (m_currentType == TrackerType::OBJECT_TRACKER) {
        m_objectTracker->reset();
    }
    else {
        // Используем публичный метод сброса FaceTracker
        m_faceTracker->reset();
    }
}

bool TrackerManager::isInitialized() const {
    if (m_currentType == TrackerType::OBJECT_TRACKER) {
        return m_objectTracker->isInitialized();
    }
    else {
        // Используем публичный метод проверки инициализации FaceTracker
        return m_faceTracker->isInitialized();
    }
}

void TrackerManager::selectNextTrg(){
    switch (m_currentType) {
    case TrackerType::OBJECT_TRACKER:
        //m_objectTracker
        break;

    case TrackerType::FACE_TRACKER:
        std::cout << "Select next target\n";
        m_faceTracker->selectNextTrg();
        break;
    }
}

void TrackerManager::selectPrevTrg() {
    switch (m_currentType) {
    case TrackerType::OBJECT_TRACKER:
        //m_objectTracker
        break;

    case TrackerType::FACE_TRACKER:
        std::cout << "Select prev target\n";
        m_faceTracker->selectPrevTrg();
        break;
    }
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
#include "TrackerManager.h"
#include <iostream>

TrackerManager::TrackerManager()
    : m_currentType_(TrackerType::OBJECT_TRACKER) {
    initializeTrackers();
}

void TrackerManager::initializeTrackers() {
    m_objectTracker_ = std::make_unique<ObjectTracker>();
    m_faceTracker_ = std::make_unique<FaceTracker>();
    m_lineModTracker_ = std::make_unique<LineModTracker>();
}

void TrackerManager::switchTracker(TrackerType type) {
    if (type == m_currentType_) {
        return;
    }

    m_currentType_ = type;

    // —брасываем оба трекера при переключении
    if (m_objectTracker_) {
        m_objectTracker_->reset();
    }
    if (m_faceTracker_) {
        m_faceTracker_->reset();
    }
    if (m_lineModTracker_) {
        m_lineModTracker_->reset();
    }

    std::cout << "Switched to " << type << std::endl;
}

TrackerManager::TrackerType TrackerManager::getCurrentTrackerType() const {
    return m_currentType_;
}

bool TrackerManager::initialize(const cv::Mat& frame) {
    switch (m_currentType_) {
    case TrackerType::OBJECT_TRACKER:
        return m_objectTracker_->initialize(frame);
    case TrackerType::FACE_TRACKER:
        return m_faceTracker_->initialize(frame);
    case TrackerType::LINEMOD_TRACKER:
        return m_lineModTracker_->initialize(frame);
    }
    return false;
}

bool TrackerManager::update(const cv::Mat& frame) {
    switch (m_currentType_) {
    case TrackerType::OBJECT_TRACKER:
        return m_objectTracker_->update(frame);
    case TrackerType::FACE_TRACKER:
        return m_faceTracker_->update(frame);
    case TrackerType::LINEMOD_TRACKER:
        return m_lineModTracker_->update(frame);
    }
    return false;
}

void TrackerManager::drawTrackingInfo(cv::Mat& frame) const {
    switch (m_currentType_) {
    case TrackerType::OBJECT_TRACKER:
        m_objectTracker_->drawTrackingInfo(frame);
        break;
    case TrackerType::FACE_TRACKER:
        m_faceTracker_->drawTrackingInfo(frame);
        break;
    case TrackerType::LINEMOD_TRACKER:
        m_lineModTracker_->drawTrackingInfo(frame);
        break;
    }
}

void TrackerManager::reset() {
    switch (m_currentType_) {
    case TrackerType::OBJECT_TRACKER:
        m_objectTracker_->reset();
        break;
    case TrackerType::FACE_TRACKER:
        m_faceTracker_->reset();
        break;
    case TrackerType::LINEMOD_TRACKER:
        m_lineModTracker_->reset();
        break;
    }
}

bool TrackerManager::isInitialized() const {
    switch (m_currentType_) {
    case TrackerType::OBJECT_TRACKER:
        return m_objectTracker_->isInitialized();
    case TrackerType::FACE_TRACKER:
        return m_faceTracker_->isInitialized();
    case TrackerType::LINEMOD_TRACKER:
        return m_lineModTracker_->isInitialized();
    }
    return false;
}

void TrackerManager::selectPrevTrg() {
    switch (m_currentType_) {
    case TrackerType::OBJECT_TRACKER:
        //m_objectTracker
        break;

    case TrackerType::FACE_TRACKER:
        std::cout << "Select prev target\n";
        m_faceTracker_->selectPrevTrg();
        break;
    }
}

void TrackerManager::selectNextTrg(){
    switch (m_currentType_) {
    case TrackerType::OBJECT_TRACKER:
        //m_objectTracker
        break;

    case TrackerType::FACE_TRACKER:
        std::cout << "Select next target\n";
        m_faceTracker_->selectNextTrg();
        break;
    }
}



void TrackerManager::selectPrevTrkMode() {
    switch (m_currentType_) {
    case TrackerType::OBJECT_TRACKER:
        //m_objectTracker
        break;

    case TrackerType::FACE_TRACKER:
        m_faceTracker_->selectPrevTrkMode();
        break;
    }
}

void TrackerManager::selectNextTrkMode() {
    switch (m_currentType_) {
    case TrackerType::OBJECT_TRACKER:
        //m_objectTracker
        break;

    case TrackerType::FACE_TRACKER:
        m_faceTracker_->selectNextTrkMode();
        break;
    }
}

ObjectTracker* TrackerManager::getObjectTracker() {
    return m_objectTracker_.get();
}

FaceTracker* TrackerManager::getFaceTracker() {
    return m_faceTracker_.get();
}

LineModTracker* TrackerManager::getLineModTracker() {
    return m_lineModTracker_.get();
}

void TrackerManager::setFaceTrackingMode(bool enable) {
    if (m_objectTracker_) {
        m_objectTracker_->setFaceTrackingMode(enable);
    }
}

bool TrackerManager::isFaceTrackingMode() const {
    if (m_objectTracker_) {
        return m_objectTracker_->isFaceTrackingMode();
    }
    return false;
}
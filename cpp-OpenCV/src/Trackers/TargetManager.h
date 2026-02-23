#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <chrono>
#include <algorithm>
#include <type_traits>

enum class TrackerMode {
    src, // Поиск всех объектов на изображении
    trc, // Слежение за 1 объектом

};


enum class TargetStatus {
    find,
    lock,
    softlock,
    lost,
};

inline std::ostream& operator<<(std::ostream& os, TargetStatus c) {
    switch (c) {
    case TargetStatus::find:   os << "find"; break;
    case TargetStatus::lock: os << "lock"; break;
    case TargetStatus::softlock:  os << "softlock"; break;
    case TargetStatus::lost:  os << "lost"; break;
    default:           os << "Unknown"; break;
    }
    return os;
}

struct TrackedFace {
    cv::Rect boundingBox;
    cv::Point2f center;
    cv::Point2f velocity;
    cv::Point2f predictedCenter;
    int id;
    int age;
    std::deque<cv::Point2f> positionHistory;
    cv::Point2f previousPosition;
    std::chrono::steady_clock::time_point previousTime;
    bool hasPreviousPosition;
    int lostFrames = 0;
    bool matched = false; // временный флаг для сопоставления
    TargetStatus currentStatus = TargetStatus::find;
    std::chrono::steady_clock::time_point lostTime;
    bool lostTimeSet = false;
    float lostLifetimeSec = 1.5f;

    TrackedFace() : id(0), age(0), currentStatus(TargetStatus::find), hasPreviousPosition(false) {
        boundingBox = cv::Rect(0, 0, 0, 0);
        center = cv::Point2f(0, 0);
        velocity = cv::Point2f(0, 0);
        predictedCenter = cv::Point2f(0, 0);
        previousPosition = cv::Point2f(0, 0);
    }

    bool IsLost() const {
        return currentStatus == TargetStatus::lost;
    }
};

class TargetManager {
public:
    TargetManager();

    // Управление списком лиц
    void addFace(const TrackedFace& face);
    void updateFace(int id, const TrackedFace& newData);
    void removeFace(int id);
    void removeLostFaces(float maxLostTimeSec);
    void clear();

    // Доступ к данным
    const std::vector<TrackedFace>& getFaces() const;
    int getSelectedId() const;
    const TrackedFace* getSelectedFace() const;

    // Переключение выбранного лица
    void selectNext();
    void selectPrev();

    TrackedFace getFaceByIndex(int i);
    TrackedFace* getFaceById(int id);

private:
    std::vector<TrackedFace> faces_;
    int selectedId_;

    void setSelectedFace(int id);
};
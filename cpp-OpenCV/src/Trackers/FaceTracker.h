#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <deque>
#include "./global.h"
#include "Renderer.h"

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

// Структура TrackedFace остаётся без изменений
struct TrackedFace {
    cv::Rect boundingBox;
    cv::Point2f center;
    cv::Point2f velocity;
    cv::Point2f predictedCenter;
    cv::Point2f predicted = { 0,0 };
    int id;
    int age;
    std::deque<cv::Point2f> positionHistory;

    cv::Point2f previousPosition;
    std::chrono::steady_clock::time_point previousTime;
    bool hasPreviousPosition;

    int lostFrames = 0;
    bool matched = false;

    TargetStatus currentStatus = TargetStatus::find;

    // Время жизни после lost
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

class FaceTracker {
public:
    FaceTracker();
    ~FaceTracker() = default;

    bool update(const cv::Mat& frame);
    bool initialize(const cv::Mat& frame);

    // Метод отрисовки теперь делегирует работу рендереру
    void drawTrackingInfo(cv::Mat& frame) const;

    void reset();
    bool isInitialized() const;
    std::vector<TrackedFace> getTrackedFaces() const;
    cv::Point2f getLargestFaceCenter() const;
    TrackedFace getLargestFace() const;

    const std::vector<TrackedFace>& getTrackedFacesRef() const { return trackedFaces; }

private:
    std::vector<TrackedFace> trackedFaces;
    int nextFaceId;
    bool initialized;

    // Параметры детектирования
    cv::CascadeClassifier faceCascade;
    double scaleFactor;
    int minNeighbors;
    cv::Size minSize;
    cv::Size maxSize;

    // Параметры прогнозирования
    float predictionTime;
    const int maxHistorySize = 10;

    // Для расчёта времени между кадрами
    std::chrono::steady_clock::time_point previousTime;
    bool hasPreviousTime;

    // ---------- Удалены все графические поля и константы ----------
    // Вместо них – экземпляр рендерера
    Renderer renderer;

    int maxLostFrames = 15;

    float smoothAlpha = 0.25f;

    // Приватные методы
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame);

    std::vector<cv::Rect> nonMaximumSuppression(const std::vector<cv::Rect>& faces, float iouThreshold) const;

    cv::Rect getLargestFaceRect(const std::vector<cv::Rect>& faces);

    void updateFaceTracking(const std::vector<cv::Rect>& detectedFaces, float deltaTime);
    void updateFacePosition(TrackedFace& face, const cv::Rect& newRect, float deltaTime);
    void updateVelocity(TrackedFace& face, const cv::Point2f& newPosition, float deltaTime);
    void updateFaceVelocity(TrackedFace& face, const cv::Point2f& newPosition, float deltaTime);
    void updatePositionHistory(TrackedFace& face, const cv::Point2f& newPosition);
    cv::Point2f getSmoothedPosition(const TrackedFace& face) const;
    cv::Point2f getPredictedPosition(const TrackedFace& face) const;

    // ---------- Удалены методы, связанные с отрисовкой ----------
    // getClosestPointOnRect, getPointOnCircle – теперь в рендерере

    float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) const;
    int findClosestFace(const cv::Point2f& center, const std::vector<TrackedFace>& faces) const;
    void removeOldFaces();
    static float computeIOU(const cv::Rect& a, const cv::Rect& b);
};
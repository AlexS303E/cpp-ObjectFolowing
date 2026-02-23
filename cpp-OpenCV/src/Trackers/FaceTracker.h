#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <deque>
#include "global.h"
#include "Renderer.h"
#include "MasterTracker.h"

class FaceTracker : public MasterTracker {
public:
    FaceTracker();
    ~FaceTracker() override = default;

    bool update(const cv::Mat& frame) override;

    bool updateSrc(const cv::Mat& frame);
    bool updateTrc(const cv::Mat& frame);

    void updateFaceTracking(const std::vector<cv::Rect>& detectedFaces, float deltaTime);
    bool initialize(const cv::Mat& frame);

    void drawTrackingInfo(cv::Mat& frame) const override;
    void reset() override;
    bool isInitialized() const override;

    // Методы для работы с выбранным лицом (используют TargetManager)
    void selectNextTrg() override;
    void selectPrevTrg() override;

    // Доступ к данным через TargetManager
    std::vector<TrackedFace> getTrackedFaces() const;
    cv::Point2f getLargestFaceCenter() const;
    TrackedFace getLargestFace() const;

private:
    int nextFaceId;
    bool initialized;

    cv::CascadeClassifier faceCascade;
    double scaleFactor;
    int minNeighbors;
    cv::Size minSize;
    cv::Size maxSize;

    float predictionTime;
    const int maxHistorySize = 10;
    std::chrono::steady_clock::time_point previousTime;
    bool hasPreviousTime;
    int maxLostFrames = 15;
    float smoothAlpha = 0.25f;

    Renderer renderer;

    // Вспомогательные методы (без изменений)
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame);
    std::vector<cv::Rect> nonMaximumSuppression(const std::vector<cv::Rect>& faces, float iouThreshold) const;
    cv::Rect getLargestFaceRect(const std::vector<cv::Rect>& faces);
    void updateFacePosition(TrackedFace& face, const cv::Rect& newRect, float deltaTime);
    void updateVelocity(TrackedFace& face, const cv::Point2f& newPosition, float deltaTime);
    void updateFaceVelocity(TrackedFace& face, const cv::Point2f& newPosition, float deltaTime);
    void updatePositionHistory(TrackedFace& face, const cv::Point2f& newPosition);
    cv::Point2f getSmoothedPosition(const TrackedFace& face) const;
    cv::Point2f getPredictedPosition(const TrackedFace& face) const;
    float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) const;
    int findClosestFace(const cv::Point2f& center, const std::vector<TrackedFace>& faces) const;
    static float computeIOU(const cv::Rect& a, const cv::Rect& b);

    // Синхронизация статуса выбранного лица
    void syncSelectedStatus();
};
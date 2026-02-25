// ContourTracker.h
#pragma once

#include "MasterTracker.h"
#include "Renderer.h"
#include <array>
#include <unordered_map>

class ContourTracker : public MasterTracker {
public:
    ContourTracker();
    ~ContourTracker() override = default;

    bool update(const cv::Mat& frame) override;
    bool initialize(const cv::Mat& frame);
    void reset() override;
    bool isInitialized() const override;
    void drawTrackingInfo(cv::Mat& frame) const override;

    void selectNextTrg() override { MasterTracker::selectNextTrg(); syncSelectedStatus(); }
    void selectPrevTrg() override { MasterTracker::selectPrevTrg(); syncSelectedStatus(); }

private:
    bool initialized;
    int nextObjectId;

    // Параметры цветовой сегментации кожи (HSV)
    int hueLow, hueHigh;
    int satLow, satHigh;
    int valLow, valHigh;

    double minContourArea;
    double maxContourArea;
    float minAspectRatio;
    float maxAspectRatio;

    double huComparisonThreshold;          // порог для сравнения Hu-моментов
    double positionDistanceWeight;         // вес расстояния при сопоставлении

    std::chrono::steady_clock::time_point previousTime;
    bool hasPreviousTime;

    int globalDetectionPeriod;
    int framesSinceLastGlobal;

    int baseSearchMargin;
    int maxSearchMargin;
    float velocityScale;
    int maxLostFrames;
    float smoothAlpha;
    float predictionTime;

    bool firstTrcFrame;
    Renderer renderer;

    // Соответствие ID -> Hu-моменты
    std::unordered_map<int, std::array<double, 7>> huMomentsMap;

    // Вспомогательные методы
    cv::Mat createSkinMask(const cv::Mat& frame) const;
    std::vector<std::vector<cv::Point>> findSkinContours(const cv::Mat& frame) const;
    std::vector<cv::Rect> contoursToRects(const std::vector<std::vector<cv::Point>>& contours) const;
    std::array<double, 7> computeHuMoments(const std::vector<cv::Point>& contour) const;
    double compareHuMoments(const std::array<double, 7>& hu1, const std::array<double, 7>& hu2) const;

    void updateSrc(const cv::Mat& frame);
    void updateTrc(const cv::Mat& frame);
    void syncSelectedStatus();

    void updateFaceVelocity(TrackedFace& face, const cv::Point2f& newPosition, float deltaTime);
    cv::Point2f getPredictedPosition(const TrackedFace& face) const;
    float computeIOU(const cv::Rect& a, const cv::Rect& b) const;
    std::vector<cv::Rect> nonMaximumSuppression(const std::vector<cv::Rect>& rects, float iouThreshold) const;
};
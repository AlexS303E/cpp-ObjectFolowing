#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct TrackedFace;

class Renderer {
public:
    Renderer();

    // Главный метод отрисовки всей информации
    void draw(cv::Mat& frame, const std::vector<TrackedFace>& faces, bool isInitialized = true) const;

    // Настройка цветов (опционально)
    void setFaceColor(const cv::Scalar& color);
    void setPredictionColor(const cv::Scalar& color);
    void setLostColor(const cv::Scalar& color);
    // ... аналогично для других цветов

private:
    // Цвета
    cv::Scalar faceColor;
    cv::Scalar innerFaceColor;
    cv::Scalar lineColor;
    cv::Scalar textColor;
    cv::Scalar predictionColor;
    cv::Scalar lostColor;       // для потерянных лиц

    // Константы отрисовки (раньше были в FaceTracker)
    int PREDICTION_RADIUS;
    int BORDER_THICKNESS;
    int INNER_BORDER_THICKNESS;
    int PREDICTION_THICKNESS;
    int LINE_THICKNESS;

    // Вспомогательные методы
    void drawTargetLock(cv::Mat& frame, const TrackedFace& face) const;
    void drawTargetSoftLock(cv::Mat& frame, const TrackedFace& face) const;
    void drawTargetFind(cv::Mat& frame, const TrackedFace& face) const;

    void drawInfoPanel(cv::Mat& frame, const std::vector<TrackedFace>& faces) const;

    // Геометрические вычисления (перенесены из FaceTracker)
    cv::Point2f getClosestPointOnRect(const cv::Rect& rect, const cv::Point2f& point) const;
    cv::Point2f getPointOnCircle(const cv::Point2f& circleCenter,
        const cv::Point2f& targetPoint,
        float radius) const;
};
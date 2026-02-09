#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <deque>
#include "./global.h"

// Структура для отслеживаемого лица с прогнозированием
struct TrackedFace {
    cv::Rect boundingBox;          // Прямоугольник лица
    cv::Point2f center;            // Текущий центр
    cv::Point2f velocity;          // Скорость (пикселей в секунду)
    cv::Point2f predictedCenter;   // Прогнозируемый центр через 1 секунду
    int id;                        // Уникальный идентификатор
    int age;                       // Возраст в кадрах
    bool lost;                     // Флаг потери лица
    std::deque<cv::Point2f> positionHistory;  // История позиций для сглаживания

    TrackedFace() : id(0), age(0), lost(false) {
        boundingBox = cv::Rect(0, 0, 0, 0);
        center = cv::Point2f(0, 0);
        velocity = cv::Point2f(0, 0);
        predictedCenter = cv::Point2f(0, 0);
    }
};

class FaceTracker {
public:
    FaceTracker();
    ~FaceTracker() = default;

    // Основной метод обновления трекинга
    bool update(const cv::Mat& frame);

    // Инициализация трекинга по первому кадру
    bool initialize(const cv::Mat& frame);

    // Отрисовка всех отслеживаемых лиц с прогнозами
    void drawTrackingInfo(cv::Mat& frame) const;

    // Сброс всех трекеров
    void reset();

    // Проверка инициализации
    bool isInitialized() const;

    // Получение всех отслеживаемых лиц
    std::vector<TrackedFace> getTrackedFaces() const;

    // Получение центра самого большого лица
    cv::Point2f getLargestFaceCenter() const;

    // Получение самого большого лица
    TrackedFace getLargestFace() const;

private:
    std::vector<TrackedFace> trackedFaces;
    int nextFaceId;
    bool initialized;

    // Параметры детектирования
    cv::CascadeClassifier faceCascade;
    cv::CascadeClassifier profileFaceCascade;
    double scaleFactor;
    int minNeighbors;
    cv::Size minSize;
    cv::Size maxSize;

    // Параметры прогнозирования
    float predictionTime;          // Время прогноза (1 секунда)
    const int maxHistorySize = 10; // Максимальный размер истории позиций

    // Для расчета времени между кадрами
    std::chrono::steady_clock::time_point previousTime;
    bool hasPreviousTime;

    // Цвета для отрисовки
    cv::Scalar faceColor;
    cv::Scalar innerFaceColor;
    cv::Scalar circleColor;
    cv::Scalar lineColor;
    cv::Scalar textColor;
    cv::Scalar predictionColor;

    // Параметры отрисовки
    const int PREDICTION_RADIUS = 26;
    const int BORDER_THICKNESS = 3;
    const int INNER_BORDER_THICKNESS = 1;
    const int CIRCLE_THICKNESS = 4;
    const int PREDICTION_THICKNESS = 3;
    const int LINE_THICKNESS = 2;

    bool enableFastTracking;
    int detectionSkipFrames;
    int framesSinceLastDetection;
    float trackingConfidenceThreshold;
    cv::Rect searchROI;

    // Приватные методы
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame);
    cv::Rect getLargestFaceRect(const std::vector<cv::Rect>& faces);

    // Методы для трекинга и прогнозирования
    void updateFaceTracking(const std::vector<cv::Rect>& detectedFaces, float deltaTime);
    void updateFacePosition(TrackedFace& face, const cv::Rect& newRect, float deltaTime);
    void updateVelocity(TrackedFace& face, const cv::Point2f& newPosition, float deltaTime);
    void updatePositionHistory(TrackedFace& face, const cv::Point2f& newPosition);
    cv::Point2f getSmoothedPosition(const TrackedFace& face) const;
    cv::Point2f getPredictedPosition(const TrackedFace& face) const;

    // Методы для отрисовки
    cv::Point2f getClosestPointOnRect(const cv::Rect& rect, const cv::Point2f& point) const;
    cv::Point2f getPointOnCircle(const cv::Point2f& circleCenter,
        const cv::Point2f& targetPoint,
        float radius) const;

    // Вспомогательные методы
    float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) const;
    int findClosestFace(const cv::Point2f& center, const std::vector<TrackedFace>& faces) const;
    void removeOldFaces();
};
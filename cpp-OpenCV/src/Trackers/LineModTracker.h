#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/rgbd/linemod.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <deque>
#include "MasterTracker.h"
#include "Renderer.h"

class LineModTracker : public MasterTracker {
public:
    LineModTracker();
    ~LineModTracker() override = default;

    // Инициализация: загрузка шаблонов и первичная детекция
    bool initialize(const cv::Mat& frame);

    // Основной цикл обновления (выбирает режим src/trc)
    bool update(const cv::Mat& frame) override;

    // Режим поиска всех объектов на кадре
    bool updateSrc(const cv::Mat& frame);

    // Режим слежения за одной целью
    bool updateTrc(const cv::Mat& frame);

    // Отрисовка информации
    void drawTrackingInfo(cv::Mat& frame) const override;

    // Сброс трекера
    void reset() override;

    // Проверка инициализации
    bool isInitialized() const override;

    // Переопределение методов выбора цели (опционально)
    void selectNextTrg() override;
    void selectPrevTrg() override;

    // Доступ к списку отслеживаемых объектов
    std::vector<TrackedFace> getTrackedObjects() const;

    // Загрузка шаблонов LineMOD из файлов
    bool loadTemplates(const std::vector<std::string>& templatePaths);

private:
    bool hasPreviousTime;
    int maxLostFrames = 15;
    float predictionTime;

    // Поля для LineMOD
    cv::Ptr<cv::linemod::Detector> detector_;
    std::vector<cv::Mat> templates_;          // загруженные шаблоны
    std::vector<std::string> classIds_;        // идентификаторы классов (например, имена объектов)
    float matchThreshold_;                      // порог совпадения для детекции

    std::vector<cv::Size> templateSizes_;  // размеры шаблонов

    // Параметры трекинга
    int nextObjectId_;
    bool initialized_;
    std::chrono::steady_clock::time_point previousTime_;
    bool hasPreviousTime_;
    int maxLostFrames_;
    float predictionTime_;
    const int maxHistorySize_ = 10;

    // Рендерер (можно использовать общий или свой)
    Renderer renderer_;

    // Вспомогательные методы
    std::vector<cv::Rect> detectObjects(const cv::Mat& frame);
    void updateObjectPosition(TrackedFace& obj, const cv::Rect& newRect, float deltaTime);
    void updateVelocity(TrackedFace& obj, const cv::Point2f& newPosition, float deltaTime);
    void updatePositionHistory(TrackedFace& obj, const cv::Point2f& newPosition);
    cv::Point2f getPredictedPosition(const TrackedFace& obj) const;
    float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) const;
    static float computeIOU(const cv::Rect& a, const cv::Rect& b);

    // Синхронизация статуса выбранной цели
    void syncSelectedStatus();

    std::vector<cv::Rect> LineModTracker::nonMaximumSuppression(const std::vector<cv::Rect>& rects, float threshold);
};
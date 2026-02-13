#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

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

class TargetManager {
public:
    TargetManager();

    // Обновить внутренний список копией актуальных лиц из трекера
    void update(const std::vector<TrackedFace>& faces);

    // Переключиться на следующее лицо в списке (циклически)
    void selectNext();

    // Переключиться на предыдущее лицо
    void selectPrev();

    // Получить ID выбранного лица (-1, если ничего не выбрано)
    int getSelectedId() const;

    // Получить указатель на выбранное лицо (nullptr, если нет)
    const TrackedFace* getSelectedFace() const;

    // Получить внутренний список лиц (для отрисовки и т.п.)
    const std::vector<TrackedFace>& getFaces() const;

private:
    std::vector<TrackedFace> faces_;   // копия списка лиц
    int selectedId_;                    // ID выбранного лица (-1 = нет)

    // Вспомогательный метод: устанавливает выбранное лицо по ID,
    // снимает выделение с предыдущего и меняет статусы
    void setSelectedFace(int id);
};
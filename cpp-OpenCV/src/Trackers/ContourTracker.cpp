// ContourTracker.cpp
#include "ContourTracker.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

ContourTracker::ContourTracker()
    : initialized(false),
    nextObjectId(1),
    hueLow(0), hueHigh(50),
    satLow(50), satHigh(150),
    valLow(50), valHigh(255),
    minContourArea(500),
    maxContourArea(10000),
    minAspectRatio(0.8f),
    maxAspectRatio(1.5f),
    huComparisonThreshold(0.5),
    positionDistanceWeight(0.5),
    hasPreviousTime(false),
    globalDetectionPeriod(5),
    framesSinceLastGlobal(0),
    baseSearchMargin(80),
    maxSearchMargin(250),
    velocityScale(0.5f),
    maxLostFrames(15),
    smoothAlpha(0.3f),
    predictionTime(1.0f),
    firstTrcFrame(true)
{
    renderer.setFaceColor(cv::Scalar(255, 0, 0));   // синий для контурного трекера
    renderer.setPredictionColor(cv::Scalar(255, 0, 0));
    renderer.setLostColor(cv::Scalar(0, 0, 255));
}

bool ContourTracker::initialize(const cv::Mat& frame) {
    if (frame.empty()) {
        std::cerr << "ContourTracker::initialize: frame is empty" << std::endl;
        return false;
    }

    // Детектируем объекты на первом кадре
    std::vector<std::vector<cv::Point>> contours = findSkinContours(frame);
    std::vector<cv::Rect> rects = contoursToRects(contours);

    if (rects.empty()) {
        std::cout << "ContourTracker::initialize: no objects detected" << std::endl;
        return false;
    }

    auto currentTime = std::chrono::steady_clock::now();

    // Добавляем все обнаруженные объекты
    for (size_t i = 0; i < rects.size(); ++i) {
        TrackedFace obj;
        obj.id = nextObjectId++;
        obj.boundingBox = rects[i];
        obj.center = cv::Point2f(rects[i].x + rects[i].width / 2.0f,
            rects[i].y + rects[i].height / 2.0f);
        obj.age = 1;
        obj.currentStatus = TargetStatus::find;
        obj.positionHistory.push_back(obj.center);
        obj.previousPosition = obj.center;
        obj.velocity = cv::Point2f(0, 0);
        obj.predictedCenter = obj.center;
        obj.hasPreviousPosition = false;
        obj.lostFrames = 0;
        obj.lostTimeSet = false;

        targetManager_.addFace(obj);

        // Вычисляем и сохраняем Hu-моменты
        std::array<double, 7> hu = computeHuMoments(contours[i]);
        huMomentsMap[obj.id] = hu;
    }

    // Выбираем первый объект и устанавливаем ему статус softlock
    if (!targetManager_.getFaces().empty()) {
        targetManager_.selectNext();
        int selectedId = targetManager_.getSelectedId();
        if (selectedId != -1) {
            TrackedFace* facePtr = targetManager_.getFaceById(selectedId);
            if (facePtr) {
                facePtr->currentStatus = TargetStatus::softlock;
                targetManager_.updateFace(selectedId, *facePtr);
                std::cout << "ContourTracker: selected object ID " << selectedId << " set to softlock" << std::endl;
            }
        }
    }

    initialized = true;
    previousTime = currentTime;
    hasPreviousTime = true;

    std::cout << "ContourTracker initialized with " << targetManager_.getFaces().size() << " objects" << std::endl;
    return true;
}

bool ContourTracker::update(const cv::Mat& frame) {
    if (!initialized) {
        std::cerr << "ContourTracker not initialized!" << std::endl;
        return false;
    }

    switch (trackerMode_) {
    case TrackerMode::src:
        updateSrc(frame);
        break;
    case TrackerMode::trc:
        updateTrc(frame);
        break;
    }
    return true; // или проверка наличия объектов
}

void ContourTracker::updateSrc(const cv::Mat& frame) {
    if (!initialized || frame.empty()) return;

    if (targetManager_.getMode() != TrackerMode::src) {
        targetManager_.setMode(TrackerMode::src);
    }

    // Расчёт deltaTime
    auto currentTime = std::chrono::steady_clock::now();
    float deltaTime = 0.033f;
    if (hasPreviousTime) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - previousTime);
        deltaTime = elapsed.count() / 1000000.0f;
        deltaTime = std::clamp(deltaTime, 0.001f, 0.1f);
    }
    hasPreviousTime = true;
    previousTime = currentTime;

    std::vector<TrackedFace> currentFaces = targetManager_.getFaces();
    std::vector<bool> faceMatched(currentFaces.size(), false);

    // --- 1. Локальный трекинг (каждый кадр) ---
    for (size_t i = 0; i < currentFaces.size(); ++i) {
        TrackedFace& face = currentFaces[i];
        if (face.currentStatus == TargetStatus::lost) continue;

        cv::Point2f searchCenter = getPredictedPosition(face);
        int margin = baseSearchMargin;
        cv::Rect searchArea(
            std::max(0, (int)searchCenter.x - margin),
            std::max(0, (int)searchCenter.y - margin),
            margin * 2, margin * 2
        );
        // Корректировка границ
        if (searchArea.x + searchArea.width > frame.cols)
            searchArea.width = frame.cols - searchArea.x;
        if (searchArea.y + searchArea.height > frame.rows)
            searchArea.height = frame.rows - searchArea.y;

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Rect> detectedRects;
        std::vector<std::array<double, 7>> detectedHu;

        if (searchArea.area() > 0) {
            cv::Mat roi = frame(searchArea);
            std::vector<std::vector<cv::Point>> roiContours = findSkinContours(roi);
            for (auto& c : roiContours) {
                // Смещаем контур обратно в координаты кадра
                for (auto& pt : c) {
                    pt.x += searchArea.x;
                    pt.y += searchArea.y;
                }
                cv::Rect r = cv::boundingRect(c);
                detectedRects.push_back(r);
                detectedHu.push_back(computeHuMoments(c));
            }
        }

        if (!detectedRects.empty()) {
            // Ищем лучшее совпадение по комбинации расстояния и разницы моментов
            int bestIdx = -1;
            double bestScore = std::numeric_limits<double>::max();
            auto it = huMomentsMap.find(face.id);
            if (it != huMomentsMap.end()) {
                const auto& refHu = it->second;
                for (size_t j = 0; j < detectedRects.size(); ++j) {
                    cv::Point2f c(detectedRects[j].x + detectedRects[j].width / 2.0f,
                        detectedRects[j].y + detectedRects[j].height / 2.0f);
                    double dist = cv::norm(c - searchCenter);
                    double huDist = compareHuMoments(refHu, detectedHu[j]);
                    double score = dist + positionDistanceWeight * huDist; // комбинированная метрика
                    if (score < bestScore) {
                        bestScore = score;
                        bestIdx = static_cast<int>(j);
                    }
                }
            }
            if (bestIdx != -1 && bestScore < huComparisonThreshold * 100) { // эмпирический порог
                // Обновляем лицо
                face.lostFrames = 0;
                face.currentStatus = TargetStatus::find;

                // Экспоненциальное сглаживание
                const cv::Rect& bestRect = detectedRects[bestIdx];
                face.boundingBox.x = (1.0f - smoothAlpha) * face.boundingBox.x + smoothAlpha * bestRect.x;
                face.boundingBox.y = (1.0f - smoothAlpha) * face.boundingBox.y + smoothAlpha * bestRect.y;
                face.boundingBox.width = (1.0f - smoothAlpha) * face.boundingBox.width + smoothAlpha * bestRect.width;
                face.boundingBox.height = (1.0f - smoothAlpha) * face.boundingBox.height + smoothAlpha * bestRect.height;

                cv::Point2f newCenter(face.boundingBox.x + face.boundingBox.width / 2.0f,
                    face.boundingBox.y + face.boundingBox.height / 2.0f);
                updateFaceVelocity(face, newCenter, deltaTime);
                face.positionHistory.push_back(newCenter);
                if (face.positionHistory.size() > 10) face.positionHistory.pop_front();
                face.center = newCenter;
                face.age++;
                face.predictedCenter = getPredictedPosition(face);
                faceMatched[i] = true;

                // Обновляем эталонные моменты (сглаживание)
                if (it != huMomentsMap.end()) {
                    for (int k = 0; k < 7; ++k) {
                        it->second[k] = (1.0 - smoothAlpha) * it->second[k] + smoothAlpha * detectedHu[bestIdx][k];
                    }
                }
            }
        }

        if (!faceMatched[i]) {
            face.lostFrames++;
            if (face.lostFrames > maxLostFrames) {
                face.currentStatus = TargetStatus::lost;
                if (!face.lostTimeSet) {
                    face.lostTime = currentTime;
                    face.lostTimeSet = true;
                }
            }
            else {
                face.predictedCenter = getPredictedPosition(face);
            }
        }
    }

    // --- 2. Периодическая глобальная детекция ---
    framesSinceLastGlobal++;
    if (framesSinceLastGlobal >= globalDetectionPeriod) {
        framesSinceLastGlobal = 0;

        std::vector<std::vector<cv::Point>> globalContours = findSkinContours(frame);
        std::vector<cv::Rect> globalRects = contoursToRects(globalContours);
        globalRects = nonMaximumSuppression(globalRects, 0.3f);
        std::vector<bool> used(globalRects.size(), false);

        // Сопоставление с существующими лицами
        for (size_t i = 0; i < currentFaces.size(); ++i) {
            TrackedFace& face = currentFaces[i];
            if (face.currentStatus == TargetStatus::lost) continue;

            int bestIdx = -1;
            double bestIOU = 0.3;
            for (size_t j = 0; j < globalRects.size(); ++j) {
                if (used[j]) continue;
                float iou = computeIOU(face.boundingBox, globalRects[j]);
                if (iou > bestIOU) {
                    bestIOU = iou;
                    bestIdx = static_cast<int>(j);
                }
            }
            if (bestIdx != -1) {
                used[bestIdx] = true;
                // Обновление позиции (аналогично локальному)
                const cv::Rect& newRect = globalRects[bestIdx];
                face.boundingBox.x = (1.0f - smoothAlpha) * face.boundingBox.x + smoothAlpha * newRect.x;
                face.boundingBox.y = (1.0f - smoothAlpha) * face.boundingBox.y + smoothAlpha * newRect.y;
                face.boundingBox.width = (1.0f - smoothAlpha) * face.boundingBox.width + smoothAlpha * newRect.width;
                face.boundingBox.height = (1.0f - smoothAlpha) * face.boundingBox.height + smoothAlpha * newRect.height;

                cv::Point2f newCenter(face.boundingBox.x + face.boundingBox.width / 2.0f,
                    face.boundingBox.y + face.boundingBox.height / 2.0f);
                if (!faceMatched[i]) {
                    updateFaceVelocity(face, newCenter, deltaTime);
                    face.positionHistory.push_back(newCenter);
                    if (face.positionHistory.size() > 10) face.positionHistory.pop_front();
                    face.center = newCenter;
                    face.predictedCenter = getPredictedPosition(face);
                }
                face.lostFrames = 0;
                face.currentStatus = TargetStatus::find;
                face.age++;
                faceMatched[i] = true;

                // Обновление моментов (по контуру, соответствующему rect)
                auto it = std::find_if(globalContours.begin(), globalContours.end(),
                    [&](const std::vector<cv::Point>& c) {
                        return cv::boundingRect(c) == newRect;
                    });
                if (it != globalContours.end()) {
                    auto newHu = computeHuMoments(*it);
                    auto mit = huMomentsMap.find(face.id);
                    if (mit != huMomentsMap.end()) {
                        for (int k = 0; k < 7; ++k)
                            mit->second[k] = (1.0 - smoothAlpha) * mit->second[k] + smoothAlpha * newHu[k];
                    }
                }
            }
        }

        // Добавление новых лиц
        for (size_t j = 0; j < globalRects.size(); ++j) {
            if (!used[j]) {
                TrackedFace newFace;
                newFace.id = nextObjectId++;
                newFace.boundingBox = globalRects[j];
                newFace.center = cv::Point2f(globalRects[j].x + globalRects[j].width / 2.0f,
                    globalRects[j].y + globalRects[j].height / 2.0f);
                newFace.previousPosition = newFace.center;
                newFace.velocity = cv::Point2f(0, 0);
                newFace.predictedCenter = newFace.center;
                newFace.age = 1;
                newFace.lostFrames = 0;
                newFace.currentStatus = TargetStatus::find;
                newFace.hasPreviousPosition = false;
                newFace.positionHistory.push_back(newFace.center);
                currentFaces.push_back(newFace);

                // Найти соответствующий контур и вычислить моменты
                auto it = std::find_if(globalContours.begin(), globalContours.end(),
                    [&](const std::vector<cv::Point>& c) {
                        return cv::boundingRect(c) == globalRects[j];
                    });
                if (it != globalContours.end()) {
                    huMomentsMap[newFace.id] = computeHuMoments(*it);
                }
            }
        }
    }

    // Обновление targetManager_
    targetManager_.setFaces(currentFaces);
    targetManager_.removeLostFaces(1.5f);
    syncSelectedStatus();

    if (targetManager_.getMode() == TrackerMode::src) {
        const TrackedFace* selected = targetManager_.getSelectedFace();
        if (selected && selected->currentStatus == TargetStatus::lost) {
            targetManager_.selectNext();
            syncSelectedStatus();
        }
    }
}

void ContourTracker::updateTrc(const cv::Mat& frame) {
    if (!initialized || frame.empty()) return;

    if (targetManager_.getMode() != TrackerMode::trc) {
        targetManager_.setMode(TrackerMode::trc, targetManager_.getSelectedId());
        firstTrcFrame = true;
    }

    if (!targetManager_.hasTrackedFace()) return;

    TrackedFace& face = targetManager_.getTrackedFace();

    auto currentTime = std::chrono::steady_clock::now();
    float deltaTime = 0.033f;
    if (hasPreviousTime) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - previousTime);
        deltaTime = elapsed.count() / 1000000.0f;
        deltaTime = std::clamp(deltaTime, 0.001f, 0.1f);
    }
    hasPreviousTime = true;
    previousTime = currentTime;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Rect> detectedRects;
    std::vector<std::array<double, 7>> detectedHu;

    if (firstTrcFrame) {
        // Глобальный поиск
        contours = findSkinContours(frame);
        detectedRects = contoursToRects(contours);
        for (auto& c : contours)
            detectedHu.push_back(computeHuMoments(c));
    }
    else {
        cv::Point2f searchCenter = (face.currentStatus == TargetStatus::lost)
            ? face.center
            : getPredictedPosition(face);
        float speed = cv::norm(face.velocity);
        int margin = baseSearchMargin + static_cast<int>(speed * velocityScale);
        if (face.lostFrames > 0) margin = static_cast<int>(margin * 1.5f);
        margin = std::min(margin, maxSearchMargin);

        cv::Rect searchArea(
            std::max(0, (int)searchCenter.x - margin),
            std::max(0, (int)searchCenter.y - margin),
            margin * 2, margin * 2
        );
        if (searchArea.x + searchArea.width > frame.cols)
            searchArea.width = frame.cols - searchArea.x;
        if (searchArea.y + searchArea.height > frame.rows)
            searchArea.height = frame.rows - searchArea.y;

        if (searchArea.area() > 0) {
            cv::Mat roi = frame(searchArea);
            std::vector<std::vector<cv::Point>> roiContours = findSkinContours(roi);
            for (auto& c : roiContours) {
                for (auto& pt : c) { pt.x += searchArea.x; pt.y += searchArea.y; }
                detectedRects.push_back(cv::boundingRect(c));
                detectedHu.push_back(computeHuMoments(c));
            }
        }
    }

    // Поиск наилучшего совпадения по моментам
    int bestIdx = -1;
    double bestHuDist = huComparisonThreshold * 5; // порог
    auto it = huMomentsMap.find(face.id);
    if (it != huMomentsMap.end()) {
        const auto& refHu = it->second;
        for (size_t j = 0; j < detectedRects.size(); ++j) {
            double huDist = compareHuMoments(refHu, detectedHu[j]);
            if (huDist < bestHuDist) {
                bestHuDist = huDist;
                bestIdx = static_cast<int>(j);
            }
        }
    }

    if (bestIdx != -1) {
        face.currentStatus = TargetStatus::lock;
        face.lostFrames = 0;

        const cv::Rect& bestRect = detectedRects[bestIdx];
        face.boundingBox.x = (1.0f - smoothAlpha) * face.boundingBox.x + smoothAlpha * bestRect.x;
        face.boundingBox.y = (1.0f - smoothAlpha) * face.boundingBox.y + smoothAlpha * bestRect.y;
        face.boundingBox.width = (1.0f - smoothAlpha) * face.boundingBox.width + smoothAlpha * bestRect.width;
        face.boundingBox.height = (1.0f - smoothAlpha) * face.boundingBox.height + smoothAlpha * bestRect.height;

        cv::Point2f newCenter(face.boundingBox.x + face.boundingBox.width / 2.0f,
            face.boundingBox.y + face.boundingBox.height / 2.0f);
        updateFaceVelocity(face, newCenter, deltaTime);
        face.positionHistory.push_back(newCenter);
        if (face.positionHistory.size() > 10) face.positionHistory.pop_front();
        face.center = newCenter;
        face.age++;
        face.predictedCenter = getPredictedPosition(face);

        // Обновляем эталонные моменты
        for (int k = 0; k < 7; ++k)
            it->second[k] = (1.0 - smoothAlpha) * it->second[k] + smoothAlpha * detectedHu[bestIdx][k];

        if (firstTrcFrame) firstTrcFrame = false;
    }
    else {
        face.lostFrames++;
        if (face.lostFrames > maxLostFrames) {
            face.currentStatus = TargetStatus::lost;
            if (!face.lostTimeSet) {
                face.lostTime = currentTime;
                face.lostTimeSet = true;
            }
        }
        else {
            face.predictedCenter = getPredictedPosition(face);
        }
        // В первом кадре не сбрасываем флаг, остаёмся в глобальном поиске
    }

    targetManager_.updateTrackedFace(face);
}

void ContourTracker::syncSelectedStatus() {
    int selectedId = targetManager_.getSelectedId();
    if (selectedId == -1) return;
    const auto& faces = targetManager_.getFaces();
    auto it = std::find_if(faces.begin(), faces.end(),
        [selectedId](const TrackedFace& f) { return f.id == selectedId; });
    if (it != faces.end() && it->currentStatus != TargetStatus::softlock && it->currentStatus != TargetStatus::lost) {
        TrackedFace updated = *it;
        updated.currentStatus = TargetStatus::softlock;
        targetManager_.updateFace(selectedId, updated);
    }
}

void ContourTracker::drawTrackingInfo(cv::Mat& frame) const {
    if (!initialized) return;

    if (trackerMode_ == TrackerMode::src) {
        renderer.draw(frame, targetManager_.getFaces(), initialized);
    }
    else {
        int selectedId = targetManager_.getSelectedId();
        if (selectedId == -1) return;
        const auto& faces = targetManager_.getFaces();
        auto it = std::find_if(faces.begin(), faces.end(),
            [selectedId](const TrackedFace& f) { return f.id == selectedId; });
        if (it == faces.end()) return;
        std::vector<TrackedFace> singleFace = { *it };
        renderer.draw(frame, singleFace, initialized);
    }
}

void ContourTracker::reset() {
    targetManager_.clear();
    huMomentsMap.clear();
    nextObjectId = 1;
    initialized = false;
    hasPreviousTime = false;
}

bool ContourTracker::isInitialized() const {
    return initialized;
}

// ---------- Вспомогательные методы ----------

cv::Mat ContourTracker::createSkinMask(const cv::Mat& frame) const {
    cv::Mat hsv, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv,
        cv::Scalar(hueLow, satLow, valLow),
        cv::Scalar(hueHigh, satHigh, valHigh),
        mask);

    // Морфологическая очистка
    cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 1);
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);
    return mask;
}

std::vector<std::vector<cv::Point>> ContourTracker::findSkinContours(const cv::Mat& frame) const {
    cv::Mat mask = createSkinMask(frame);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Фильтрация по площади и соотношению сторон
    contours.erase(std::remove_if(contours.begin(), contours.end(),
        [&](const std::vector<cv::Point>& c) {
            double area = cv::contourArea(c);
            if (area < minContourArea || area > maxContourArea) return true;
            cv::Rect r = cv::boundingRect(c);
            float aspect = (float)r.width / r.height;
            return (aspect < minAspectRatio || aspect > maxAspectRatio);
        }), contours.end());

    return contours;
}

std::vector<cv::Rect> ContourTracker::contoursToRects(const std::vector<std::vector<cv::Point>>& contours) const {
    std::vector<cv::Rect> rects;
    rects.reserve(contours.size());
    for (const auto& c : contours) {
        rects.push_back(cv::boundingRect(c));
    }
    return rects;
}

std::array<double, 7> ContourTracker::computeHuMoments(const std::vector<cv::Point>& contour) const {
    cv::Moments m = cv::moments(contour);
    double hu[7];
    cv::HuMoments(m, hu);
    std::array<double, 7> result;
    for (int i = 0; i < 7; ++i) result[i] = hu[i];
    return result;
}

double ContourTracker::compareHuMoments(const std::array<double, 7>& hu1, const std::array<double, 7>& hu2) const {
    double dist = 0.0;
    for (int i = 0; i < 7; ++i) {
        double a = std::abs(hu1[i]);
        double b = std::abs(hu2[i]);
        a = (a == 0) ? 1e-10 : a;
        b = (b == 0) ? 1e-10 : b;
        dist += std::abs(std::log(a) - std::log(b));
    }
    return dist;
}

void ContourTracker::updateFaceVelocity(TrackedFace& face, const cv::Point2f& newPosition, float deltaTime) {
    if (!face.hasPreviousPosition) {
        face.previousPosition = newPosition;
        face.previousTime = std::chrono::steady_clock::now();
        face.hasPreviousPosition = true;
        return;
    }
    auto currentTime = std::chrono::steady_clock::now();
    float actualDelta = std::chrono::duration<float>(currentTime - face.previousTime).count();
    if (actualDelta < 0.001f) return;
    cv::Point2f displacement = newPosition - face.previousPosition;
    cv::Point2f newVel = displacement / actualDelta;
    float alpha = 0.3f * std::min(actualDelta * 30.0f, 1.0f);
    face.velocity = (1.0f - alpha) * face.velocity + alpha * newVel;
    face.previousPosition = newPosition;
    face.previousTime = currentTime;
}

cv::Point2f ContourTracker::getPredictedPosition(const TrackedFace& face) const {
    return face.center + face.velocity * predictionTime;
}

float ContourTracker::computeIOU(const cv::Rect& a, const cv::Rect& b) const {
    int x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width), y2 = std::min(a.y + a.height, b.y + b.height);
    int inter = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - inter;
    return unionArea > 0 ? float(inter) / unionArea : 0.0f;
}

std::vector<cv::Rect> ContourTracker::nonMaximumSuppression(const std::vector<cv::Rect>& rects, float iouThreshold) const {
    std::vector<cv::Rect> result;
    std::vector<int> indices(rects.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&rects](int i, int j) { return rects[i].area() > rects[j].area(); });

    while (!indices.empty()) {
        int best = indices[0];
        result.push_back(rects[best]);
        std::vector<int> remaining;
        for (size_t i = 1; i < indices.size(); ++i) {
            int idx = indices[i];
            if (computeIOU(rects[best], rects[idx]) < iouThreshold)
                remaining.push_back(idx);
        }
        indices = remaining;
    }
    return result;
}
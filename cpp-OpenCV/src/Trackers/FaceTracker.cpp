#include "FaceTracker.h"
#include <iostream>
#include <algorithm>
#include <numeric>

FaceTracker::FaceTracker()
    : nextFaceId(1), initialized(false), predictionTime(1.0f), hasPreviousTime(false) {

    scaleFactor = 1.12;
    minNeighbors = 7;
    minSize = cv::Size(25, 25);
    maxSize = cv::Size(400, 400);

    renderer.setFaceColor(cv::Scalar(0, 255, 0));
    renderer.setPredictionColor(cv::Scalar(0, 255, 0));
    renderer.setLostColor(cv::Scalar(0, 0, 255));

    try {
        faceCascade.load(FACE_CASCADE_FRONTAL);
    }
    catch (...) {
        std::cerr << "Не удалось загрузить каскады Хаара" << std::endl;
    }
}

bool FaceTracker::initialize(const cv::Mat& frame) {
    if (frame.empty()) {
        std::cerr << "FaceTracker::initialize: frame is empty" << std::endl;
        return false;
    }

    // Детектируем лица
    std::vector<cv::Rect> faces = detectFaces(frame);
    if (faces.empty()) {
        std::cout << "FaceTracker::initialize: no faces detected" << std::endl;
        return false;
    }

    auto currentTime = std::chrono::steady_clock::now();

    // Добавляем все обнаруженные лица
    for (const auto& faceRect : faces) {
        TrackedFace face;
        face.id = nextFaceId++;
        face.boundingBox = faceRect;
        face.center = cv::Point2f(faceRect.x + faceRect.width / 2.0f,
            faceRect.y + faceRect.height / 2.0f);
        face.age = 1;
        face.currentStatus = TargetStatus::find;
        face.positionHistory.push_back(face.center);
        face.previousPosition = face.center;
        face.velocity = cv::Point2f(0, 0);
        face.predictedCenter = face.center;
        face.hasPreviousPosition = false;
        face.lostFrames = 0;
        face.lostTimeSet = false;
        targetManager_.addFace(face);
    }

    // Выбираем первое лицо и устанавливаем ему статус softlock
    if (!targetManager_.getFaces().empty()) {
        targetManager_.selectNext();   // выбирает первое, если selectedId_ == -1
        int selectedId = targetManager_.getSelectedId();
        if (selectedId != -1) {
            TrackedFace* facePtr = targetManager_.getFaceById(selectedId);
            if (facePtr) {
                facePtr->currentStatus = TargetStatus::softlock;
                targetManager_.updateFace(selectedId, *facePtr);
                std::cout << "FaceTracker: selected face ID " << selectedId << " set to softlock" << std::endl;
            }
        }
    }

    initialized = true;
    previousTime = currentTime;
    hasPreviousTime = true;

    std::cout << "FaceTracker initialized with " << targetManager_.getFaces().size() << " faces" << std::endl;
    return true;
}

bool FaceTracker::update(const cv::Mat& frame) {
    if (!initialized) {
        std::cerr << "FaceTracker not initialized!" << std::endl;
        return false;
    }
    //updateSrc(frame);
    switch (trackerMode_) {
    case TrackerMode::src:
        updateSrc(frame);
        break;

    case TrackerMode::trc:
        updateTrc(frame);
        break;
    }
    
}

bool FaceTracker::updateSrc(const cv::Mat& frame) {
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

    // Детекция лиц
    std::vector<cv::Rect> detectedRects = detectFaces(frame);
    detectedRects = nonMaximumSuppression(detectedRects, 0.3f);

    // Получаем текущий список лиц из TargetManager
    const auto& currentFaces = targetManager_.getFaces();
    int selectedId = targetManager_.getSelectedId();

    std::vector<bool> used(detectedRects.size(), false);

    // 1. Обновление существующих лиц
    for (const auto& face : currentFaces) {
        if (face.currentStatus == TargetStatus::lost) continue;

        float bestIoU = 0.3f;
        int bestIdx = -1;
        for (size_t i = 0; i < detectedRects.size(); ++i) {
            if (used[i]) continue;
            float iou = computeIOU(face.boundingBox, detectedRects[i]);
            if (iou > bestIoU) {
                bestIoU = iou;
                bestIdx = static_cast<int>(i);
            }
        }

        TrackedFace updatedFace = face;

        if (bestIdx != -1) {
            used[bestIdx] = true;
            updateFacePosition(updatedFace, detectedRects[bestIdx], deltaTime);
            updatedFace.lostFrames = 0;
            if (updatedFace.currentStatus == TargetStatus::lost) {
                updatedFace.currentStatus = TargetStatus::find;
            }
        }
        else {
            updatedFace.lostFrames++;
            if (updatedFace.lostFrames > maxLostFrames) {
                updatedFace.currentStatus = TargetStatus::lost;
                updatedFace.lostTime = currentTime;
                updatedFace.lostTimeSet = true;
            }
            else {
                // Прогнозируем позицию по скорости
                updatedFace.predictedCenter = getPredictedPosition(updatedFace);
            }
        }

        // Если это выбранное лицо и оно не потеряно, восстанавливаем статус softlock
        if (face.id == selectedId && updatedFace.currentStatus != TargetStatus::lost) {
            updatedFace.currentStatus = TargetStatus::softlock;
        }

        targetManager_.updateFace(face.id, updatedFace);
    }

    // 2. Добавление новых лиц
    for (size_t i = 0; i < detectedRects.size(); ++i) {
        if (!used[i]) {
            TrackedFace newFace;
            newFace.id = nextFaceId++;
            newFace.boundingBox = detectedRects[i];
            newFace.center = cv::Point2f(
                detectedRects[i].x + detectedRects[i].width / 2.0f,
                detectedRects[i].y + detectedRects[i].height / 2.0f
            );
            newFace.previousPosition = newFace.center;
            newFace.velocity = cv::Point2f(0, 0);
            newFace.predictedCenter = newFace.center;
            newFace.age = 0;
            newFace.lostFrames = 0;
            newFace.currentStatus = TargetStatus::find;
            newFace.hasPreviousPosition = false;
            newFace.positionHistory.push_back(newFace.center);
            targetManager_.addFace(newFace);
        }
    }

    // 3. Удаление старых потерянных лиц
    targetManager_.removeLostFaces(1.5f);

    // 4. Синхронизация статуса выбранного лица
    syncSelectedStatus();

    return !targetManager_.getFaces().empty();
}

bool FaceTracker::updateTrc(const cv::Mat& frame) {
    if (!initialized || frame.empty()) return false;

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

    // Получаем текущий список лиц и выбранный ID
    const auto& allFaces = targetManager_.getFaces();   // объявляем один раз
    int selectedId = targetManager_.getSelectedId();

    // Если ничего не выбрано – выбираем первое лицо и ставим статус lock
    if (selectedId == -1) {
        if (!allFaces.empty()) {
            targetManager_.selectNext();
            selectedId = targetManager_.getSelectedId();
            TrackedFace* facePtr = targetManager_.getFaceById(selectedId);
            if (facePtr) {
                facePtr->currentStatus = TargetStatus::lock;
                targetManager_.updateFace(selectedId, *facePtr);
            }
        }
        else {
            return false; // нет лиц для слежения
        }
    }

    // Находим выбранное лицо по ID (используем allFaces, чтобы не делать лишний запрос)
    auto it = std::find_if(allFaces.begin(), allFaces.end(),
        [selectedId](const TrackedFace& f) { return f.id == selectedId; });
    if (it == allFaces.end()) return false; // лицо не найдено (возможно удалено)

    TrackedFace face = *it; // копия для обновления

    // Детекция лиц на всём кадре
    std::vector<cv::Rect> detectedRects = detectFaces(frame);
    detectedRects = nonMaximumSuppression(detectedRects, 0.3f);

    // Поиск лучшего соответствия среди детекций, близких к предсказанной позиции
    cv::Point2f predictedCenter = getPredictedPosition(face);
    const float MAX_DIST = 150.0f;

    std::vector<int> candidates;
    for (size_t i = 0; i < detectedRects.size(); ++i) {
        cv::Point2f detCenter(detectedRects[i].x + detectedRects[i].width / 2.0f,
            detectedRects[i].y + detectedRects[i].height / 2.0f);
        float dist = calculateDistance(predictedCenter, detCenter);
        if (dist < MAX_DIST) {
            candidates.push_back(static_cast<int>(i));
        }
    }

    int bestIdx = -1;
    float bestIOU = 0.3f;
    for (int idx : candidates) {
        float iou = computeIOU(face.boundingBox, detectedRects[idx]);
        if (iou > bestIOU) {
            bestIOU = iou;
            bestIdx = idx;
        }
    }

    if (bestIdx != -1) {
        // Лицо найдено – обновляем позицию
        updateFacePosition(face, detectedRects[bestIdx], deltaTime);
        face.lostFrames = 0;
        face.currentStatus = TargetStatus::lock;
    }
    else {
        // Лицо не найдено
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
            if (face.currentStatus != TargetStatus::lost) {
                face.currentStatus = TargetStatus::lock; // сохраняем lock при временной потере
            }
        }
    }

    // Гарантируем, что если не потеряно, то статус lock
    if (face.currentStatus != TargetStatus::lost) {
        face.currentStatus = TargetStatus::lock;
    }

    // Сохраняем обновлённое лицо
    targetManager_.updateFace(selectedId, face);
    return true;
}


void FaceTracker::updateFaceTracking(const std::vector<cv::Rect>& detectedFaces, float deltaTime) {
    // Получаем текущий список лиц из TargetManager (копия, чтобы безопасно модифицировать)
    auto currentFaces = targetManager_.getFaces(); // копия
    std::vector<TrackedFace> updatedFaces;
    updatedFaces.reserve(currentFaces.size() + detectedFaces.size());

    // Сначала помечаем все существующие лица как потерянные (временный флаг)
    for (auto& face : currentFaces) {
        face.currentStatus = TargetStatus::lost;
    }

    // Сопоставляем детекции с существующими лицами
    std::vector<bool> used(detectedFaces.size(), false);

    for (size_t i = 0; i < currentFaces.size(); ++i) {
        auto& face = currentFaces[i];
        cv::Point2f faceCenter = face.center;

        // Ищем ближайшую детекцию к центру лица
        int bestIdx = -1;
        float minDist = 200.0f; // порог расстояния
        for (size_t j = 0; j < detectedFaces.size(); ++j) {
            if (used[j]) continue;
            cv::Point2f detCenter(detectedFaces[j].x + detectedFaces[j].width / 2.0f,
                detectedFaces[j].y + detectedFaces[j].height / 2.0f);
            float dist = calculateDistance(faceCenter, detCenter);
            if (dist < minDist) {
                minDist = dist;
                bestIdx = static_cast<int>(j);
            }
        }

        if (bestIdx != -1) {
            used[bestIdx] = true;
            // Обновляем лицо
            updateFacePosition(face, detectedFaces[bestIdx], deltaTime);
            face.currentStatus = TargetStatus::find;
            face.age++;
            face.lostFrames = 0;
        }
        else {
            // Лицо не найдено – увеличиваем счётчик потерянных
            face.lostFrames++;
            if (face.lostFrames > maxLostFrames) {
                face.currentStatus = TargetStatus::lost;
                face.lostTime = std::chrono::steady_clock::now();
                face.lostTimeSet = true;
            }
            else {
                // Прогнозируем позицию по скорости
                face.predictedCenter = getPredictedPosition(face);
            }
        }
        updatedFaces.push_back(face);
    }

    // Добавляем новые лица из неиспользованных детекций
    for (size_t j = 0; j < detectedFaces.size(); ++j) {
        if (!used[j]) {
            TrackedFace newFace;
            newFace.id = nextFaceId++;
            newFace.boundingBox = detectedFaces[j];
            newFace.center = cv::Point2f(detectedFaces[j].x + detectedFaces[j].width / 2.0f,
                detectedFaces[j].y + detectedFaces[j].height / 2.0f);
            newFace.previousPosition = newFace.center;
            newFace.velocity = cv::Point2f(0, 0);
            newFace.predictedCenter = newFace.center;
            newFace.age = 1;
            newFace.lostFrames = 0;
            newFace.currentStatus = TargetStatus::find;
            newFace.hasPreviousPosition = false;
            newFace.positionHistory.push_back(newFace.center);
            updatedFaces.push_back(newFace);
        }
    }

    // Обновляем TargetManager: очищаем и добавляем все заново
    // (более эффективно было бы обновлять по одному, но для простоты так)
    targetManager_.clear();
    for (const auto& face : updatedFaces) {
        targetManager_.addFace(face);
    }

    // Синхронизируем статус выбранного лица
    syncSelectedStatus();
}

void FaceTracker::syncSelectedStatus() {
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

void FaceTracker::selectNextTrg() {
    if (trackerMode_ == TrackerMode::trc) {
        return;
    }
    MasterTracker::selectNextTrg();   // базовый вызов (переключает ID в targetManager_)
    syncSelectedStatus();              // устанавливает softlock для новой цели
}

void FaceTracker::selectPrevTrg() {
    if (trackerMode_ == TrackerMode::trc) {
        return;
    }
    MasterTracker::selectPrevTrg();
    syncSelectedStatus();
}

void FaceTracker::drawTrackingInfo(cv::Mat& frame) const {
    if (!initialized) return;

    if (trackerMode_ == TrackerMode::src) {
        // Режим SRC: рисуем все лица
        renderer.draw(frame, targetManager_.getFaces(), initialized);
    }
    else { // TrackerMode::trc
        int selectedId = targetManager_.getSelectedId();
        if (selectedId == -1) return;

        const auto& faces = targetManager_.getFaces();
        auto it = std::find_if(faces.begin(), faces.end(),
            [selectedId](const TrackedFace& f) { return f.id == selectedId; });
        if (it == faces.end()) return;

        // Передаём вектор только с одним лицом
        std::vector<TrackedFace> singleFace = { *it };
        renderer.draw(frame, singleFace, initialized);
    }
}

void FaceTracker::reset() {
    targetManager_.clear();
    nextFaceId = 1;
    initialized = false;
    hasPreviousTime = false;
}

bool FaceTracker::isInitialized() const {
    return initialized;
}

std::vector<TrackedFace> FaceTracker::getTrackedFaces() const {
    return targetManager_.getFaces();
}

cv::Point2f FaceTracker::getLargestFaceCenter() const {
    auto faces = targetManager_.getFaces();
    if (faces.empty()) return cv::Point2f(0, 0);
    auto largest = std::max_element(faces.begin(), faces.end(),
        [](const TrackedFace& a, const TrackedFace& b) {
            return a.boundingBox.area() < b.boundingBox.area();
        });
    return largest->center;
}

TrackedFace FaceTracker::getLargestFace() const {
    auto faces = targetManager_.getFaces();
    if (faces.empty()) return TrackedFace();
    auto largest = std::max_element(faces.begin(), faces.end(),
        [](const TrackedFace& a, const TrackedFace& b) {
            return a.boundingBox.area() < b.boundingBox.area();
        });
    return *largest;
}

// ---------- Вспомогательные методы (без изменений) ----------

std::vector<cv::Rect> FaceTracker::detectFaces(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;
    if (frame.empty()) return faces;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    faceCascade.detectMultiScale(gray, faces, scaleFactor, minNeighbors, 0, minSize, maxSize);

    faces.erase(std::remove_if(faces.begin(), faces.end(),
        [](const cv::Rect& r) {
            float aspect = (float)r.width / r.height;
            return aspect < 0.8f || aspect > 1.4f;
        }), faces.end());

    faces = nonMaximumSuppression(faces, 0.4f);
    return faces;
}

std::vector<cv::Rect> FaceTracker::nonMaximumSuppression(const std::vector<cv::Rect>& faces, float iouThreshold) const {
    std::vector<cv::Rect> result;
    std::vector<int> indices(faces.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
        [&faces](int i, int j) { return faces[i].area() > faces[j].area(); });

    while (!indices.empty()) {
        int best = indices[0];
        result.push_back(faces[best]);

        std::vector<int> remaining;
        for (size_t i = 1; i < indices.size(); ++i) {
            int idx = indices[i];
            if (computeIOU(faces[best], faces[idx]) < iouThreshold)
                remaining.push_back(idx);
        }
        indices = remaining;
    }
    return result;
}

void FaceTracker::updateFacePosition(TrackedFace& face, const cv::Rect& newRect, float deltaTime) {
    face.boundingBox.x = 0.7f * face.boundingBox.x + 0.3f * newRect.x;
    face.boundingBox.y = 0.7f * face.boundingBox.y + 0.3f * newRect.y;
    face.boundingBox.width = 0.7f * face.boundingBox.width + 0.3f * newRect.width;
    face.boundingBox.height = 0.7f * face.boundingBox.height + 0.3f * newRect.height;
    cv::Point2f newCenter(face.boundingBox.x + face.boundingBox.width / 2.0f,
        face.boundingBox.y + face.boundingBox.height / 2.0f);
    updateVelocity(face, newCenter, deltaTime);
    updatePositionHistory(face, newCenter);
    face.center = newCenter;
    face.predictedCenter = getPredictedPosition(face);
}

void FaceTracker::updateVelocity(TrackedFace& face, const cv::Point2f& newPosition, float deltaTime) {
    if (face.positionHistory.size() >= 2) {
        cv::Point2f lastPosition = face.positionHistory.back();
        cv::Point2f displacement = newPosition - lastPosition;
        if (deltaTime > 0) {
            cv::Point2f newVelocity = displacement / deltaTime;
            float alpha = 0.3f;
            face.velocity = (1.0f - alpha) * face.velocity + alpha * newVelocity;
        }
    }
}

void FaceTracker::updateFaceVelocity(TrackedFace& face, const cv::Point2f& newPosition, float deltaTime) {
    if (!face.hasPreviousPosition) {
        face.previousPosition = newPosition;
        face.previousTime = std::chrono::steady_clock::now();
        face.hasPreviousPosition = true;
        return;
    }
    auto currentTime = std::chrono::steady_clock::now();
    float actualDeltaTime = std::chrono::duration<float>(currentTime - face.previousTime).count();
    if (actualDeltaTime < 0.001f) return;
    cv::Point2f displacement = newPosition - face.previousPosition;
    cv::Point2f newVelocity = displacement / actualDeltaTime;
    float alpha = 0.3f * std::min(actualDeltaTime * 30.0f, 1.0f);
    face.velocity = (1.0f - alpha) * face.velocity + alpha * newVelocity;
    face.previousPosition = newPosition;
    face.previousTime = currentTime;
}

void FaceTracker::updatePositionHistory(TrackedFace& face, const cv::Point2f& newPosition) {
    face.positionHistory.push_back(newPosition);
    if (face.positionHistory.size() > maxHistorySize)
        face.positionHistory.pop_front();
}

cv::Point2f FaceTracker::getSmoothedPosition(const TrackedFace& face) const {
    if (face.positionHistory.empty()) return face.center;
    cv::Point2f sum(0, 0);
    for (const auto& pos : face.positionHistory) sum += pos;
    return cv::Point2f(sum.x / face.positionHistory.size(), sum.y / face.positionHistory.size());
}

cv::Point2f FaceTracker::getPredictedPosition(const TrackedFace& face) const {
    return face.center + face.velocity * predictionTime;
}

cv::Rect FaceTracker::getLargestFaceRect(const std::vector<cv::Rect>& faces) {
    if (faces.empty()) return cv::Rect();
    cv::Rect largestFace = faces[0];
    int maxArea = largestFace.area();
    for (size_t i = 1; i < faces.size(); ++i) {
        int area = faces[i].area();
        if (area > maxArea) {
            maxArea = area;
            largestFace = faces[i];
        }
    }
    return largestFace;
}

float FaceTracker::calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) const {
    float dx = p1.x - p2.x, dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

int FaceTracker::findClosestFace(const cv::Point2f& center, const std::vector<TrackedFace>& faces) const {
    if (faces.empty()) return -1;
    int closestIndex = -1;
    float minDistance = 1000.0f;
    for (size_t i = 0; i < faces.size(); ++i) {
        float distance = calculateDistance(center, faces[i].center);
        if (distance < minDistance) {
            minDistance = distance;
            closestIndex = i;
        }
    }
    return closestIndex;
}

float FaceTracker::computeIOU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width), y2 = std::min(a.y + a.height, b.y + b.height);
    int inter = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - inter;
    return unionArea > 0 ? float(inter) / unionArea : 0.0f;
}
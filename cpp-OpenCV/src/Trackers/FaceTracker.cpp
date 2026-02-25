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
    if (!initialized || frame.empty()) return false;

    // Убеждаемся, что режим SRC
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

    // Получаем текущий список лиц (копия для безопасной модификации)
    std::vector<TrackedFace> currentFaces = targetManager_.getFaces();
    std::vector<bool> faceMatched(currentFaces.size(), false);

    // --- 1. Локальный трекинг для каждого лица (выполняется каждый кадр) ---
    for (size_t i = 0; i < currentFaces.size(); ++i) {
        TrackedFace& face = currentFaces[i];
        if (face.currentStatus == TargetStatus::lost) continue; // потерянных не трекаем локально

        // Определяем центр поиска (предсказанная позиция)
        cv::Point2f searchCenter = getPredictedPosition(face);
        int margin = baseSearchMargin; // можно использовать меньший margin, чем в TRC, например 50
        cv::Rect searchArea(
            std::max(0, (int)searchCenter.x - margin),
            std::max(0, (int)searchCenter.y - margin),
            margin * 2,
            margin * 2
        );
        // Корректировка границ
        if (searchArea.x + searchArea.width > frame.cols)
            searchArea.width = frame.cols - searchArea.x;
        if (searchArea.y + searchArea.height > frame.rows)
            searchArea.height = frame.rows - searchArea.y;

        std::vector<cv::Rect> detected;
        if (searchArea.area() > 0) {
            cv::Mat roi = frame(searchArea);
            detected = detectFaces(roi);
            for (auto& r : detected) {
                r.x += searchArea.x;
                r.y += searchArea.y;
            }
        }

        if (!detected.empty()) {
            // Выбираем ближайшее к предсказанной позиции
            float bestDist = std::numeric_limits<float>::max();
            cv::Rect bestRect;
            for (const auto& r : detected) {
                cv::Point2f c(r.x + r.width / 2.0f, r.y + r.height / 2.0f);
                float dist = cv::norm(c - searchCenter);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestRect = r;
                }
            }
            if (bestRect.area() > 0) {
                // Обновляем лицо
                face.lostFrames = 0;
                face.currentStatus = TargetStatus::find; // временно, потом синхронизируем
                // Экспоненциальное сглаживание
                const float alpha = smoothAlpha;
                face.boundingBox.x = (1.0f - alpha) * face.boundingBox.x + alpha * bestRect.x;
                face.boundingBox.y = (1.0f - alpha) * face.boundingBox.y + alpha * bestRect.y;
                face.boundingBox.width = (1.0f - alpha) * face.boundingBox.width + alpha * bestRect.width;
                face.boundingBox.height = (1.0f - alpha) * face.boundingBox.height + alpha * bestRect.height;
                cv::Point2f newCenter(
                    face.boundingBox.x + face.boundingBox.width / 2.0f,
                    face.boundingBox.y + face.boundingBox.height / 2.0f
                );
                updateFaceVelocity(face, newCenter, deltaTime);
                face.positionHistory.push_back(newCenter);
                if (face.positionHistory.size() > maxHistorySize)
                    face.positionHistory.pop_front();
                face.center = newCenter;
                face.age++;
                face.predictedCenter = getPredictedPosition(face);
                faceMatched[i] = true; // помечаем, что лицо обновлено локально
            }
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
        }
    }

    // --- 2. Периодическая глобальная детекция ---
    framesSinceLastGlobal++;
    bool doGlobal = (framesSinceLastGlobal >= globalDetectionPeriod);

    if (doGlobal) {
        framesSinceLastGlobal = 0;

        // Детекция на всём кадре
        std::vector<cv::Rect> globalDetections = detectFaces(frame);
        globalDetections = nonMaximumSuppression(globalDetections, 0.3f);
        std::vector<bool> used(globalDetections.size(), false);

        // Сопоставление с существующими лицами (по IOU или расстоянию)
        for (size_t i = 0; i < currentFaces.size(); ++i) {
            TrackedFace& face = currentFaces[i];
            if (face.currentStatus == TargetStatus::lost) continue; // потерянных не обновляем глобально

            float bestIOU = 0.3f;
            int bestIdx = -1;
            for (size_t j = 0; j < globalDetections.size(); ++j) {
                if (used[j]) continue;
                float iou = computeIOU(face.boundingBox, globalDetections[j]);
                if (iou > bestIOU) {
                    bestIOU = iou;
                    bestIdx = static_cast<int>(j);
                }
            }
            if (bestIdx != -1) {
                used[bestIdx] = true;
                // Обновляем позицию (можно применить сглаживание или просто заменить)
                const float alpha = smoothAlpha;
                face.boundingBox.x = (1.0f - alpha) * face.boundingBox.x + alpha * globalDetections[bestIdx].x;
                face.boundingBox.y = (1.0f - alpha) * face.boundingBox.y + alpha * globalDetections[bestIdx].y;
                face.boundingBox.width = (1.0f - alpha) * face.boundingBox.width + alpha * globalDetections[bestIdx].width;
                face.boundingBox.height = (1.0f - alpha) * face.boundingBox.height + alpha * globalDetections[bestIdx].height;
                cv::Point2f newCenter(
                    face.boundingBox.x + face.boundingBox.width / 2.0f,
                    face.boundingBox.y + face.boundingBox.height / 2.0f
                );
                // Обновляем скорость и историю (если не делали это в локальном трекинге)
                if (!faceMatched[i]) {
                    updateFaceVelocity(face, newCenter, deltaTime);
                    face.positionHistory.push_back(newCenter);
                    if (face.positionHistory.size() > maxHistorySize)
                        face.positionHistory.pop_front();
                    face.center = newCenter;
                    face.predictedCenter = getPredictedPosition(face);
                }
                face.lostFrames = 0;
                face.currentStatus = TargetStatus::find;
                face.age++;
            }
        }

        // Добавление новых лиц
        for (size_t j = 0; j < globalDetections.size(); ++j) {
            if (!used[j]) {
                TrackedFace newFace;
                newFace.id = nextFaceId++;
                newFace.boundingBox = globalDetections[j];
                newFace.center = cv::Point2f(
                    globalDetections[j].x + globalDetections[j].width / 2.0f,
                    globalDetections[j].y + globalDetections[j].height / 2.0f
                );
                newFace.previousPosition = newFace.center;
                newFace.velocity = cv::Point2f(0, 0);
                newFace.predictedCenter = newFace.center;
                newFace.age = 1;
                newFace.lostFrames = 0;
                newFace.currentStatus = TargetStatus::find;
                newFace.hasPreviousPosition = false;
                newFace.positionHistory.push_back(newFace.center);
                currentFaces.push_back(newFace);
            }
        }
    }

    // --- 3. Очистка старых потерянных лиц ---
    // Используем существующий метод removeLostFrames, но он работает с вектором внутри TargetManager.
    // Проще обновить TargetManager целиком.
    targetManager_.setFaces(currentFaces);
    targetManager_.removeLostFaces(1.5f);
    syncSelectedStatus();

    if (targetManager_.getMode() == TrackerMode::src) {
        const TrackedFace* selected = targetManager_.getSelectedFace();
        if (selected && selected->currentStatus == TargetStatus::lost) {
            // Текущее выбранное лицо потеряно - переключаем на следующее
            targetManager_.selectNext();
            // Обновляем статус нового выбранного лица на softlock
            syncSelectedStatus(); // повторно вызываем, чтобы установить softlock
        }
    }

    return !targetManager_.getFaces().empty();
}

bool FaceTracker::updateTrc(const cv::Mat& frame) {
    if (!initialized || frame.empty()) return false;

    // Убеждаемся, что режим TRC
    if (targetManager_.getMode() != TrackerMode::trc) {
        targetManager_.setMode(TrackerMode::trc, targetManager_.getSelectedId());
        firstTrcFrame = true;   // взводим флаг при смене режима
    }

    if (!targetManager_.hasTrackedFace()) return false;

    TrackedFace& face = targetManager_.getTrackedFace();

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

    // Определяем область поиска
    std::vector<cv::Rect> detected;
    cv::Rect searchArea;

    if (firstTrcFrame) {
        // Первый кадр (или пока лицо не захвачено) – глобальный поиск по всему кадру
        detected = detectFaces(frame);
    }
    else {
        // Обычный локальный поиск с адаптивным размером
        cv::Point2f searchCenter = (face.currentStatus == TargetStatus::lost)
            ? face.center
            : getPredictedPosition(face);

        float speed = cv::norm(face.velocity);
        int margin = baseSearchMargin + static_cast<int>(speed * velocityScale);
        if (face.lostFrames > 0) {
            margin = static_cast<int>(margin * 1.5f); // расширение при потере
        }
        margin = std::min(margin, maxSearchMargin);

        searchArea = cv::Rect(
            std::max(0, (int)searchCenter.x - margin),
            std::max(0, (int)searchCenter.y - margin),
            margin * 2,
            margin * 2
        );
        // Корректировка границ
        if (searchArea.x + searchArea.width > frame.cols)
            searchArea.width = frame.cols - searchArea.x;
        if (searchArea.y + searchArea.height > frame.rows)
            searchArea.height = frame.rows - searchArea.y;

        if (searchArea.area() > 0) {
            cv::Mat roi = frame(searchArea);
            detected = detectFaces(roi);
            for (auto& r : detected) {
                r.x += searchArea.x;
                r.y += searchArea.y;
            }
        }
    }

    // Выбор наилучшего совпадения
    cv::Rect bestRect;
    if (!detected.empty()) {
        cv::Point2f targetCenter;
        if (firstTrcFrame) {
            // При глобальном поиске ищем ближайшее к последнему известному центру
            targetCenter = face.center;
        }
        else {
            targetCenter = getPredictedPosition(face);
        }

        float minDist = std::numeric_limits<float>::max();
        for (const auto& r : detected) {
            cv::Point2f c(r.x + r.width / 2.0f, r.y + r.height / 2.0f);
            float dist = cv::norm(c - targetCenter);
            if (dist < minDist) {
                minDist = dist;
                bestRect = r;
            }
        }
    }

    if (bestRect.area() > 0) {
        // Лицо найдено
        face.currentStatus = TargetStatus::lock;
        face.lostFrames = 0;

        // Экспоненциальное сглаживание
        const float alpha = smoothAlpha;
        face.boundingBox.x = (1.0f - alpha) * face.boundingBox.x + alpha * bestRect.x;
        face.boundingBox.y = (1.0f - alpha) * face.boundingBox.y + alpha * bestRect.y;
        face.boundingBox.width = (1.0f - alpha) * face.boundingBox.width + alpha * bestRect.width;
        face.boundingBox.height = (1.0f - alpha) * face.boundingBox.height + alpha * bestRect.height;

        cv::Point2f newCenter(
            face.boundingBox.x + face.boundingBox.width / 2.0f,
            face.boundingBox.y + face.boundingBox.height / 2.0f
        );

        updateFaceVelocity(face, newCenter, deltaTime);
        face.positionHistory.push_back(newCenter);
        if (face.positionHistory.size() > maxHistorySize)
            face.positionHistory.pop_front();
        face.center = newCenter;
        face.age++;
        face.predictedCenter = getPredictedPosition(face);

        // Если это был первый кадр (глобальный поиск), переходим в обычный режим
        if (firstTrcFrame) {
            firstTrcFrame = false;
        }
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
        }

        // Если это первый кадр и лицо не найдено, НЕ сбрасываем флаг
        // (остаёмся в режиме глобального поиска для следующего кадра)
    }

    targetManager_.updateTrackedFace(face);
    return face.currentStatus != TargetStatus::lost;
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
    if (trackerMode_ == TrackerMode::trc) return;
    MasterTracker::selectNextTrg(); // меняет selectedId_ в SRC
    syncSelectedStatus();
}

void FaceTracker::selectPrevTrg() {
    if (trackerMode_ == TrackerMode::trc) return;
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
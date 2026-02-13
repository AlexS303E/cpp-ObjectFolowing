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

    // Настройка цветов рендерера (соответствуют прежним)
    renderer.setFaceColor(cv::Scalar(0, 255, 0));        // зелёный
    renderer.setPredictionColor(cv::Scalar(0, 255, 0));  // зелёный
    renderer.setLostColor(cv::Scalar(0, 0, 255));        // красный

    try {
        faceCascade.load(FACE_CASCADE_FRONTAL);
    }
    catch (...) {
        std::cerr << "Не удалось загрузить каскады Хаара" << std::endl;
    }
}

bool FaceTracker::initialize(const cv::Mat& frame) {
    if (frame.empty()) return false;

    std::vector<cv::Rect> faces = detectFaces(frame);
    if (faces.empty()) return false;

    auto currentTime = std::chrono::steady_clock::now();
    for (const auto& faceRect : faces) {
        TrackedFace face;
        face.id = nextFaceId++;
        face.boundingBox = faceRect;
        face.center = cv::Point2f(faceRect.x + faceRect.width / 2.0f,
            faceRect.y + faceRect.height / 2.0f);
        face.age = 1;
        face.currentStatus = TargetStatus::find;
        updatePositionHistory(face, face.center);
        trackedFaces.push_back(face);
    }

    initialized = true;
    previousTime = currentTime;
    hasPreviousTime = true;

    std::cout << "FaceTracker инициализирован с " << trackedFaces.size() << " лицами" << std::endl;
    return true;
}

bool FaceTracker::update(const cv::Mat& frame) {
    if (frame.empty()) return false;

    std::vector<cv::Rect> detectedFaces = detectFaces(frame);

    // Сброс флагов
    for (auto& face : trackedFaces) {
        face.matched = false;
    }

    // -------- MATCH BY IOU --------
    for (const auto& det : detectedFaces) {
        float bestIOU = 0.0f;
        int bestIdx = -1;

        for (int i = 0; i < (int)trackedFaces.size(); i++) {
            auto& face = trackedFaces[i];
            if (face.IsLost()) continue;

            float iou = computeIOU(det, face.boundingBox);
            if (iou > bestIOU) {
                bestIOU = iou;
                bestIdx = i;
            }
        }

        // Нашли соответствие
        if (bestIdx >= 0 && bestIOU > 0.3f) {
            auto& face = trackedFaces[bestIdx];

            cv::Point2f newCenter(
                det.x + det.width * 0.5f,
                det.y + det.height * 0.5f
            );

            // Плавное сглаживание, БЕЗ скорости
            face.center = smoothAlpha * newCenter + (1.0f - smoothAlpha) * face.center;

            face.boundingBox = det;
            face.lostFrames = 0;
            face.matched = true;
            face.currentStatus = TargetStatus::find;

            updatePositionHistory(face, face.center);
        }
    }

    // -------- LOST HANDLING --------
    for (auto& face : trackedFaces) {
        if (!face.matched) {
            face.lostFrames++;
            face.currentStatus = TargetStatus::lost;
        }
    }

    // -------- ADD NEW --------
    for (const auto& det : detectedFaces) {
        bool matched = false;
        for (auto& face : trackedFaces) {
            if (computeIOU(det, face.boundingBox) > 0.3f) {
                matched = true;
                break;
            }
        }

        if (!matched) {
            TrackedFace f;
            f.id = nextFaceId++;
            f.boundingBox = det;
            f.center = {
                det.x + det.width * 0.5f,
                det.y + det.height * 0.5f
            };
            f.velocity = { 0,0 };
            f.predictedCenter = f.center;
            f.lostFrames = 0;
            f.matched = true;
            f.currentStatus = TargetStatus::find;

            trackedFaces.push_back(f);
        }
    }

    removeOldFaces();
    return true;
}


// ---------- Единственное изменение в drawTrackingInfo ----------
void FaceTracker::drawTrackingInfo(cv::Mat& frame) const {
    // Делегируем всю отрисовку рендереру
    renderer.draw(frame, trackedFaces, initialized);
}

// -------------------------------------------------------------

std::vector<cv::Rect> FaceTracker::detectFaces(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;
    if (frame.empty()) return faces;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    // Детекция
    faceCascade.detectMultiScale(gray, faces, scaleFactor, minNeighbors, 0, minSize, maxSize);

    // Фильтр 1: соотношение сторон
    faces.erase(std::remove_if(faces.begin(), faces.end(),
        [](const cv::Rect& r) {
            float aspect = (float)r.width / r.height;
            return aspect < 0.8f || aspect > 1.4f;
        }), faces.end());

    // Фильтр 2: NMS
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

void FaceTracker::updateFaceTracking(const std::vector<cv::Rect>& detectedFaces, float deltaTime) {
    for (auto& face : trackedFaces) face.currentStatus = TargetStatus::lost;
    for (const auto& detectedFace : detectedFaces) {
        cv::Point2f detectedCenter(detectedFace.x + detectedFace.width / 2.0f,
            detectedFace.y + detectedFace.height / 2.0f);
        int closestIndex = findClosestFace(detectedCenter, trackedFaces);
        if (closestIndex >= 0) {
            float distance = calculateDistance(detectedCenter, trackedFaces[closestIndex].center);
            if (distance < 200.0f) {
                updateFacePosition(trackedFaces[closestIndex], detectedFace, deltaTime);
                trackedFaces[closestIndex].currentStatus = TargetStatus::find;
                trackedFaces[closestIndex].age++;
            }
        }
        else {
            TrackedFace newFace;
            newFace.id = nextFaceId++;
            newFace.boundingBox = detectedFace;
            newFace.center = detectedCenter;
            newFace.age = 1;
            newFace.currentStatus = TargetStatus::find;
            updatePositionHistory(newFace, newFace.center);
            trackedFaces.push_back(newFace);
        }
    }
    for (auto& face : trackedFaces) {
        if (face.IsLost()) {
            face.age++;
            face.predictedCenter = getPredictedPosition(face);
        }
    }
    // ... второй цикл (добавление новых лиц) уже есть в другом месте?
    // В оригинале дублирование, оставляем как есть для совместимости
    for (const auto& det : detectedFaces) {
        bool matched = false;
        for (auto& face : trackedFaces) {
            if (computeIOU(det, face.boundingBox) > 0.3f) {
                matched = true;
                break;
            }
        }
        if (!matched) {
            TrackedFace newFace;
            newFace.id = nextFaceId++;
            newFace.boundingBox = det;
            newFace.center = { det.x + det.width * 0.5f, det.y + det.height * 0.5f };
            newFace.age = 1;
            newFace.currentStatus = TargetStatus::find;
            updatePositionHistory(newFace, newFace.center);
            trackedFaces.push_back(newFace);
        }
    }
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

void FaceTracker::reset() {
    trackedFaces.clear();
    nextFaceId = 1;
    initialized = false;
    hasPreviousTime = false;
}

bool FaceTracker::isInitialized() const { return initialized; }

std::vector<TrackedFace> FaceTracker::getTrackedFaces() const { return trackedFaces; }

cv::Point2f FaceTracker::getLargestFaceCenter() const {
    if (trackedFaces.empty()) return cv::Point2f(-1, -1);
    return getLargestFace().center;
}

TrackedFace FaceTracker::getLargestFace() const {
    if (trackedFaces.empty()) return TrackedFace();
    TrackedFace largestFace = trackedFaces[0];
    int maxArea = largestFace.boundingBox.area();
    for (size_t i = 1; i < trackedFaces.size(); ++i) {
        int area = trackedFaces[i].boundingBox.area();
        if (area > maxArea) {
            maxArea = area;
            largestFace = trackedFaces[i];
        }
    }
    return largestFace;
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

// Удалены getClosestPointOnRect и getPointOnCircle – они теперь в рендерере

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

void FaceTracker::removeOldFaces() {
    auto now = std::chrono::steady_clock::now();

    trackedFaces.erase(
        std::remove_if(trackedFaces.begin(), trackedFaces.end(),
            [&](const TrackedFace& f) {
                if (f.currentStatus != TargetStatus::lost) return false;

                float sec = std::chrono::duration<float>(now - f.lostTime).count();
                return sec > lostLifetimeSec;
            }),
        trackedFaces.end()
    );
}


float FaceTracker::computeIOU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width), y2 = std::min(a.y + a.height, b.y + b.height);
    int inter = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - inter;
    return unionArea > 0 ? float(inter) / unionArea : 0.0f;
}
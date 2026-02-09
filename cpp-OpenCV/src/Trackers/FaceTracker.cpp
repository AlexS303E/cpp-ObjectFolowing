#include "FaceTracker.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

FaceTracker::FaceTracker()
    : nextFaceId(1), initialized(false), predictionTime(1.0f), hasPreviousTime(false),
    detectionSkipFrames(2), framesSinceLastDetection(0), enableOptimizedDetection(true),
    confidenceThreshold(0.5f), minTrackFramesForStable(5) {

    // Оптимизированные параметры детектирования
    scaleFactor = 1.15;          // Умеренный компромисс между скоростью и точностью
    minNeighbors = 2;            // Быстрее чем 3 (меньше проверок)
    minSize = cv::Size(40, 40);  // Больше минимальный размер (исключаем мелкие лица)
    maxSize = cv::Size(400, 400); // Увеличиваем максимальный размер

    // Цвета как в ObjectTracker (faceTrackingMode)
    faceColor = cv::Scalar(0, 255, 0);        // Зеленый для внешней рамки
    innerFaceColor = cv::Scalar(0, 200, 0);   // Светло-зеленый для внутренней рамки
    lineColor = cv::Scalar(0, 255, 0);        // Зеленый для линии
    textColor = cv::Scalar(0, 255, 0);        // Зеленый для текста
    predictionColor = cv::Scalar(0, 255, 0);  // Зеленый для прогнозируемой позиции

    try {
        profileFaceCascade.load(FACE_CASCADE_PROFILE);
        faceCascade.load(FACE_CASCADE_FRONTAL);
        disableOptimization();
    }
    catch (...) {
        std::cerr << "Не удалось загрузить каскады Хаара" << std::endl;
    }
}

bool FaceTracker::initialize(const cv::Mat& frame) {
    if (frame.empty()) {
        return false;
    }

    // Обнаружение лиц на первом кадре
    std::vector<cv::Rect> faces = detectFaces(frame);

    if (faces.empty()) {
        return false;
    }

    // Инициализация трекинга для каждого обнаруженного лица
    auto currentTime = std::chrono::steady_clock::now();

    for (const auto& faceRect : faces) {
        TrackedFace face;
        face.id = nextFaceId++;
        face.boundingBox = faceRect;
        face.center = cv::Point2f(
            faceRect.x + faceRect.width / 2.0f,
            faceRect.y + faceRect.height / 2.0f
        );
        face.age = 1;
        face.lost = false;

        // Инициализация истории позиций
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
    if (frame.empty()) {
        return false;
    }

    // Получение текущего времени
    auto currentTime = std::chrono::steady_clock::now();
    float deltaTime = 1.0f / 30.0f; // Значение по умолчанию

    if (hasPreviousTime) {
        deltaTime = std::chrono::duration<float>(currentTime - previousTime).count();
        if (deltaTime < 0.001f) {
            deltaTime = 0.001f; // Минимальное значение
        }
    }

    // Увеличиваем счетчик пропущенных кадров
    framesSinceLastDetection++;

    // Определяем, нужно ли выполнять детектирование на этом кадре
    bool shouldDetect = false;

    // Определяем, нужно ли выполнять детектирование
    if (framesSinceLastDetection >= detectionSkipFrames) {
        shouldDetect = true;
    }

    // Если нет активных лиц, всегда детектируем
    bool hasActiveFaces = false;
    for (const auto& face : trackedFaces) {
        if (!face.lost) {
            hasActiveFaces = true;
            break;
        }
    }

    if (!hasActiveFaces) {
        shouldDetect = true;
        framesSinceLastDetection = 0;
    }

    std::vector<cv::Rect> detectedFaces;

    if (shouldDetect) {
        // Выполняем детектирование лиц
        detectedFaces = detectFaces(frame);
        framesSinceLastDetection = 0;
    }

    // Обновление трекинга (всегда вызываем, даже при пропуске детектирования)
    if (shouldDetect) {
        updateTracksWithDetection(detectedFaces, deltaTime);
    }
    else {
        predictTracksWithoutDetection(deltaTime);
    }

    // Удаление старых лиц (которые потеряны слишком долго)
    removeOldFaces();

    // Обновление времени
    previousTime = currentTime;
    hasPreviousTime = true;

    return !trackedFaces.empty();
}

void FaceTracker::updateFaceTracking(const std::vector<cv::Rect>& detectedFaces, float deltaTime) {
    // Для обратной совместимости оставляем старый метод
    if (!detectedFaces.empty()) {
        updateTracksWithDetection(detectedFaces, deltaTime);
    }
    else {
        predictTracksWithoutDetection(deltaTime);
    }
}

void FaceTracker::updateTracksWithDetection(const std::vector<cv::Rect>& detectedFaces, float deltaTime) {
    // Если обнаружены лица, пытаемся сопоставить их с существующими треками
    for (auto& face : trackedFaces) {
        face.lost = true; // Помечаем все как потерянные
    }

    // Сопоставление обнаруженных лиц с существующими трекерами
    std::vector<bool> matchedDetections(detectedFaces.size(), false);

    // Сначала пытаемся сопоставить с активными треками
    for (size_t i = 0; i < trackedFaces.size(); ++i) {
        if (trackedFaces[i].lost && trackedFaces[i].age < 10) {
            // Пропускаем треки, которые недавно потеряны
            continue;
        }

        float minDistance = std::numeric_limits<float>::max();
        int bestMatchIdx = -1;

        for (size_t j = 0; j < detectedFaces.size(); ++j) {
            if (matchedDetections[j]) continue;

            cv::Point2f detectedCenter(
                detectedFaces[j].x + detectedFaces[j].width / 2.0f,
                detectedFaces[j].y + detectedFaces[j].height / 2.0f
            );

            float distance = calculateDistance(detectedCenter, trackedFaces[i].center);

            // Учитываем размер лица для лучшего сопоставления
            float sizeSimilarity = std::abs(
                (detectedFaces[j].width * detectedFaces[j].height) -
                (trackedFaces[i].boundingBox.width * trackedFaces[i].boundingBox.height)
            ) / (trackedFaces[i].boundingBox.width * trackedFaces[i].boundingBox.height + 1);

            float matchScore = distance * (1.0f + sizeSimilarity * 0.5f);

            if (matchScore < minDistance && distance < 150.0f) {
                minDistance = matchScore;
                bestMatchIdx = j;
            }
        }

        if (bestMatchIdx != -1) {
            updateFacePosition(trackedFaces[i], detectedFaces[bestMatchIdx], deltaTime);
            trackedFaces[i].lost = false;
            trackedFaces[i].age++;
            matchedDetections[bestMatchIdx] = true;
        }
    }

    // Создаем новые треки для нераспределенных обнаружений
    for (size_t i = 0; i < detectedFaces.size(); ++i) {
        if (!matchedDetections[i]) {
            cv::Point2f detectedCenter(
                detectedFaces[i].x + detectedFaces[i].width / 2.0f,
                detectedFaces[i].y + detectedFaces[i].height / 2.0f
            );

            TrackedFace newFace;
            newFace.id = nextFaceId++;
            newFace.boundingBox = detectedFaces[i];
            newFace.center = detectedCenter;
            newFace.age = 1;
            newFace.lost = false;
            updatePositionHistory(newFace, newFace.center);
            trackedFaces.push_back(newFace);
        }
    }

    // Обновляем возраст и прогноз для потерянных треков
    for (auto& face : trackedFaces) {
        if (face.lost) {
            face.age++;
            // Для стабильных треков увеличиваем уверенность
            if (face.age < 15) {
                // Используем прогноз на основе скорости
                face.center += face.velocity * deltaTime;
                face.boundingBox.x += face.velocity.x * deltaTime;
                face.boundingBox.y += face.velocity.y * deltaTime;
                face.predictedCenter = getPredictedPosition(face);
                updatePositionHistory(face, face.center);
            }
        }
    }
}

void FaceTracker::predictTracksWithoutDetection(float deltaTime) {
    // Прогнозируем позиции на основе скорости
    for (auto& face : trackedFaces) {
        if (!face.lost) {
            // Для активных треков обновляем позицию на основе скорости
            face.center += face.velocity * deltaTime;
            face.boundingBox.x += face.velocity.x * deltaTime;
            face.boundingBox.y += face.velocity.y * deltaTime;

            // Постепенно уменьшаем скорость (демпфирование)
            face.velocity *= 0.95f;

            // Обновляем историю позиций
            updatePositionHistory(face, face.center);

            // Обновляем прогноз
            face.predictedCenter = getPredictedPosition(face);

            face.age++;

            // Если трек стабильный, не помечаем как потерянный
            if (face.age > minTrackFramesForStable) {
                face.lost = false;
            }
        }
        else {
            // Для потерянных треков просто увеличиваем возраст
            face.age++;
        }
    }
}

std::vector<cv::Rect> FaceTracker::detectFaces(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;

    if (frame.empty()) {
        return faces;
    }

    // Преобразование в оттенки серого
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Выравнивание гистограммы
    cv::equalizeHist(gray, gray);

    // Используем оптимизированные параметры для быстрого детектирования
    if (enableOptimizedDetection) {
        // Быстрые параметры для детектирования
        double fastScaleFactor = 1.15;     // Быстрее сканирование
        int fastMinNeighbors = 2;          // Меньше проверок
        cv::Size fastMinSize = cv::Size(40, 40);
        cv::Size fastMaxSize = cv::Size(400, 400);

        // Обнаружение фронтальных лиц с оптимизированными параметрами
        faceCascade.detectMultiScale(
            gray,
            faces,
            fastScaleFactor,
            fastMinNeighbors,
            0,
            fastMinSize,
            fastMaxSize
        );
    }
    else {
        faceCascade.detectMultiScale(
            gray,
            faces,
            scaleFactor,
            minNeighbors,
            0,
            minSize,
            maxSize
        );
    }

    // Если фронтальные лица не найдены, ищем профильные (реже)
    if (faces.empty() && !profileFaceCascade.empty()) {
        // Для профильных лиц используем более агрессивные параметры
        std::vector<cv::Rect> profileFaces;
        profileFaceCascade.detectMultiScale(
            gray,
            profileFaces,
            1.2,       // Быстрее для профильных лиц
            2,         // Меньше проверок
            0,
            cv::Size(50, 50),
            cv::Size(300, 300)
        );

        faces.insert(faces.end(), profileFaces.begin(), profileFaces.end());
    }

    return faces;
}

void FaceTracker::updateFacePosition(TrackedFace& face, const cv::Rect& newRect, float deltaTime) {
    // Сглаживание bounding box с адаптивным коэффициентом
    float smoothingFactor = 0.7f; // Основной коэффициент сглаживания

    // Для старых треков используем большее сглаживание
    if (face.age > minTrackFramesForStable) {
        smoothingFactor = 0.8f;
    }

    float newFactor = 1.0f - smoothingFactor;

    face.boundingBox.x = smoothingFactor * face.boundingBox.x + newFactor * newRect.x;
    face.boundingBox.y = smoothingFactor * face.boundingBox.y + newFactor * newRect.y;
    face.boundingBox.width = smoothingFactor * face.boundingBox.width + newFactor * newRect.width;
    face.boundingBox.height = smoothingFactor * face.boundingBox.height + newFactor * newRect.height;

    // Обновление центра
    cv::Point2f newCenter(
        face.boundingBox.x + face.boundingBox.width / 2.0f,
        face.boundingBox.y + face.boundingBox.height / 2.0f
    );

    // Обновление скорости
    updateVelocity(face, newCenter, deltaTime);

    // Обновление истории позиций
    updatePositionHistory(face, newCenter);

    // Обновление центра
    face.center = newCenter;

    // Расчет прогнозируемой позиции
    face.predictedCenter = getPredictedPosition(face);
}

void FaceTracker::updateVelocity(TrackedFace& face, const cv::Point2f& newPosition, float deltaTime) {
    if (face.positionHistory.size() >= 2 && deltaTime > 0.001f) {
        // Берем последнюю позицию из истории
        cv::Point2f lastPosition = face.positionHistory.back();

        // Вычисляем смещение
        cv::Point2f displacement = newPosition - lastPosition;

        // Вычисляем скорость (пикселей в секунду)
        cv::Point2f newVelocity = displacement / deltaTime;

        // Адаптивное сглаживание скорости
        float alpha = 0.3f;
        if (face.age > 10) {
            alpha = 0.2f; // Для старых треков более плавное обновление
        }

        face.velocity = (1.0f - alpha) * face.velocity + alpha * newVelocity;

        // Ограничиваем максимальную скорость
        float maxSpeed = 300.0f; // пикселей в секунду
        float currentSpeed = std::sqrt(face.velocity.x * face.velocity.x + face.velocity.y * face.velocity.y);
        if (currentSpeed > maxSpeed) {
            face.velocity = face.velocity * (maxSpeed / currentSpeed);
        }
    }
}

void FaceTracker::updatePositionHistory(TrackedFace& face, const cv::Point2f& newPosition) {
    face.positionHistory.push_back(newPosition);

    // Ограничиваем размер истории
    if (face.positionHistory.size() > maxHistorySize) {
        face.positionHistory.pop_front();
    }
}

cv::Point2f FaceTracker::getSmoothedPosition(const TrackedFace& face) const {
    if (face.positionHistory.empty()) {
        return face.center;
    }

    cv::Point2f sum(0, 0);
    for (const auto& pos : face.positionHistory) {
        sum += pos;
    }

    return cv::Point2f(
        sum.x / face.positionHistory.size(),
        sum.y / face.positionHistory.size()
    );
}

cv::Point2f FaceTracker::getPredictedPosition(const TrackedFace& face) const {
    // Прогнозируем позицию через predictionTime секунд
    cv::Point2f predicted = face.center + face.velocity * predictionTime;

    return predicted;
}

void FaceTracker::drawTrackingInfo(cv::Mat& frame) const {
    if (trackedFaces.empty()) {
        // Если лица не обнаружены
        cv::Rect infoPanel(10, 10, 300, 90);
        cv::rectangle(frame, infoPanel, cv::Scalar(50, 50, 50), -1);
        cv::rectangle(frame, infoPanel, cv::Scalar(200, 200, 200), 1);

        cv::putText(frame, "Face Tracker Mode", cv::Point(20, 35),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, "Status: NO FACES", cv::Point(20, 65),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

        return;
    }

    // Отрисовка каждого отслеживаемого лица
    for (size_t i = 0; i < trackedFaces.size(); ++i) {
        const TrackedFace& face = trackedFaces[i];

        // 1. Отрисовка рамки лица
        if (face.lost) {
            // Прерывистая красная рамка для потерянных лиц
            int dashLength = 10;
            int gapLength = 5;

            // Верхняя граница
            for (int x = face.boundingBox.x; x < face.boundingBox.x + face.boundingBox.width; x += dashLength + gapLength) {
                int endX = std::min(x + dashLength, face.boundingBox.x + face.boundingBox.width);
                cv::line(frame, cv::Point(x, face.boundingBox.y),
                    cv::Point(endX, face.boundingBox.y),
                    cv::Scalar(0, 0, 255), BORDER_THICKNESS);
            }

            // Нижняя граница
            for (int x = face.boundingBox.x; x < face.boundingBox.x + face.boundingBox.width; x += dashLength + gapLength) {
                int endX = std::min(x + dashLength, face.boundingBox.x + face.boundingBox.width);
                cv::line(frame, cv::Point(x, face.boundingBox.y + face.boundingBox.height),
                    cv::Point(endX, face.boundingBox.y + face.boundingBox.height),
                    cv::Scalar(0, 0, 255), BORDER_THICKNESS);
            }

            // Левая граница
            for (int y = face.boundingBox.y; y < face.boundingBox.y + face.boundingBox.height; y += dashLength + gapLength) {
                int endY = std::min(y + dashLength, face.boundingBox.y + face.boundingBox.height);
                cv::line(frame, cv::Point(face.boundingBox.x, y),
                    cv::Point(face.boundingBox.x, endY),
                    cv::Scalar(0, 0, 255), BORDER_THICKNESS);
            }

            // Правая граница
            for (int y = face.boundingBox.y; y < face.boundingBox.y + face.boundingBox.height; y += dashLength + gapLength) {
                int endY = std::min(y + dashLength, face.boundingBox.y + face.boundingBox.height);
                cv::line(frame, cv::Point(face.boundingBox.x + face.boundingBox.width, y),
                    cv::Point(face.boundingBox.x + face.boundingBox.width, endY),
                    cv::Scalar(0, 0, 255), BORDER_THICKNESS);
            }

            // Подпись "LOST"
            cv::putText(frame, "LOST",
                cv::Point(face.boundingBox.x, face.boundingBox.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
        else {
            // Зеленая рамка для найденных лиц
            cv::rectangle(frame, face.boundingBox, faceColor, BORDER_THICKNESS);
            cv::rectangle(frame, face.boundingBox, innerFaceColor, INNER_BORDER_THICKNESS);

            // Подпись "FACE"
            cv::putText(frame, "FACE",
                cv::Point(face.boundingBox.x, face.boundingBox.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2);

            // 2. Отрисовка желтого круга в прогнозируемой позиции
            cv::Point2f prediction = face.predictedCenter;

            // Ограничиваем прогнозируемую позицию рамками кадра
            prediction.x = std::max(0.0f, std::min(prediction.x, (float)frame.cols));
            prediction.y = std::max(0.0f, std::min(prediction.y, (float)frame.rows));

            cv::circle(frame, prediction, PREDICTION_RADIUS, predictionColor, PREDICTION_THICKNESS);

            // 3. Определение, находится ли круг внутри прямоугольника
            bool circleInsideRect = prediction.x >= face.boundingBox.x &&
                prediction.x <= face.boundingBox.x + face.boundingBox.width &&
                prediction.y >= face.boundingBox.y &&
                prediction.y <= face.boundingBox.y + face.boundingBox.height;

            // 4. Отрисовка линий соединения (как в ObjectTracker)
            if (circleInsideRect) {
                // Если круг внутри квадрата - рисуем линию из центра квадрата до границы окружности
                cv::Point2f rectCenter = face.center;

                // Вычисляем вектор от центра квадрата к центру круга
                cv::Point2f direction = prediction - rectCenter;
                float distance = std::sqrt(direction.x * direction.x + direction.y * direction.y);

                if (distance > 0) {
                    // Нормализуем вектор
                    direction.x /= distance;
                    direction.y /= distance;

                    // Вычисляем точку на границе окружности (вдоль того же направления)
                    cv::Point2f circleBoundary = prediction - direction * PREDICTION_RADIUS;

                    // Рисуем линию от центра квадрата до границы окружности
                    cv::line(frame, rectCenter, circleBoundary, lineColor, LINE_THICKNESS);
                }
            }
            else {
                // Если круг НЕ внутри квадрата - находим точку на границе квадрата и окружности
                // Находим точку на границе квадрата, ближайшую к кругу
                cv::Point2f rectBoundary = getClosestPointOnRect(face.boundingBox, prediction);

                // Вычисляем вектор от центра окружности к точке на границе квадрата
                cv::Point2f direction = rectBoundary - prediction;
                float distance = std::sqrt(direction.x * direction.x + direction.y * direction.y);

                if (distance > 0) {
                    // Нормализуем вектор
                    direction.x /= distance;
                    direction.y /= distance;

                    // Вычисляем точку на границе окружности (вдоль того же направления)
                    cv::Point2f circleBoundary = prediction + direction * PREDICTION_RADIUS;

                    // Рисуем линию от границы квадрата до границы окружности
                    cv::line(frame, rectBoundary, circleBoundary, lineColor, LINE_THICKNESS);
                }
            }

            // 5. Подписи
            cv::putText(frame, "prediction",
                cv::Point(prediction.x + PREDICTION_RADIUS + 5, prediction.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, predictionColor, 1);

            // 6. Информация о лице
            std::string info = "ID: " + std::to_string(face.id) +
                " Age: " + std::to_string(face.age);
            cv::putText(frame, info,
                cv::Point(face.boundingBox.x, face.boundingBox.y + face.boundingBox.height + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
        }
    }

    // 7. Информационная панель
    cv::Rect infoPanel(10, 10, 300, 140);
    cv::rectangle(frame, infoPanel, cv::Scalar(50, 50, 50), -1);
    cv::rectangle(frame, infoPanel, cv::Scalar(200, 200, 200), 1);

    cv::putText(frame, "Face Tracker Mode", cv::Point(20, 35),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // Статус
    int activeFaces = 0;
    for (const auto& face : trackedFaces) {
        if (!face.lost) activeFaces++;
    }

    std::string statusText = "Status: " + std::to_string(activeFaces) +
        " active / " + std::to_string(trackedFaces.size()) + " total";
    cv::Scalar statusColor = (activeFaces > 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

    cv::putText(frame, statusText, cv::Point(20, 65),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, statusColor, 2);

    // Статистика по самому большому лицу
    if (!trackedFaces.empty()) {
        TrackedFace largestFace = getLargestFace();

        std::string sizeStr = "Largest: " + std::to_string(largestFace.boundingBox.width) +
            "x" + std::to_string(largestFace.boundingBox.height);
        cv::putText(frame, sizeStr, cv::Point(20, 95),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

        std::string velStr = "Velocity: " +
            std::to_string((int)largestFace.velocity.x) + ", " +
            std::to_string((int)largestFace.velocity.y) + " px/s";
        cv::putText(frame, velStr, cv::Point(20, 115),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

        std::string predStr = "Prediction: " +
            std::to_string((int)largestFace.predictedCenter.x) + ", " +
            std::to_string((int)largestFace.predictedCenter.y);
        cv::putText(frame, predStr, cv::Point(20, 135),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    }
}

void FaceTracker::reset() {
    trackedFaces.clear();
    nextFaceId = 1;
    initialized = false;
    hasPreviousTime = false;
    framesSinceLastDetection = 0;
}

bool FaceTracker::isInitialized() const {
    return initialized;
}

std::vector<TrackedFace> FaceTracker::getTrackedFaces() const {
    return trackedFaces;
}

cv::Point2f FaceTracker::getLargestFaceCenter() const {
    if (trackedFaces.empty()) {
        return cv::Point2f(-1, -1);
    }

    TrackedFace largestFace = getLargestFace();
    return largestFace.center;
}

TrackedFace FaceTracker::getLargestFace() const {
    if (trackedFaces.empty()) {
        return TrackedFace();
    }

    TrackedFace largestFace = trackedFaces[0];
    int maxArea = largestFace.boundingBox.width * largestFace.boundingBox.height;

    for (size_t i = 1; i < trackedFaces.size(); ++i) {
        int area = trackedFaces[i].boundingBox.width * trackedFaces[i].boundingBox.height;
        if (area > maxArea) {
            maxArea = area;
            largestFace = trackedFaces[i];
        }
    }

    return largestFace;
}

cv::Rect FaceTracker::getLargestFaceRect(const std::vector<cv::Rect>& faces) {
    if (faces.empty()) {
        return cv::Rect();
    }

    cv::Rect largestFace = faces[0];
    int maxArea = largestFace.width * largestFace.height;

    for (size_t i = 1; i < faces.size(); ++i) {
        int area = faces[i].width * faces[i].height;
        if (area > maxArea) {
            maxArea = area;
            largestFace = faces[i];
        }
    }

    return largestFace;
}

cv::Point2f FaceTracker::getClosestPointOnRect(const cv::Rect& rect, const cv::Point2f& point) const {
    float closestX = std::max((float)rect.x, std::min(point.x, (float)(rect.x + rect.width)));
    float closestY = std::max((float)rect.y, std::min(point.y, (float)(rect.y + rect.height)));

    if (closestX > rect.x && closestX < rect.x + rect.width &&
        closestY > rect.y && closestY < rect.y + rect.height) {

        float distToLeft = closestX - rect.x;
        float distToRight = (rect.x + rect.width) - closestX;
        float distToTop = closestY - rect.y;
        float distToBottom = (rect.y + rect.height) - closestY;

        float minDist = std::min({ distToLeft, distToRight, distToTop, distToBottom });

        if (minDist == distToLeft) closestX = rect.x;
        else if (minDist == distToRight) closestX = rect.x + rect.width;
        else if (minDist == distToTop) closestY = rect.y;
        else closestY = rect.y + rect.height;
    }

    return cv::Point2f(closestX, closestY);
}

cv::Point2f FaceTracker::getPointOnCircle(const cv::Point2f& circleCenter,
    const cv::Point2f& targetPoint,
    float radius) const {
    cv::Point2f direction = targetPoint - circleCenter;
    float distance = std::sqrt(direction.x * direction.x + direction.y * direction.y);

    if (distance > 0) {
        direction.x = direction.x / distance * radius;
        direction.y = direction.y / distance * radius;
    }

    return cv::Point2f(circleCenter.x + direction.x, circleCenter.y + direction.y);
}

float FaceTracker::calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) const {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

int FaceTracker::findClosestFace(const cv::Point2f& center, const std::vector<TrackedFace>& faces) const {
    if (faces.empty()) {
        return -1;
    }

    int closestIndex = -1;
    float minDistance = 200.0f; // Максимальное расстояние для сопоставления

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
    // Удаляем лица, которые потеряны слишком долго (более 30 кадров)
    trackedFaces.erase(
        std::remove_if(trackedFaces.begin(), trackedFaces.end(),
            [](const TrackedFace& face) {
                return face.lost && face.age > 30;
            }),
        trackedFaces.end()
    );
}

void FaceTracker::setOptimizationParams(int skipFrames, bool optimizedDetection) {
    detectionSkipFrames = std::max(1, skipFrames); // Минимум 1 кадр
    enableOptimizedDetection = optimizedDetection;
    framesSinceLastDetection = 0;

    std::cout << "FaceTracker оптимизация: пропуск кадров = " << detectionSkipFrames
        << ", оптимизированное детектирование = " << (optimizedDetection ? "да" : "нет") << std::endl;
}

void FaceTracker::disableOptimization() {
    detectionSkipFrames = 1;  // Детектировать на каждом кадре
    enableOptimizedDetection = false;  // Использовать стандартные параметры детектирования
    framesSinceLastDetection = 0;

    // Используем более точные параметры детектирования
    scaleFactor = 1.1;          // Стандартный параметр (больше точность, меньше скорость)
    minNeighbors = 3;           // Стандартный параметр (больше проверок, меньше ложных срабатываний)
    minSize = cv::Size(30, 30); // Стандартный минимальный размер
    maxSize = cv::Size(300, 300); // Стандартный максимальный размер

    std::cout << "FaceTracker оптимизация ОТКЛЮЧЕНА: детектирование на каждом кадре с точными параметрами" << std::endl;
}

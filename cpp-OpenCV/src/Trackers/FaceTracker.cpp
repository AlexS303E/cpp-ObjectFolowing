#include "FaceTracker.h"
#include <iostream>
#include <algorithm>

FaceTracker::FaceTracker()
    : nextFaceId(1), initialized(false), predictionTime(1.0f), hasPreviousTime(false) {

    // Инициализация параметров
    scaleFactor = 1.1;
    minNeighbors = 3;
    minSize = cv::Size(30, 30);
    maxSize = cv::Size(300, 300);

    // Цвета как в ObjectTracker (faceTrackingMode)
    faceColor = cv::Scalar(0, 255, 0);        // Зеленый для внешней рамки
    innerFaceColor = cv::Scalar(0, 200, 0);   // Светло-зеленый для внутренней рамки
    lineColor = cv::Scalar(0, 255, 0);        // Зеленый для линии
    textColor = cv::Scalar(0, 255, 0);        // Зеленый для текста
    predictionColor = cv::Scalar(0, 255, 0); // Желтый для прогнозируемой позиции

    try {
        //profileFaceCascade.load(FACE_CASCADE_PROFILE);
        faceCascade.load(FACE_CASCADE_FRONTAL);
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
    if (frame.empty())
        return false;

    auto currentTime = std::chrono::steady_clock::now();
    float deltaTime = 0.0f;

    // Правильный расчет времени между кадрами
    if (hasPreviousTime) {
        deltaTime = std::chrono::duration<float>(currentTime - previousTime).count();
    }
    hasPreviousTime = true;
    previousTime = currentTime;

    // Если deltaTime слишком мала, используем дефолтное значение
    if (deltaTime < 0.001f) {
        deltaTime = 1.0f / 30.0f; // Предполагаем 30 FPS
    }

    static int frameCounter = 0;
    frameCounter++;

    // Детектор не каждый кадр
    bool doDetect = (frameCounter % 5 == 0);

    std::vector<cv::Rect> detectedFaces;
    if (doDetect) {
        detectedFaces = detectFaces(frame);
    }

    // --- TRACK EXISTING ---
    for (auto& face : trackedFaces) {
        face.matched = false;

        // Правильное предсказание с использованием времени
        face.predicted = face.center + face.velocity * deltaTime;
    }

    // --- MATCH DETECTIONS TO TRACKS ---
    for (const auto& det : detectedFaces) {
        float bestIOU = 0.0f;
        int bestIdx = -1;

        for (int i = 0; i < (int)trackedFaces.size(); i++) {
            auto& face = trackedFaces[i];
            if (face.lost) continue;

            float iou = computeIOU(det, face.boundingBox);
            if (iou > bestIOU) {
                bestIOU = iou;
                bestIdx = i;
            }
        }

        if (bestIdx >= 0 && bestIOU > 0.55f) {
            auto& face = trackedFaces[bestIdx];

            cv::Point2f newCenter(
                det.x + det.width * 0.5f,
                det.y + det.height * 0.5f
            );

            cv::Point2f lastCenter = face.center;

            // Обновление центра с сглаживанием
            face.center = 0.7f * face.center + 0.3f * newCenter;

            // ОБНОВЛЕНИЕ СКОРОСТИ (как в ObjectTracker)
            cv::Point2f displacement = face.center - lastCenter;

            if (deltaTime > 0) {
                cv::Point2f newVelocity = displacement / deltaTime;

                // Сглаживание скорости (фильтр низких частот)
                float alpha = 0.3f * std::min(deltaTime * 30.0f, 1.0f);
                face.velocity = (1.0f - alpha) * face.velocity + alpha * newVelocity;
            }

            face.boundingBox = det;
            face.lostFrames = 0;
            face.matched = true;

            // Обновление истории позиций
            updatePositionHistory(face, face.center);
        }
    }

    // --- HANDLE LOST ---
    for (auto& face : trackedFaces) {
        if (!face.matched) {
            face.lostFrames++;
            face.center = face.predicted; // Используем предсказание
            face.lost = (face.lostFrames > 10); // Маркируем как потерянное после 10 кадров
        }
        else {
            face.lost = false;
            face.lostFrames = 0;
        }

        // Обновление предсказанной позиции (через 1 секунду)
        face.predictedCenter = face.center + face.velocity * predictionTime;
    }

    // --- ADD NEW FACES ---
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
            f.center = { det.x + det.width * 0.5f, det.y + det.height * 0.5f };
            f.velocity = { 0,0 };
            f.predictedCenter = f.center;
            f.lostFrames = 0;
            f.matched = true;
            f.lost = false;
            f.age = 1;

            // Инициализация времени
            f.previousTime = currentTime;
            f.hasPreviousPosition = false;

            trackedFaces.push_back(f);
        }
    }

    // --- REMOVE DEAD ---
    trackedFaces.erase(
        std::remove_if(trackedFaces.begin(), trackedFaces.end(),
            [&](const TrackedFace& f) {
                return f.lostFrames > maxLostFrames;
            }),
        trackedFaces.end()
    );

    return true;
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

    // Обнаружение фронтальных лиц
    faceCascade.detectMultiScale(
        gray,
        faces,
        scaleFactor,
        minNeighbors,
        0,
        minSize,
        maxSize
    );



    // Если фронтальные лица не найдены, ищем профильные
    //if (faces.empty() && !profileFaceCascade.empty()) {
    //    std::vector<cv::Rect> profileFaces;
    //    profileFaceCascade.detectMultiScale(
    //        gray,
    //        profileFaces,
    //        scaleFactor,
    //        minNeighbors,
    //        0,
    //        minSize,
    //        maxSize
    //    );

    //    faces.insert(faces.end(), profileFaces.begin(), profileFaces.end());
    //}

    return faces;
}

void FaceTracker::updateFaceTracking(const std::vector<cv::Rect>& detectedFaces, float deltaTime) {
    // Отметим все текущие лица как потерянные
    for (auto& face : trackedFaces) {
        face.lost = true;
    }

    // Сопоставление обнаруженных лиц с существующими трекерами
    for (const auto& detectedFace : detectedFaces) {
        cv::Point2f detectedCenter(
            detectedFace.x + detectedFace.width / 2.0f,
            detectedFace.y + detectedFace.height / 2.0f
        );

        // Поиск ближайшего существующего лица
        int closestIndex = findClosestFace(detectedCenter, trackedFaces);

        if (closestIndex >= 0) {
            // Обновляем существующее лицо
            float distance = calculateDistance(detectedCenter, trackedFaces[closestIndex].center);

            if (distance < 200.0f) { // Максимальное расстояние для сопоставления
                updateFacePosition(trackedFaces[closestIndex], detectedFace, deltaTime);
                trackedFaces[closestIndex].lost = false;
                trackedFaces[closestIndex].age++;
            }
        }
        else {
            // Создаем новый трекер для лица
            TrackedFace newFace;
            newFace.id = nextFaceId++;
            newFace.boundingBox = detectedFace;
            newFace.center = detectedCenter;
            newFace.age = 1;
            newFace.lost = false;

            updatePositionHistory(newFace, newFace.center);
            trackedFaces.push_back(newFace);
        }
    }

    // Для потерянных лиц увеличиваем возраст и обновляем прогноз
    for (auto& face : trackedFaces) {
        if (face.lost) {
            face.age++;
            // Используем последнюю известную скорость для прогноза
            face.predictedCenter = getPredictedPosition(face);
        }
    }

    for (const auto& det : detectedFaces) {
        bool matched = false;

        for (auto& face : trackedFaces) {
            float iou = computeIOU(det, face.boundingBox);
            if (iou > 0.3f) {
                matched = true;
                break;
            }
        }

        if (!matched) {
            TrackedFace newFace;
            newFace.id = nextFaceId++;
            newFace.boundingBox = det;
            newFace.center = {
                det.x + det.width * 0.5f,
                det.y + det.height * 0.5f
            };
            newFace.age = 1;
            newFace.lost = false;

            updatePositionHistory(newFace, newFace.center);
            trackedFaces.push_back(newFace);
        }
    }
}

void FaceTracker::updateFacePosition(TrackedFace& face, const cv::Rect& newRect, float deltaTime) {
    // Сглаживание bounding box (70% старого + 30% нового)
    face.boundingBox.x = 0.7f * face.boundingBox.x + 0.3f * newRect.x;
    face.boundingBox.y = 0.7f * face.boundingBox.y + 0.3f * newRect.y;
    face.boundingBox.width = 0.7f * face.boundingBox.width + 0.3f * newRect.width;
    face.boundingBox.height = 0.7f * face.boundingBox.height + 0.3f * newRect.height;

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
    if (face.positionHistory.size() >= 2) {
        // Берем последнюю позицию из истории
        cv::Point2f lastPosition = face.positionHistory.back();

        // Вычисляем смещение
        cv::Point2f displacement = newPosition - lastPosition;

        // Вычисляем скорость (пикселей в секунду)
        if (deltaTime > 0) {
            cv::Point2f newVelocity = displacement / deltaTime;

            // Сглаживание скорости (фильтр низких частот)
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

    // Если время слишком маленькое, пропускаем
    if (actualDeltaTime < 0.001f) {
        return;
    }

    // Вычисление скорости
    cv::Point2f displacement = newPosition - face.previousPosition;
    cv::Point2f newVelocity = displacement / actualDeltaTime;

    // Сглаживание скорости
    float alpha = 0.3f * std::min(actualDeltaTime * 30.0f, 1.0f);
    face.velocity = (1.0f - alpha) * face.velocity + alpha * newVelocity;

    // Обновление времени и позиции
    face.previousPosition = newPosition;
    face.previousTime = currentTime;
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
    cv::Point2f lastCenter = face.center;


    cv::Point2f predicted = face.center + face.velocity * predictionTime;

    // Ограничиваем прогноз рамками изображения
    // (это будет уточнено при отрисовке)
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
                float distance = sqrt(direction.x * direction.x + direction.y * direction.y);

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
                float distance = sqrt(direction.x * direction.x + direction.y * direction.y);

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
    float minDistance = 1000.0f; // Большое начальное значение

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

float FaceTracker::computeIOU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int inter = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - inter;

    return unionArea > 0 ? float(inter) / unionArea : 0.0f;
}
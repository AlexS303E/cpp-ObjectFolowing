#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "EdgeDetector.h"
#include "ObjectTracker.h"
#include "WebcamViewer.h"

// Функция для обработки кадра (сохранение оригинальной логики)
void processFrame(cv::Mat& frame,
    CannyEdgeDetector& cannyDetector,
    CombinedEdgeDetector& combinedDetector,
    ObjectTracker& objectTracker,
    bool& useCombinedDetector,
    bool& showOnlyEdges,
    bool& objectTrackingEnabled,
    bool& faceTrackingMode,
    float fps) {

    cv::Mat displayFrame;

    // Если разрешение слишком большое, уменьшаем для обработки
    if (frame.cols > 800) {
        float scale = 0.75f;
        cv::resize(frame, displayFrame, cv::Size(), scale, scale);
    }
    else {
        displayFrame = frame.clone();
    }

    cv::Mat originalFrame = displayFrame.clone();

    // Обработка в зависимости от выбранного детектора
    if (useCombinedDetector) {
        if (showOnlyEdges) {
            combinedDetector.detectOnlyEdges(displayFrame);
            cv::putText(displayFrame, "Mode: Edges Only (Combined)", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        }
        else {
            combinedDetector.detectAndDraw(displayFrame);
        }
    }
    else {
        if (showOnlyEdges) {
            cannyDetector.detectOnlyEdges(displayFrame);
            cv::putText(displayFrame, "Mode: Edges Only (Canny)", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        }
        else {
            cannyDetector.detectAndDraw(displayFrame);
        }
    }

    // Отображаем информацию о детекторе
    std::string detectorName = useCombinedDetector ?
        "Combined Edge Detector" : "Canny Edge Detector";

    cv::putText(displayFrame, "Detector: " + detectorName,
        cv::Point(10, displayFrame.rows - 100), cv::FONT_HERSHEY_SIMPLEX,
        0.6, cv::Scalar(255, 200, 0), 2);

    // Обработка трекинга объекта
    if (objectTrackingEnabled) {
        if (!objectTracker.isInitialized()) {
            // Инициализируем трекер
            if (faceTrackingMode) {
                std::cout << "Initializing face tracker..." << std::endl;
            }

            if (objectTracker.initialize(originalFrame)) {
                std::cout << "Object tracker initialized successfully!" << std::endl;
            }
            else {
                std::cout << "Failed to initialize object tracker!" << std::endl;
                objectTrackingEnabled = false;
            }
        }
        else {
            // Обновляем трекер в реальном времени
            bool trackingSuccess = objectTracker.update(originalFrame);

            // Рисуем информацию о трекинге поверх всего
            objectTracker.drawTrackingInfo(displayFrame);

            // Отображаем статус трекинга
            std::string trackingStatus = trackingSuccess ? "ACTIVE" : "SEARCHING";
            cv::Scalar statusColor = trackingSuccess ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

            cv::putText(displayFrame, "TRACKING: " + trackingStatus,
                cv::Point(displayFrame.cols - 220, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);
        }
    }

    // Отображение информации о режиме
    std::string modeInfo = showOnlyEdges ? "[Edges Only]" : "[Overlay]";
    if (objectTrackingEnabled) {
        modeInfo += faceTrackingMode ? " [Face Tracking]" : " [Object Tracking]";
    }

    cv::putText(displayFrame, modeInfo, cv::Point(displayFrame.cols - 250, 60),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 100, 0), 2);

    // Отображение FPS
    std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
    cv::putText(displayFrame, fpsText, cv::Point(displayFrame.cols - 150, 90),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

    // Отображение подсказок управления
    cv::putText(displayFrame, "[ESC/Q] - Exit", cv::Point(10, displayFrame.rows - 70),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    cv::putText(displayFrame, "[1/2] - Switch Detector", cv::Point(10, displayFrame.rows - 45),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    cv::putText(displayFrame, "[c] - Toggle Edge Display", cv::Point(10, displayFrame.rows - 25),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    cv::putText(displayFrame, "[t] - Toggle Tracking", cv::Point(displayFrame.cols - 250, displayFrame.rows - 25),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    cv::putText(displayFrame, "[f] - Face/Object Mode", cv::Point(displayFrame.cols - 250, displayFrame.rows - 45),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

    // Копируем обработанный кадр назад
    frame = displayFrame;
}

// Функция для обработки клавиш (сохранение оригинальной логики)
void processKey(int key,
    bool& useCombinedDetector,
    bool& showOnlyEdges,
    bool& objectTrackingEnabled,
    bool& faceTrackingMode,
    CannyEdgeDetector& cannyDetector,
    CombinedEdgeDetector& combinedDetector,
    ObjectTracker& objectTracker,
    double& cannyThresh1, double& cannyThresh2,
    double& combinedThresh1, double& combinedThresh2,
    int& dilateSize, int& erodeSize,
    WebcamViewer& webcam) {

    // Переключение детекторов
    if (key == '1') {
        useCombinedDetector = false;
        std::cout << "Switched to Canny Edge Detector" << std::endl;
    }
    if (key == '2') {
        useCombinedDetector = true;
        std::cout << "Switched to Combined Edge Detector" << std::endl;
    }

    // Переключение режима отображения
    if (key == 'c' || key == 'C') {
        showOnlyEdges = !showOnlyEdges;
        std::cout << "Mode switched: "
            << (showOnlyEdges ? "edges only" : "overlay mode")
            << std::endl;
    }

    // Включение/выключение трекинга объекта
    if (key == 't' || key == 'T') {
        objectTrackingEnabled = !objectTrackingEnabled;
        if (objectTrackingEnabled) {
            std::cout << "Object tracking ENABLED" << std::endl;
            if (faceTrackingMode) {
                std::cout << "Mode: Face Tracking" << std::endl;
            }
        }
        else {
            std::cout << "Object tracking DISABLED" << std::endl;
            objectTracker.reset();
        }
    }

    // Переключение режима лица/объекта
    if (key == 'f' || key == 'F') {
        faceTrackingMode = !faceTrackingMode;
        objectTracker.setFaceTrackingMode(faceTrackingMode);

        if (faceTrackingMode) {
            std::cout << "Face tracking mode ENABLED" << std::endl;
            if (!objectTracker.isFaceTrackingMode()) {
                std::cout << "WARNING: Face cascade not loaded! Check haarcascade_frontalface_default.xml" << std::endl;
                faceTrackingMode = false;
            }
        }
        else {
            std::cout << "Object tracking mode ENABLED" << std::endl;
        }

        if (objectTrackingEnabled) {
            objectTracker.reset();
            std::cout << "Tracker reset for new mode" << std::endl;
        }
    }

    // Сброс трекера
    if (key == '0') {
        objectTracker.reset();
        std::cout << "Object tracker reset" << std::endl;
    }

    // Сброс параметров детектора
    if (key == 'r' || key == 'R') {
        if (useCombinedDetector) {
            combinedThresh1 = 50.0; combinedThresh2 = 150.0;
            dilateSize = 2; erodeSize = 2;
            combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
            std::cout << "Combined detector parameters reset: thresholds " << combinedThresh1 << ", " << combinedThresh2
                << ", dilation " << dilateSize << ", erosion " << erodeSize << std::endl;
        }
        else {
            cannyThresh1 = 50.0; cannyThresh2 = 150.0;
            cannyDetector = CannyEdgeDetector(cannyThresh1, cannyThresh2);
            std::cout << "Canny detector parameters reset: " << cannyThresh1 << ", " << cannyThresh2 << std::endl;
        }
    }

    // Изменение порогов Канни
    if (key == '+' || key == '=') {
        if (useCombinedDetector) {
            combinedThresh1 += 10; combinedThresh2 += 20;
            combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
            std::cout << "Combined Canny thresholds increased: " << combinedThresh1 << ", " << combinedThresh2 << std::endl;
        }
        else {
            cannyThresh1 += 10; cannyThresh2 += 20;
            cannyDetector = CannyEdgeDetector(cannyThresh1, cannyThresh2);
            std::cout << "Canny thresholds increased: " << cannyThresh1 << ", " << cannyThresh2 << std::endl;
        }
    }

    if (key == '-' || key == '_') {
        if (useCombinedDetector) {
            combinedThresh1 = std::max(10.0, combinedThresh1 - 10);
            combinedThresh2 = std::max(30.0, combinedThresh2 - 20);
            combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
            std::cout << "Combined Canny thresholds decreased: " << combinedThresh1 << ", " << combinedThresh2 << std::endl;
        }
        else {
            cannyThresh1 = std::max(10.0, cannyThresh1 - 10);
            cannyThresh2 = std::max(30.0, cannyThresh2 - 20);
            cannyDetector = CannyEdgeDetector(cannyThresh1, cannyThresh2);
            std::cout << "Canny thresholds decreased: " << cannyThresh1 << ", " << cannyThresh2 << std::endl;
        }
    }

    // Изменение параметров дилатации/эрозии для Combined детектора
    if (useCombinedDetector) {
        if (key == 'd') {
            dilateSize = std::min(10, dilateSize + 1);
            combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
            std::cout << "Dilation size increased: " << dilateSize << std::endl;
        }
        if (key == 'D') {
            dilateSize = std::max(1, dilateSize - 1);
            combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
            std::cout << "Dilation size decreased: " << dilateSize << std::endl;
        }
        if (key == 'e') {
            erodeSize = std::min(10, erodeSize + 1);
            combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
            std::cout << "Erosion size increased: " << erodeSize << std::endl;
        }
        if (key == 'E') {
            erodeSize = std::max(1, erodeSize - 1);
            combinedDetector = CombinedEdgeDetector(combinedThresh1, combinedThresh2, dilateSize, erodeSize);
            std::cout << "Erosion size decreased: " << erodeSize << std::endl;
        }
    }

    // Сохранение кадра
    if (key == 's' || key == 'S') {
        std::string filename = "frame_" + std::to_string(time(nullptr)) + ".jpg";
        webcam.saveCurrentFrame(filename);
    }
}

int main() {
    setlocale(LC_ALL, "Russian");

    try {
        std::cout << "\n=============================================\n";
        std::cout << "       Edge Detection with Object Tracking\n";
        std::cout << "       (Using WebcamViewer)\n";
        std::cout << "=============================================\n";
        std::cout << "Controls:\n";
        std::cout << "  [1/2] - Switch detector (Canny/Combined)\n";
        std::cout << "  [+/-] - Adjust Canny thresholds\n";
        std::cout << "  [c/C] - Toggle display mode (overlay/edges only)\n";
        std::cout << "  [d/D] - Increase/decrease dilation (Combined)\n";
        std::cout << "  [e/E] - Increase/decrease erosion (Combined)\n";
        std::cout << "  [t/T] - Enable/disable object tracking\n";
        std::cout << "  [f/F] - Toggle face tracking mode\n";
        std::cout << "  [0]   - Reset object tracker\n";
        std::cout << "  [r/R] - Reset detector parameters\n";
        std::cout << "  [s/S] - Save current frame\n";
        std::cout << "  [SPACE] - Pause/Resume\n";
        std::cout << "  [ESC/Q] - Exit\n";
        std::cout << "=============================================\n\n";

        // === Создание детекторов ===
        CannyEdgeDetector cannyDetector(50.0, 150.0);
        CombinedEdgeDetector combinedDetector(50.0, 150.0, 2, 2);
        ObjectTracker objectTracker;

        // Параметры по умолчанию
        bool useCombinedDetector = true;
        bool showOnlyEdges = false;
        bool objectTrackingEnabled = false;
        bool faceTrackingMode = true;

        double cannyThresh1 = 50.0, cannyThresh2 = 150.0;
        double combinedThresh1 = 50.0, combinedThresh2 = 150.0;
        int dilateSize = 2, erodeSize = 2;

        // === Создание WebcamViewer ===
        WebcamViewer webcam;

        objectTracker.setFaceTrackingMode(faceTrackingMode);

        if (!webcam.initialize(0)) {
            std::cerr << "Error: Could not open camera!" << std::endl;
            return -1;
        }

        // === Для измерения FPS ===
        auto lastTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        float currentFPS = 0.0f;

        // === Функция обработки кадра ===
        auto frameProcessor = [&](cv::Mat& frame) {
            // Подсчет FPS
            frameCount++;
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);

            if (elapsedTime.count() >= 1000) {
                currentFPS = frameCount * 1000.0f / elapsedTime.count();
                frameCount = 0;
                lastTime = currentTime;
            }

            // Обработка кадра с сохранением всей оригинальной логики
            processFrame(frame, cannyDetector, combinedDetector, objectTracker,
                useCombinedDetector, showOnlyEdges, objectTrackingEnabled,
                faceTrackingMode, currentFPS);
            };

        // === Функция обработки клавиш ===
        auto keyProcessor = [&](int key) {
            processKey(key, useCombinedDetector, showOnlyEdges, objectTrackingEnabled,
                faceTrackingMode, cannyDetector, combinedDetector, objectTracker,
                cannyThresh1, cannyThresh2, combinedThresh1, combinedThresh2,
                dilateSize, erodeSize, webcam);
            };

        // === Запуск WebcamViewer с обработчиками ===
        webcam.run(frameProcessor, keyProcessor);

        std::cout << "\n=============================================\n";
        std::cout << "           Program finished\n";
        std::cout << "=============================================\n";
    }
    catch (const std::exception& e) {
        std::cerr << "\nCritical error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
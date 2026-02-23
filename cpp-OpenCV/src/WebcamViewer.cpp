#include "WebcamViewer.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace cv;
using namespace std;

WebcamViewer::WebcamViewer() {
    // Пустой конструктор
}

WebcamViewer::WebcamViewer(int cameraIndex) {
    initialize(cameraIndex);
}

WebcamViewer::~WebcamViewer() {
    stop();
}

bool WebcamViewer::initialize(int cameraIndex) {
    // Освобождаем предыдущий захват
    if (m_cap.isOpened()) {
        m_cap.release();
    }

    // Открываем камеру
    m_cap.open(cameraIndex);

    if (!m_cap.isOpened()) {
        cerr << "Error: Could not open camera!" << endl;
        return false;
    }

    // Устанавливаем стандартные параметры
    setResolution(1980, 1080);
    setFPS(30);

    // Отключаем автофокус для скорости
    m_cap.set(CAP_PROP_AUTOFOCUS, 0);
    m_cap.set(CAP_PROP_AUTO_EXPOSURE, 1);

    m_isRunning = true;
    m_paused = false;
    m_frameCount = 0;

    cout << "Webcam initialized: "
        << m_cap.get(CAP_PROP_FRAME_WIDTH) << "x"
        << m_cap.get(CAP_PROP_FRAME_HEIGHT)
        << " @ " << m_cap.get(CAP_PROP_FPS) << " FPS" << endl;

    return true;
}

void WebcamViewer::setResolution(int width, int height) {
    if (m_cap.isOpened()) {
        m_cap.set(CAP_PROP_FRAME_WIDTH, width);
        m_cap.set(CAP_PROP_FRAME_HEIGHT, height);
    }
}

void WebcamViewer::setFPS(int fps) {
    if (m_cap.isOpened()) {
        m_cap.set(CAP_PROP_FPS, fps);
    }
}

void WebcamViewer::run(FrameProcessor frameProcessor, KeyProcessor keyProcessor) {
    if (!m_cap.isOpened()) {
        cerr << "Error: Camera not initialized!" << endl;
        return;
    }

    // Для измерения FPS
    auto lastTime = chrono::high_resolution_clock::now();
    int frames = 0;
    float fps = 0.0f;

    Mat frame;
    m_isRunning = true;

    // Создаем окно (как в оригинальном main.cpp)
    namedWindow("Object Following", WINDOW_NORMAL);

    while (m_isRunning) {
        if (!m_cap.read(frame)) {
            cerr << "Failed to grab frame!" << endl;
            break;
        }

        m_currentFrame = frame.clone();
        m_frameCount++;

        // Подсчет FPS
        frames++;
        auto currentTime = chrono::high_resolution_clock::now();
        auto elapsedTime = chrono::duration_cast<chrono::milliseconds>(currentTime - lastTime);

        if (elapsedTime.count() >= 1000) {
            fps = frames * 1000.0f / elapsedTime.count();
            frames = 0;
            lastTime = currentTime;
        }

        // Если не на паузе, обрабатываем кадр
        if (!m_paused) {
            // Вызываем обработчик кадра, если он установлен
            if (frameProcessor) {
                frameProcessor(frame);
            }
        }
        else {
            // Если на паузе, показываем сообщение
            putText(frame, "[PAUSED]", Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        }

        // Показываем кадр
        imshow("Object Following", frame);

        // Обработка клавиш
        int key = waitKey(1);

        // Проверка на выход
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }

        // Обработка паузы
        if (key == ' ') {
            m_paused = !m_paused;
            cout << (m_paused ? "PAUSED" : "RESUMED") << endl;
        }

        // Вызываем обработчик клавиш, если он установлен
        if (keyProcessor && key != -1) {
            keyProcessor(key);
        }

        // Проверка, закрыто ли окно
        if (getWindowProperty("Object Following", WND_PROP_VISIBLE) < 1) {
            cout << "Window closed. Stopping..." << endl;
            break;
        }
    }

    stop();
}

void WebcamViewer::saveCurrentFrame(const std::string& filename) {
    if (!m_currentFrame.empty()) {
        imwrite(filename, m_currentFrame);
        cout << "Frame saved to: " << filename << endl;
    }
}

void WebcamViewer::stop() {
    m_isRunning = false;
    if (m_cap.isOpened()) {
        m_cap.release();
    }
    destroyAllWindows();
    cout << "Webcam stopped." << endl;
}
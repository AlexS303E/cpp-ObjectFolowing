#pragma once

#include <opencv2/opencv.hpp>
#include <functional>
#include <string>
#include <memory>

// Типы функций для обработки кадра и клавиш
using FrameProcessor = std::function<void(cv::Mat& frame)>;
using KeyProcessor = std::function<void(int key)>;

class WebcamViewer {
public:
    WebcamViewer();
    WebcamViewer(int cameraIndex);
    ~WebcamViewer();

    // Инициализация камеры
    bool initialize(int cameraIndex = 0);

    // Запуск основного цикла с обработчиками
    void run(FrameProcessor frameProcessor = nullptr, KeyProcessor keyProcessor = nullptr);

    // Остановка
    void stop();

    // Настройки
    void setResolution(int width, int height);
    void setFPS(int fps);

    // Сохранение кадра
    void saveCurrentFrame(const std::string& filename);

    // Проверка состояния
    bool isRunning() const { return m_isRunning; }
    bool isCameraOpened() const { return m_cap.isOpened(); }

    // Получение текущего кадра
    cv::Mat getCurrentFrame() { return m_currentFrame; }

    // Управление паузой
    void pause() { m_paused = true; }
    void resume() { m_paused = false; }
    bool isPaused() const { return m_paused; }

private:
    cv::VideoCapture m_cap;
    cv::Mat m_currentFrame;
    bool m_isRunning = false;
    bool m_paused = false;
    int m_frameCount = 0;
};
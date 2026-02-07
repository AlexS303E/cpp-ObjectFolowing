#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class WebcamViewer {
public:
    WebcamViewer();
    WebcamViewer(int cameraIndex);
    ~WebcamViewer();

    bool initialize(int cameraIndex = 0);
    void run();
    void stop();

    void setResolution(int width, int height);
    void setFPS(int fps);
    void saveCurrentFrame(const std::string& filename);

    bool isRunning() const { return m_isRunning; }

private:
    cv::VideoCapture m_cap;
    bool m_isRunning = false;
    int m_frameCount = 0;

    void processFrame(cv::Mat& frame);
    void handleKeyPress(int key);
};
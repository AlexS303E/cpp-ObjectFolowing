#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

// Включите заголовочные файлы TensorFlow Lite
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

class TFLiteDetector {
public:
    TFLiteDetector(const std::string& modelPath,
        const std::string& labelsPath,
        int inputWidth = 320,
        int inputHeight = 320);
    ~TFLiteDetector();

    bool initialize();
    void detectAndDraw(cv::Mat& frame);

private:
    void loadLabels(const std::string& filename);
    void preprocess(const cv::Mat& input, cv::Mat& output);
    void postprocess(cv::Mat& frame, const float* outputData);
    void drawBox(const cv::Rect& box, const std::string& label, cv::Mat& frame);

    std::string modelPath_;
    std::string labelsPath_;
    int inputWidth_;
    int inputHeight_;

    std::vector<std::string> labels_;
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;

    // Настройки делегата для ускорения
    bool useGPUDelegate_ = true; // Попробуйте включить для AMD
    std::unique_ptr<tflite::TfLiteDelegate, void(*)(tflite::TfLiteDelegate*)> delegate_{ nullptr, nullptr };
};
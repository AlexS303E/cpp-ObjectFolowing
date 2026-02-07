#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <filesystem>
#include "global.cpp"


bool IsFileExist(const std::string& file);


class YOLOv3Detector {
public:
    YOLOv3Detector(float confThreshold = 0.5f, float nmsThresh = 0.4f, int width = 416, int height = 416);

    void loadClasses(const std::string& filename);

    void generateColors();

    void detectAndDraw(cv::Mat& frame);

    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs);

    void drawPred(int classId, float conf, const cv::Rect& box, cv::Mat& frame);

private:
    cv::dnn::Net net;
    std::vector<std::string> classNames;
    float confidenceThreshold;
    float nmsThreshold;
    std::vector<cv::Scalar> colors;
    int inputWidth;
    int inputHeight;
};


class YOLOv8Detector {
public:
    YOLOv8Detector(float confThreshold = 0.5f, float nmsThresh = 0.45f,
        int width = 640, int height = 640);

    void detectAndDraw(cv::Mat& frame);

private:
    cv::dnn::Net net;
    std::vector<std::string> classNames;
    float confidenceThreshold;
    float nmsThreshold;
    int inputWidth;
    int inputHeight;
    std::vector<cv::Scalar> colors;

    void loadClassNames(const std::string& filename);
    void generateColors();
    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outputs);
    void drawBox(int classId, float conf, const cv::Rect& box, cv::Mat& frame);
};


class YOLOv26Detector {
public:
    YOLOv26Detector(float confThreshold = 0.5f, float nmsThresh = 0.45f, int width = 640, int height = 640);

    void loadClassNames(const std::string& filename);

    void generateColors();

    void detectAndDraw(cv::Mat& frame);

    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outputs);

    void drawBox(int classId, float conf, const cv::Rect& box, cv::Mat& frame);

private:
    cv::dnn::Net net;
    std::vector<std::string> classNames;
    float confidenceThreshold;
    float nmsThreshold;
    int inputWidth;
    int inputHeight;
    std::vector<cv::Scalar> colors;
};
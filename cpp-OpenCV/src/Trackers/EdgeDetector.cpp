#include "EdgeDetector.h"

void CannyEdgeDetector::detectAndDraw(cv::Mat& frame) {
    cv::Mat gray, blurred, edges;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);

    cv::Canny(blurred, edges,
        threshold1, threshold2,
        apertureSize, useL2Gradient);

    // Белые границы на чёрном фоне
    cv::Mat result;
    cv::cvtColor(edges, result, cv::COLOR_GRAY2BGR);

    frame = result.clone();

    cv::putText(frame, "Edge map", cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2);
}

void CannyEdgeDetector::detectOnlyEdges(cv::Mat& frame) {
    cv::Mat gray, blurred, edges;

    // Convert to grayscale
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Blur to reduce noise
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);

    // Canny edge detection
    cv::Canny(blurred, edges, threshold1, threshold2,
        apertureSize, useL2Gradient);

    // Convert to BGR for display
    cv::cvtColor(edges, frame, cv::COLOR_GRAY2BGR);
}

void CombinedEdgeDetector::detectAndDraw(cv::Mat& frame) {
    cv::Mat gray, edges, dilated, filled, eroded, result;

    // 1. Преобразование в черно-белое
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 2. Выделение контуров методом Канни
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);
    cv::Canny(gray, edges, cannyThreshold1, cannyThreshold2, 3);

    // 3. Дилатация (расширение) границ
    cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1));
    cv::dilate(edges, dilated, dilateKernel);

    // 4. Заполнение областей внутри границ (морфологическое закрытие + заливка)
    cv::Mat closed;
    cv::morphologyEx(dilated, closed, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));

    // Заливка фона для получения маски внутренних областей
    cv::Mat mask = cv::Mat::zeros(closed.rows + 2, closed.cols + 2, CV_8UC1);
    cv::floodFill(closed, mask, cv::Point(0, 0), cv::Scalar(255));
    cv::bitwise_not(mask(cv::Rect(1, 1, frame.cols, frame.rows)), filled);

    // 5. Эрозия заполненного изображения
    cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1));
    cv::erode(filled, eroded, erodeKernel);

    // 6. Вычитание: filled - eroded (внешние границы)
    cv::subtract(filled, eroded, result);

    // Преобразование результата в цветное изображение для наложения
    cv::Mat resultColor;
    cv::cvtColor(result, resultColor, cv::COLOR_GRAY2BGR);

    // Наложение границ на оригинальное изображение (полупрозрачное)
    cv::addWeighted(frame, 0.7, resultColor, 0.3, 0, frame);

    // Добавление информационного текста - используем простой английский текст
    cv::putText(frame, "Combined method (Canny + morphology)", cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    std::string cannyText = "Canny thresholds: " + std::to_string((int)cannyThreshold1) +
        ", " + std::to_string((int)cannyThreshold2);
    cv::putText(frame, cannyText, cv::Point(10, 60),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 255), 1);

    std::string morphText = "Dilation: " + std::to_string(dilationSize) +
        ", Erosion: " + std::to_string(erosionSize);
    cv::putText(frame, morphText, cv::Point(10, 80),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 255), 1);
}

void CombinedEdgeDetector::detectOnlyEdges(cv::Mat& frame) {
    cv::Mat gray, edges, dilated, filled, eroded, result;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);
    cv::Canny(gray, edges, cannyThreshold1, cannyThreshold2, 3);

    cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1));
    cv::dilate(edges, dilated, dilateKernel);

    cv::Mat closed;
    cv::morphologyEx(dilated, closed, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));

    cv::Mat mask = cv::Mat::zeros(closed.rows + 2, closed.cols + 2, CV_8UC1);
    cv::floodFill(closed, mask, cv::Point(0, 0), cv::Scalar(255));
    cv::bitwise_not(mask(cv::Rect(1, 1, frame.cols, frame.rows)), filled);

    cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1));
    cv::erode(filled, eroded, erodeKernel);

    cv::subtract(filled, eroded, result);
    cv::cvtColor(result, frame, cv::COLOR_GRAY2BGR);
}
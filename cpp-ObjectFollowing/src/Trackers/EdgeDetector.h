#pragma once

#include <opencv2/opencv.hpp>

class CannyEdgeDetector {
public:
    CannyEdgeDetector(double thresh1 = 100.0, double thresh2 = 200.0,
        int aperture = 3, bool useL2 = false)
        : threshold1(thresh1), threshold2(thresh2),
        apertureSize(aperture), useL2Gradient(useL2) {
    }

    void detectAndDraw(cv::Mat& frame);
    void detectOnlyEdges(cv::Mat& frame);

private:
    double threshold1;
    double threshold2;
    int apertureSize;
    bool useL2Gradient;
};


class CombinedEdgeDetector {
public:
    CombinedEdgeDetector(double thresh1 = 50.0, double thresh2 = 150.0, 
                         int dilateSize = 2, int erodeSize = 2)
        : cannyThreshold1(thresh1), cannyThreshold2(thresh2),
          dilationSize(dilateSize), erosionSize(erodeSize) {}

    // Комбинированный метод по статье https://engjournal.bmstu.ru/articles/920/920.pdf
    // 1. Выделение контуров методом Канни
    // 2. Дилатация выделенных границ
    // 3. Заполнение областей внутри границ
    // 4. Эрозия полученного изображения
    // 5. Вычитание результатов шагов 3 и 4
    void detectAndDraw(cv::Mat& frame);
    void detectOnlyEdges(cv::Mat& frame);

private:
    double cannyThreshold1;
    double cannyThreshold2;
    int dilationSize;
    int erosionSize;
};
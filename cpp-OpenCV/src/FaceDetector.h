#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>

class FaceDetector {
public:
    FaceDetector();

    // ����������� ��� �� �����
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame);

    // ��������� ��������������� ������ ���
    void drawFaces(cv::Mat& frame, const std::vector<cv::Rect>& faces);

    // ��������� ������ ������ �������� ����
    cv::Point2f getLargestFaceCenter(const std::vector<cv::Rect>& faces);

    // ��������� ������ �������� ����
    cv::Rect getLargestFace(const std::vector<cv::Rect>& faces);

private:
    cv::CascadeClassifier faceCascade;
    cv::CascadeClassifier profileFaceCascade;

    // ��������� ���������
    double scaleFactor;
    int minNeighbors;
    cv::Size minSize;
    cv::Size maxSize;

    // ����� ��� ���������
    cv::Scalar faceColor;
    cv::Scalar eyeColor;
};
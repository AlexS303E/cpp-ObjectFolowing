#include "FaceDetector.h"
#include <iostream>

FaceDetector::FaceDetector() {
    // ��������� �� ���������
    scaleFactor = 1.1;
    minNeighbors = 3;
    minSize = cv::Size(30, 30);
    maxSize = cv::Size(300, 300);

    faceColor = cv::Scalar(0, 255, 0);  // ������� ��� ���
    eyeColor = cv::Scalar(255, 0, 0);   // ����� ��� ����

    try {
        profileFaceCascade.load(FACE_CASCADE_PROFILE);
        faceCascade.load(FACE_CASCADE_FRONTAL);
    }
    catch (...) {
        std::cerr << "Cant Load Cascade\n";
    }
}

std::vector<cv::Rect> FaceDetector::detectFaces(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;

    if (frame.empty()) {
        return faces;
    }

    // ����������� � ������� ������ ��� ������� �����������
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // ����������� ������� � ��������
    cv::equalizeHist(gray, gray);

    // ����������� ����������� ���
    faceCascade.detectMultiScale(
        gray,
        faces,
        scaleFactor,
        minNeighbors,
        0,  // ����� (0 - ������������ ������ ������)
        minSize,
        maxSize
    );

    // ����� �������� ���������� ���������� ����, ���� ����������� ���
    if (faces.empty() && profileFaceCascade.empty() == false) {
        std::vector<cv::Rect> profileFaces;
        profileFaceCascade.detectMultiScale(
            gray,
            profileFaces,
            scaleFactor,
            minNeighbors,
            0,
            minSize,
            maxSize
        );

        // ��������� ���������� ���� � ����� ������
        faces.insert(faces.end(), profileFaces.begin(), profileFaces.end());
    }

    return faces;
}

void FaceDetector::drawFaces(cv::Mat& frame, const std::vector<cv::Rect>& faces) {
    if (faces.empty()) {
        // ���� ��� �� ����������, ������ ���������
        cv::putText(frame, "No faces detected", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        return;
    }

    // ������ �������������� ������ ���� ���
    for (size_t i = 0; i < faces.size(); ++i) {
        const cv::Rect& face = faces[i];

        // ������� �������������� ������� �� ������� ����
        int thickness = std::max(2, face.width / 100);

        // ������ �������������
        cv::rectangle(frame, face, faceColor, thickness);

        // ����������� ����� ����
        std::string label = "Face " + std::to_string(i + 1);
        cv::putText(frame, label,
            cv::Point(face.x, face.y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, faceColor, 1);

        // ������ ����� � ������ ����
        cv::Point center(face.x + face.width / 2, face.y + face.height / 2);
        int crossSize = face.width / 10;

        cv::line(frame,
            cv::Point(center.x - crossSize, center.y),
            cv::Point(center.x + crossSize, center.y),
            cv::Scalar(0, 0, 255), 2);  // ������� �������������� �����

        cv::line(frame,
            cv::Point(center.x, center.y - crossSize),
            cv::Point(center.x, center.y + crossSize),
            cv::Scalar(0, 0, 255), 2);  // ������� ������������ �����

        // ������ ����� ������ ���� (��������� ������������)
        int eyeWidth = face.width / 4;
        int eyeHeight = face.height / 8;
        int eyeY = face.y + face.height / 3;

        cv::Rect leftEye(face.x + face.width / 4 - eyeWidth / 2, eyeY, eyeWidth, eyeHeight);
        cv::Rect rightEye(face.x + 3 * face.width / 4 - eyeWidth / 2, eyeY, eyeWidth, eyeHeight);

        cv::rectangle(frame, leftEye, eyeColor, 1);
        cv::rectangle(frame, rightEye, eyeColor, 1);

        // ������������ � ���������� ����������
        std::string info = "W: " + std::to_string(face.width) +
            " H: " + std::to_string(face.height);
        cv::putText(frame, info,
            cv::Point(face.x, face.y + face.height + 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
    }

    // ���������� ����� ����������
    std::string stats = "Faces detected: " + std::to_string(faces.size());
    cv::putText(frame, stats, cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
}

cv::Point2f FaceDetector::getLargestFaceCenter(const std::vector<cv::Rect>& faces) {
    if (faces.empty()) {
        return cv::Point2f(-1, -1);  // ��������� ���������� ���
    }

    cv::Rect largestFace = getLargestFace(faces);
    return cv::Point2f(
        largestFace.x + largestFace.width / 2.0f,
        largestFace.y + largestFace.height / 2.0f
    );
}

cv::Rect FaceDetector::getLargestFace(const std::vector<cv::Rect>& faces) {
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
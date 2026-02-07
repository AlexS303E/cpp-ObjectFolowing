#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>

// ��������� ��� �������� ���������� � ����������� �������
struct TrackedObject {
    cv::Rect boundingBox;    // �������������� �����
    cv::Point2f center;      // ����� �������
    int id;                  // ���������� �������������
    int age;                 // ������� ������� � ������
    bool lost;               // ������� �� ������
    std::string type;        // ��� ������� ("face", "object")

    TrackedObject() : id(1), age(0), lost(false), type("object") {
        boundingBox = cv::Rect(0, 0, 0, 0);
        center = cv::Point2f(0, 0);
    }
};

// ������� ������ ��������
class ObjectTracker {
public:
    ObjectTracker();
    ~ObjectTracker();

    // ������������� �������
    bool initialize(const cv::Mat& frame);

    // ������������� ��� �������� �� �����
    bool initializeForFaceTracking(const cv::Mat& frame);

    // ���������� ������� � ����� ������
    bool update(const cv::Mat& frame);

    // ���������� � ������������ ���
    bool updateWithFaceDetection(const cv::Mat& frame);

    // ��������� ���������� � ����������� �������
    TrackedObject getTrackedObject() const;

    // ��������� ���������� �� ������������ �� �����
    void drawTrackingInfo(cv::Mat& frame) const;

    // ��������, ������� �� ������
    bool isInitialized() const;

    // ����� �������
    void reset();

    // ��������� ������ ������������ ����
    void setFaceTrackingMode(bool enable);

    // ��������� ������� ������ ����
    bool isFaceTrackingMode() const;

private:
    // ����������� ������
    TrackedObject trackedObject;

    // ���� �������������
    bool initialized;

    // ����� ������������ ����
    bool faceTrackingMode;

    // ������� ��� ���������� �������
    int updateCounter;

    // �������� � �������� ��� ������������
    float predictionTime;

    // �������������� ������� ������
    float frameRate;

    // ������� ��������� ��� �����������
    std::vector<cv::Point2f> positionHistory;
    static const int maxHistorySize = 10;

    // ������� �������� ��� ������������
    struct VelocitySample {
        cv::Point2f velocity;  // �������� � �������� � �������
        std::chrono::steady_clock::time_point timestamp;
    };
    std::vector<VelocitySample> velocityHistory;
    cv::Point2f currentVelocity;  // ������� �������� (����/���)

    // ��� ���������� ��������
    cv::Point2f previousPosition;
    std::chrono::steady_clock::time_point previousTime;
    bool hasPreviousPosition;

    // ������ ��� ����������� ������� �� ��������
    cv::Rect detectObjectInCenter(const cv::Mat& frame);
    cv::Rect findLargestContour(const cv::Mat& frame, const cv::Rect& searchArea);

    // ������ ��� ����������� ���
    cv::Rect detectFaceInCenter(const cv::Mat& frame);
    cv::Rect findLargestFace(const cv::Mat& frame, const cv::Rect& searchArea);

    // ��������������� ������
    void updateBoundingBox(const cv::Rect& newBox);
    void updatePositionHistory(const cv::Point2f& newPosition);
    cv::Point2f getSmoothedPosition() const;

    // ������ ��� ������������ ��������
    void updateVelocity(const cv::Point2f& newPosition);
    cv::Point2f getPredictedPosition() const;
    cv::Point2f getClosestPointOnRect(const cv::Rect& rect, const cv::Point2f& point) const;
    cv::Point2f getPointOnCircle(const cv::Point2f& circleCenter,
        const cv::Point2f& targetPoint,
        float radius) const;

    // ������ ����� ��� ����������� ���
    cv::CascadeClassifier faceCascade;
    bool loadFaceCascade();
};
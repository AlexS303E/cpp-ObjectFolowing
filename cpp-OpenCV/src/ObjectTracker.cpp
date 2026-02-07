#include "ObjectTracker.h"
#include <iostream>
#include <limits>
#include <algorithm>

using namespace std;
using namespace cv;

ObjectTracker::ObjectTracker() : initialized(false), updateCounter(0),
faceTrackingMode(false), predictionTime(1.0f), currentVelocity(0, 0),
hasPreviousPosition(false), frameRate(30.0f) {
    trackedObject = TrackedObject();
    positionHistory.reserve(maxHistorySize);
}

ObjectTracker::~ObjectTracker() {
    reset();
}

bool ObjectTracker::loadFaceCascade() {
    // ���� ��� ������ ������� (������������ ���������� ������������ �����)
    std::vector<std::string> possiblePaths = {
        "haarcascade_frontalface_default.xml",
        "./haarcascade_frontalface_default.xml",
        "../haarcascade_frontalface_default.xml",
        "../../haarcascade_frontalface_default.xml",
        "models/haarcascade_frontalface_default.xml",
        "./models/haarcascade_frontalface_default.xml",
        "../models/haarcascade_frontalface_default.xml",
        "../../models/haarcascade_frontalface_default.xml",
        "haarcascades/haarcascade_frontalface_default.xml",
        "./haarcascades/haarcascade_frontalface_default.xml",
        "../haarcascades/haarcascade_frontalface_default.xml",
        "../../haarcascades/haarcascade_frontalface_default.xml",
        "data/haarcascade_frontalface_default.xml",
        "C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    };

    // �������� ��������� ������
    for (const auto& path : possiblePaths) {
        std::cout << "������ ��������� ������ ��: " << path << std::endl;
        if (faceCascade.load(path)) {
            std::cout << "������ ����� ������� �������� ��: " << path << std::endl;
            return true;
        }
    }

    // ���� �� �����, ������� samples::findFile
    try {
        std::string cascadePath = cv::samples::findFile("haarcascade_frontalface_default.xml");
        std::cout << "������ ��������� ����� samples::findFile: " << cascadePath << std::endl;
        if (faceCascade.load(cascadePath)) {
            std::cout << "������ ����� �������� ����� samples::findFile: " << cascadePath << std::endl;
            return true;
        }
    }
    catch (const cv::Exception& e) {
        std::cerr << "������ samples::findFile: " << e.what() << std::endl;
    }

    std::cerr << "������: �� ������� ��������� ������ �����!" << std::endl;
    std::cerr << "���� haarcascade_frontalface_default.xml ������ ���������� � ����� �� ��������� ����������:" << std::endl;
    std::cerr << "1. � ��� �� ����������, ��� � ����������� ����" << std::endl;
    std::cerr << "2. � ������������� models/" << std::endl;
    std::cerr << "3. � ������������� haarcascades/" << std::endl;

    return false;
}

void ObjectTracker::setFaceTrackingMode(bool enable) {
    faceTrackingMode = enable;
    trackedObject.type = enable ? "face" : "object";

    if (enable) {
        if (!loadFaceCascade()) {
            cerr << "Failed to load face cascade, disabling face tracking mode" << endl;
            faceTrackingMode = false;
            trackedObject.type = "object";
        }
    }

    // ���������� ������ ��� ����� ������
    reset();
}

bool ObjectTracker::isFaceTrackingMode() const {
    return faceTrackingMode;
}

bool ObjectTracker::initialize(const cv::Mat& frame) {
    if (frame.empty()) {
        std::cerr << "Frame is empty!" << std::endl;
        return false;
    }

    if (faceTrackingMode) {
        std::cout << "Initializing in FACE tracking mode..." << std::endl;
        return initializeForFaceTracking(frame);
    }

    std::cout << "Initializing in OBJECT tracking mode..." << std::endl;

    // ���������� ������� � ������ ����� ��� ������ �������
    Rect centerRegion = detectObjectInCenter(frame);

    if (centerRegion.area() > 0) {
        trackedObject.boundingBox = centerRegion;
        trackedObject.center = Point2f(
            centerRegion.x + centerRegion.width / 2.0f,
            centerRegion.y + centerRegion.height / 2.0f
        );

        initialized = true;
        trackedObject.age = 0;
        trackedObject.lost = false;
        updateCounter = 0;
        positionHistory.clear();
        velocityHistory.clear();

        cout << "Object tracker initialized with object at: ("
            << trackedObject.boundingBox.x << ", "
            << trackedObject.boundingBox.y << ")"
            << " size: " << trackedObject.boundingBox.width
            << "x" << trackedObject.boundingBox.height << endl;

        return true;
    }

    // ���� ������ �� ������, ������� ��������� ����� � ������ (������ ��� ������ �������)
    int centerX = frame.cols / 2;
    int centerY = frame.rows / 2;
    int boxSize = min(frame.cols, frame.rows) / 4;

    trackedObject.boundingBox = Rect(
        centerX - boxSize / 2,
        centerY - boxSize / 2,
        boxSize,
        boxSize
    );

    trackedObject.center = Point2f(
        trackedObject.boundingBox.x + trackedObject.boundingBox.width / 2.0f,
        trackedObject.boundingBox.y + trackedObject.boundingBox.height / 2.0f
    );

    initialized = true;
    trackedObject.age = 0;
    trackedObject.lost = false;
    updateCounter = 0;
    positionHistory.clear();
    velocityHistory.clear();

    cout << "Tracker initialized with default box at center" << endl;
    return true;
}

bool ObjectTracker::initializeForFaceTracking(const cv::Mat& frame) {
    if (frame.empty()) {
        std::cerr << "Frame is empty!" << std::endl;
        return false;
    }

    if (faceCascade.empty()) {
        std::cerr << "Face cascade is not loaded!" << std::endl;
        if (!loadFaceCascade()) {
            std::cerr << "Failed to load face cascade!" << std::endl;
            return false;
        }
    }

    // ���� ���� �� ���� �����, � �� ������ � ������
    std::vector<cv::Rect> faces;
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    // ��������� ��� ������� �����������
    faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));

    std::cout << "Found " << faces.size() << " faces in frame." << std::endl;

    if (!faces.empty()) {
        // �������� ����� ������� ����
        cv::Rect largestFace = faces[0];
        int maxArea = largestFace.width * largestFace.height;

        for (size_t i = 1; i < faces.size(); ++i) {
            int area = faces[i].width * faces[i].height;
            if (area > maxArea) {
                maxArea = area;
                largestFace = faces[i];
            }
        }

        trackedObject.boundingBox = largestFace;
        trackedObject.center = cv::Point2f(
            largestFace.x + largestFace.width / 2.0f,
            largestFace.y + largestFace.height / 2.0f
        );

        initialized = true;
        trackedObject.age = 0;
        trackedObject.lost = false;
        updateCounter = 0;
        positionHistory.clear();
        velocityHistory.clear();
        trackedObject.type = "face";

        std::cout << "Face tracker initialized with face at: ("
            << trackedObject.boundingBox.x << ", "
            << trackedObject.boundingBox.y << ")"
            << " size: " << trackedObject.boundingBox.width
            << "x" << trackedObject.boundingBox.height << std::endl;

        return true;
    }

    std::cout << "No face found in the frame!" << std::endl;

    // �� ������� ��������� ���� ��� ������ ����!
    // ������ ���������� false
    return false;
}

bool ObjectTracker::update(const cv::Mat& frame) {
    if (!initialized || frame.empty()) {
        return false;
    }

    if (faceTrackingMode) {
        return updateWithFaceDetection(frame);
    }

    updateCounter++;

    // ������ 2-� ���� ��������� ����� ������� (����� �� ����������� CPU)
    if (updateCounter % 2 == 0 || trackedObject.lost) {
        // ���������� ������� ������ ������ �������� ��������� �������
        int searchMargin = 50; // �������� ��� ������ ������
        Rect searchArea(
            max(0, trackedObject.boundingBox.x - searchMargin),
            max(0, trackedObject.boundingBox.y - searchMargin),
            min(frame.cols, trackedObject.boundingBox.width + 2 * searchMargin),
            min(frame.rows, trackedObject.boundingBox.height + 2 * searchMargin)
        );

        // ������������ ��������� ������� ��������� �����
        if (searchArea.x + searchArea.width > frame.cols) {
            searchArea.width = frame.cols - searchArea.x;
        }
        if (searchArea.y + searchArea.height > frame.rows) {
            searchArea.height = frame.rows - searchArea.y;
        }

        Rect newBox = findLargestContour(frame, searchArea);

        if (newBox.area() > 0) {
            // ������ ������, ��������� ���������
            updateBoundingBox(newBox);
            trackedObject.lost = false;
            trackedObject.age++;
            updatePositionHistory(trackedObject.center);
            updateVelocity(trackedObject.center);
        }
        else {
            // ������ �� ������ � ������� ������
            trackedObject.lost = true;

            // �������� ����� ������ � ������ �����
            if (updateCounter % 10 == 0) { // ������ 10-� ����
                Rect centerBox = detectObjectInCenter(frame);
                if (centerBox.area() > 0) {
                    updateBoundingBox(centerBox);
                    trackedObject.lost = false;
                    updatePositionHistory(trackedObject.center);
                    updateVelocity(trackedObject.center);
                }
            }
        }
    }
    else {
        // ������ ����������� ������� �������
        trackedObject.age++;
    }

    return !trackedObject.lost;
}

bool ObjectTracker::updateWithFaceDetection(const cv::Mat& frame) {
    if (!initialized || frame.empty() || faceCascade.empty()) {
        return false;
    }

    updateCounter++;

    // ���������� ������� ������ ������ �������� ��������� ����
    int searchMargin = 100; // ������ ��� ����, ��� ��� ������ ����� ��������� �������
    Rect searchArea(
        max(0, trackedObject.boundingBox.x - searchMargin),
        max(0, trackedObject.boundingBox.y - searchMargin),
        min(frame.cols, trackedObject.boundingBox.width + 2 * searchMargin),
        min(frame.rows, trackedObject.boundingBox.height + 2 * searchMargin)
    );

    // ������������ ��������� ������� ��������� �����
    if (searchArea.x + searchArea.width > frame.cols) {
        searchArea.width = frame.cols - searchArea.x;
    }
    if (searchArea.y + searchArea.height > frame.rows) {
        searchArea.height = frame.rows - searchArea.y;
    }

    Rect newFace = findLargestFace(frame, searchArea);

    if (newFace.area() > 0) {
        // ���� �������, ��������� ���������
        updateBoundingBox(newFace);
        trackedObject.lost = false;
        trackedObject.age++;
        updatePositionHistory(trackedObject.center);
        updateVelocity(trackedObject.center);
    }
    else {
        // ���� �� ������� � ������� ������
        trackedObject.lost = true;

        // �������� ����� ���� �� ���� �����
        if (updateCounter % 5 == 0) { // ������ 5-� ���� ���� ���� �� ���� �����
            vector<Rect> faces;
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            equalizeHist(gray, gray);

            faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

            if (!faces.empty()) {
                // ������� ����� ������� ����
                Rect largestFace = faces[0];
                int maxArea = largestFace.width * largestFace.height;

                for (size_t i = 1; i < faces.size(); ++i) {
                    int area = faces[i].width * faces[i].height;
                    if (area > maxArea) {
                        maxArea = area;
                        largestFace = faces[i];
                    }
                }

                updateBoundingBox(largestFace);
                trackedObject.lost = false;
                updatePositionHistory(trackedObject.center);
                updateVelocity(trackedObject.center);
            }
        }
    }

    return !trackedObject.lost;
}

cv::Rect ObjectTracker::detectObjectInCenter(const cv::Mat& frame) {
    // ���������� ����������� ������� ����� (20% �� ��������)
    int centerWidth = frame.cols * 0.2;
    int centerHeight = frame.rows * 0.2;
    int centerX = frame.cols / 2 - centerWidth / 2;
    int centerY = frame.rows / 2 - centerHeight / 2;

    Rect centerRegion(centerX, centerY, centerWidth, centerHeight);

    return findLargestContour(frame, centerRegion);
}

cv::Rect ObjectTracker::findLargestContour(const cv::Mat& frame, const cv::Rect& searchArea) {
    if (searchArea.area() <= 0) {
        return Rect();
    }

    // �������� ������� ������
    Mat roi = frame(searchArea);

    // ����������� � �������� ������
    Mat gray;
    cvtColor(roi, gray, COLOR_BGR2GRAY);

    // ��������� �������� ��� ���������� ����
    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 1.5);

    // ��������� ���������� ����� ��� ������� ��������� ��������
    Mat thresholded;
    adaptiveThreshold(blurred, thresholded, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY_INV, 11, 2);

    // ������� �������
    vector<vector<Point>> contours;
    findContours(thresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return Rect();
    }

    // ������� ����� ������� ������
    double maxArea = 0;
    int maxIdx = -1;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxIdx = i;
        }
    }

    if (maxIdx >= 0 && maxArea > 100) { // ����������� ������� 100 ��������
        // �������� �������������� �������������
        Rect rect = boundingRect(contours[maxIdx]);

        // ��������� ���������� � ������� ����� �����
        rect.x += searchArea.x;
        rect.y += searchArea.y;

        // ��������� ������������� �� 10% ��� ������� ����������� ����������
        int expandX = rect.width * 0.1;
        int expandY = rect.height * 0.1;

        rect.x = max(0, rect.x - expandX);
        rect.y = max(0, rect.y - expandY);
        rect.width = min(frame.cols - rect.x, rect.width + 2 * expandX);
        rect.height = min(frame.rows - rect.y, rect.height + 2 * expandY);

        return rect;
    }

    return Rect();
}

cv::Rect ObjectTracker::detectFaceInCenter(const cv::Mat& frame) {
    if (faceCascade.empty()) {
        return Rect();
    }

    // ���������� ����������� ������� ����� (30% �� ��������)
    int centerWidth = frame.cols * 0.3;
    int centerHeight = frame.rows * 0.3;
    int centerX = frame.cols / 2 - centerWidth / 2;
    int centerY = frame.rows / 2 - centerHeight / 2;

    Rect centerRegion(centerX, centerY, centerWidth, centerHeight);

    return findLargestFace(frame, centerRegion);
}

cv::Rect ObjectTracker::findLargestFace(const cv::Mat& frame, const cv::Rect& searchArea) {
    if (searchArea.area() <= 0 || faceCascade.empty()) {
        return Rect();
    }

    // �������� ������� ������
    Mat roi = frame(searchArea);
    Mat gray;
    cvtColor(roi, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    // ������������ ����
    vector<Rect> faces;
    faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

    if (faces.empty()) {
        return Rect();
    }

    // ������� ����� ������� ����
    Rect largestFace = faces[0];
    int maxArea = largestFace.width * largestFace.height;

    for (size_t i = 1; i < faces.size(); ++i) {
        int area = faces[i].width * faces[i].height;
        if (area > maxArea) {
            maxArea = area;
            largestFace = faces[i];
        }
    }

    // ��������� ���������� � ������� ����� �����
    largestFace.x += searchArea.x;
    largestFace.y += searchArea.y;

    return largestFace;
}

void ObjectTracker::updateBoundingBox(const cv::Rect& newBox) {
    // ������� ���������� ��������� (������ ������ ������)
    if (trackedObject.boundingBox.area() > 0) {
        // 70% ������ ���������, 30% ����� ��������� ��� ���������
        trackedObject.boundingBox.x = 0.7 * trackedObject.boundingBox.x + 0.3 * newBox.x;
        trackedObject.boundingBox.y = 0.7 * trackedObject.boundingBox.y + 0.3 * newBox.y;
        trackedObject.boundingBox.width = 0.7 * trackedObject.boundingBox.width + 0.3 * newBox.width;
        trackedObject.boundingBox.height = 0.7 * trackedObject.boundingBox.height + 0.3 * newBox.height;
    }
    else {
        trackedObject.boundingBox = newBox;
    }

    // ��������� ����� � ������ �����������
    trackedObject.center = Point2f(
        trackedObject.boundingBox.x + trackedObject.boundingBox.width / 2.0f,
        trackedObject.boundingBox.y + trackedObject.boundingBox.height / 2.0f
    );
}

void ObjectTracker::updatePositionHistory(const cv::Point2f& newPosition) {
    positionHistory.push_back(newPosition);

    if (positionHistory.size() > maxHistorySize) {
        positionHistory.erase(positionHistory.begin());
    }
}

void ObjectTracker::updateVelocity(const cv::Point2f& newPosition) {
    // ���� ��� ������ �����, ������ ��������� �������
    if (!hasPreviousPosition) {
        previousPosition = newPosition;
        previousTime = std::chrono::steady_clock::now();
        hasPreviousPosition = true;
        return;
    }

    auto currentTime = std::chrono::steady_clock::now();
    float deltaTime = std::chrono::duration<float>(currentTime - previousTime).count();

    // ����������� ������ ������� ��� ��������� ������� �� ����
    if (deltaTime < 0.001f) {
        return;
    }

    // ��������� �����������
    cv::Point2f displacement = newPosition - previousPosition;

    // ��������� �������� (�������� � �������)
    cv::Point2f velocity = displacement / deltaTime;

    // ���������� �������� (���������������� ���������� �������)
    // ����������� ����������� ������� �� ������ �������
    float alpha = 0.3f * std::min(deltaTime * 30.0f, 1.0f); // ���������� �����������
    currentVelocity = (1.0f - alpha) * currentVelocity + alpha * velocity;

    // ��������� ���������� ��������
    previousPosition = newPosition;
    previousTime = currentTime;

    // ��������� � ������� ��� �������
    VelocitySample sample;
    sample.velocity = currentVelocity;
    sample.timestamp = currentTime;
    velocityHistory.push_back(sample);

    // ������������ ������ �������
    if (velocityHistory.size() > 10) {
        velocityHistory.erase(velocityHistory.begin());
    }

    // ���������� ����� (����� ����������������)
    // std::cout << "Velocity: (" << currentVelocity.x << ", " << currentVelocity.y 
    //          << ") px/s, deltaTime: " << deltaTime << "s" << std::endl;
}

cv::Point2f ObjectTracker::getPredictedPosition() const {
    // ������������� ��������� ����� predictionTime ������
    cv::Point2f predicted = trackedObject.center + currentVelocity * predictionTime;

    // ���������� �����
    // std::cout << "Current: (" << trackedObject.center.x << ", " << trackedObject.center.y 
    //          << "), Predicted: (" << predicted.x << ", " << predicted.y << ")" << std::endl;

    return predicted;
}

cv::Point2f ObjectTracker::getSmoothedPosition() const {
    if (positionHistory.empty()) {
        return trackedObject.center;
    }

    Point2f sum(0, 0);
    for (const auto& pos : positionHistory) {
        sum += pos;
    }

    return Point2f(sum.x / positionHistory.size(), sum.y / positionHistory.size());
}

TrackedObject ObjectTracker::getTrackedObject() const {
    return trackedObject;
}

// ������� ��������� ����� �� ������� �������������� � �������� �����
cv::Point2f ObjectTracker::getClosestPointOnRect(const cv::Rect& rect, const cv::Point2f& point) const {
    // ������� ��������� ����� �� ������� ��������������
    float closestX = std::max((float)rect.x, std::min(point.x, (float)(rect.x + rect.width)));
    float closestY = std::max((float)rect.y, std::min(point.y, (float)(rect.y + rect.height)));

    // ���� ����� ������ ��������������, ���������� �� � ��������� �������
    if (closestX > rect.x && closestX < rect.x + rect.width &&
        closestY > rect.y && closestY < rect.y + rect.height) {
        // ����� ������, ������� ��������� �������
        float distToLeft = closestX - rect.x;
        float distToRight = (rect.x + rect.width) - closestX;
        float distToTop = closestY - rect.y;
        float distToBottom = (rect.y + rect.height) - closestY;

        float minDist = std::min({ distToLeft, distToRight, distToTop, distToBottom });

        if (minDist == distToLeft) closestX = rect.x;
        else if (minDist == distToRight) closestX = rect.x + rect.width;
        else if (minDist == distToTop) closestY = rect.y;
        else closestY = rect.y + rect.height;
    }

    return cv::Point2f(closestX, closestY);
}

// ������� ����� �� ���������� �� ����������� � ������ �����
cv::Point2f ObjectTracker::getPointOnCircle(const cv::Point2f& circleCenter,
    const cv::Point2f& targetPoint,
    float radius) const {
    // ��������� ������ �� ������ ����� � ������� �����
    cv::Point2f direction = targetPoint - circleCenter;

    // ����������� ������
    float distance = sqrt(direction.x * direction.x + direction.y * direction.y);
    if (distance > 0) {
        direction.x = direction.x / distance * radius;
        direction.y = direction.y / distance * radius;
    }

    // ���������� ����� �� ����������
    return cv::Point2f(circleCenter.x + direction.x, circleCenter.y + direction.y);
}

void ObjectTracker::drawTrackingInfo(cv::Mat& frame) const {
    if (!initialized) {
        return;
    }

    // Получаем предсказанную позицию через 1 секунду
    cv::Point2f predictedPosition = getPredictedPosition();

    // Ограничиваем предсказанную позицию рамками кадра
    predictedPosition.x = std::max(0.0f, std::min(predictedPosition.x, (float)frame.cols));
    predictedPosition.y = std::max(0.0f, std::min(predictedPosition.y, (float)frame.rows));

    if (faceTrackingMode) {
        // Рисуем рамку для лица (ЗЕЛЕНЫЙ цвет)
        if (trackedObject.lost) {
            // Если лицо потеряно - рисуем красную прерывистую рамку
            int lineType = LINE_8;
            int thickness = 3;
            Scalar color(0, 0, 255); // Красный

            // Рисуем пунктирную рамку
            for (int i = 0; i < trackedObject.boundingBox.width; i += 15) {
                line(frame,
                    Point(trackedObject.boundingBox.x + i, trackedObject.boundingBox.y),
                    Point(trackedObject.boundingBox.x + min(i + 8, trackedObject.boundingBox.width),
                        trackedObject.boundingBox.y),
                    color, thickness, lineType);
                line(frame,
                    Point(trackedObject.boundingBox.x + i,
                        trackedObject.boundingBox.y + trackedObject.boundingBox.height),
                    Point(trackedObject.boundingBox.x + min(i + 8, trackedObject.boundingBox.width),
                        trackedObject.boundingBox.y + trackedObject.boundingBox.height),
                    color, thickness, lineType);
            }

            for (int i = 0; i < trackedObject.boundingBox.height; i += 15) {
                line(frame,
                    Point(trackedObject.boundingBox.x, trackedObject.boundingBox.y + i),
                    Point(trackedObject.boundingBox.x,
                        trackedObject.boundingBox.y + min(i + 8, trackedObject.boundingBox.height)),
                    color, thickness, lineType);
                line(frame,
                    Point(trackedObject.boundingBox.x + trackedObject.boundingBox.width,
                        trackedObject.boundingBox.y + i),
                    Point(trackedObject.boundingBox.x + trackedObject.boundingBox.width,
                        trackedObject.boundingBox.y + min(i + 8, trackedObject.boundingBox.height)),
                    color, thickness, lineType);
            }

            cv::putText(frame, "FACE LOST!",
                Point(trackedObject.boundingBox.x, trackedObject.boundingBox.y - 20),
                FONT_HERSHEY_SIMPLEX, 0.8, color, 3);
        }
        else {
            // Если лицо найдено - рисуем ЗЕЛЕНУЮ сплошную рамку
            rectangle(frame, trackedObject.boundingBox, Scalar(0, 255, 0), 3);

            // Рисуем дополнительный внутренний контур
            rectangle(frame, trackedObject.boundingBox, Scalar(0, 200, 0), 1);

            // Подпись "FACE" (зеленым цветом)
            cv::putText(frame, "FACE",
                Point(trackedObject.boundingBox.x, trackedObject.boundingBox.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

            // Рисуем ЗЕЛЕНУЮ ОКРУЖНОСТЬ в предсказанной позиции через 1 секунду
            int circleRadius = 26;
            // Окружность (полый круг)
            circle(frame, predictedPosition, circleRadius, Scalar(0, 255, 0), 2);

            // Определяем, находится ли круг внутри квадрата
            bool circleInsideRect = predictedPosition.x >= trackedObject.boundingBox.x &&
                predictedPosition.x <= trackedObject.boundingBox.x + trackedObject.boundingBox.width &&
                predictedPosition.y >= trackedObject.boundingBox.y &&
                predictedPosition.y <= trackedObject.boundingBox.y + trackedObject.boundingBox.height;

            if (circleInsideRect) {
                // Если круг внутри квадрата - рисуем линию из центра квадрата до границы окружности
                cv::Point2f rectCenter = trackedObject.center;

                // Вычисляем вектор от центра квадрата к центру круга
                cv::Point2f direction = predictedPosition - rectCenter;
                float distance = sqrt(direction.x * direction.x + direction.y * direction.y);

                if (distance > 0) {
                    // Нормализуем вектор
                    direction.x /= distance;
                    direction.y /= distance;

                    // Вычисляем точку на границе окружности (вдоль того же направления)
                    cv::Point2f circleBoundary = predictedPosition - direction * circleRadius;

                    // Рисуем ЗЕЛЕНУЮ ЛИНИЮ от центра квадрата до границы окружности
                    line(frame, rectCenter, circleBoundary, Scalar(0, 255, 0), 2);
                }
            }
            else {
                // Если круг НЕ внутри квадрата - находим точку на границе квадрата и окружности
                // Находим точку на границе квадрата, ближайшую к кругу
                cv::Point2f rectBoundary = getClosestPointOnRect(trackedObject.boundingBox, predictedPosition);

                // Вычисляем вектор от центра окружности к точке на границе квадрата
                cv::Point2f direction = rectBoundary - predictedPosition;
                float distance = sqrt(direction.x * direction.x + direction.y * direction.y);

                if (distance > 0) {
                    // Нормализуем вектор
                    direction.x /= distance;
                    direction.y /= distance;

                    // Вычисляем точку на границе окружности (вдоль того же направления)
                    cv::Point2f circleBoundary = predictedPosition + direction * circleRadius;

                    // Рисуем ЗЕЛЕНУЮ ЛИНИЮ от границы квадрата до границы окружности
                    line(frame, rectBoundary, circleBoundary, Scalar(0, 255, 0), 2);
                }
            }

            // Подпись рядом с кругом
            cv::putText(frame, "prediction",
                Point(predictedPosition.x + 15, predictedPosition.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        }
    }
    else {
        // Оригинальная отрисовка для обычного объекта
        if (trackedObject.lost) {
            // Если объект потерян - рисуем красную прерывистую рамку
            int lineType = LINE_8;
            int thickness = 2;
            Scalar color(0, 0, 255); // Красный

            // Рисуем пунктирную рамку
            for (int i = 0; i < trackedObject.boundingBox.width; i += 10) {
                line(frame,
                    Point(trackedObject.boundingBox.x + i, trackedObject.boundingBox.y),
                    Point(trackedObject.boundingBox.x + min(i + 5, trackedObject.boundingBox.width),
                        trackedObject.boundingBox.y),
                    color, thickness, lineType);
                line(frame,
                    Point(trackedObject.boundingBox.x + i,
                        trackedObject.boundingBox.y + trackedObject.boundingBox.height),
                    Point(trackedObject.boundingBox.x + min(i + 5, trackedObject.boundingBox.width),
                        trackedObject.boundingBox.y + trackedObject.boundingBox.height),
                    color, thickness, lineType);
            }

            for (int i = 0; i < trackedObject.boundingBox.height; i += 10) {
                line(frame,
                    Point(trackedObject.boundingBox.x, trackedObject.boundingBox.y + i),
                    Point(trackedObject.boundingBox.x,
                        trackedObject.boundingBox.y + min(i + 5, trackedObject.boundingBox.height)),
                    color, thickness, lineType);
                line(frame,
                    Point(trackedObject.boundingBox.x + trackedObject.boundingBox.width,
                        trackedObject.boundingBox.y + i),
                    Point(trackedObject.boundingBox.x + trackedObject.boundingBox.width,
                        trackedObject.boundingBox.y + min(i + 5, trackedObject.boundingBox.height)),
                    color, thickness, lineType);
            }
        }
        else {
            // Если объект найден - рисуем зеленую сплошную рамку
            rectangle(frame, trackedObject.boundingBox, Scalar(0, 255, 0), 3);

            // Рисуем дополнительный внутренний контур
            rectangle(frame, trackedObject.boundingBox, Scalar(0, 200, 0), 1);
        }
    }

    // Рисуем информационную панель
    Rect infoPanel(10, 10, 300, 140);
    rectangle(frame, infoPanel, Scalar(50, 50, 50), -1);
    rectangle(frame, infoPanel, Scalar(200, 200, 200), 1);

    // Отображаем информацию об объекте
    string status = trackedObject.lost ? "LOST" : "TRACKING";
    Scalar statusColor = trackedObject.lost ? Scalar(0, 0, 255) :
        (faceTrackingMode ? Scalar(0, 255, 0) : Scalar(0, 255, 0));

    string trackerType = faceTrackingMode ? "Face Tracker" : "Object Tracker";
    putText(frame, trackerType, Point(20, 35),
        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

    putText(frame, "Status: " + status, Point(20, 65),
        FONT_HERSHEY_SIMPLEX, 0.6, statusColor, 2);

    string ageStr = "Age: " + to_string(trackedObject.age) + " frames";
    putText(frame, ageStr, Point(20, 95),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 1);

    // Отображаем координаты центра
    string coordStr = "Center: (" + to_string((int)trackedObject.center.x) +
        ", " + to_string((int)trackedObject.center.y) + ")";
    putText(frame, coordStr, Point(20, 115),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

    // Отображаем размер
    string sizeStr = "Size: " + to_string(trackedObject.boundingBox.width) +
        "x" + to_string(trackedObject.boundingBox.height);
    putText(frame, sizeStr, Point(20, 135),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

    // Отображаем скорость
    string velocityStr = "Velocity: " + to_string((int)currentVelocity.x) +
        ", " + to_string((int)currentVelocity.y) + " px/s";
    putText(frame, velocityStr, Point(20, 155),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

    // Изменяем курсор в центре кадра на ПЛЮС (+)
    Point2f frameCenter(static_cast<float>(frame.cols) / 2.0f,
        static_cast<float>(frame.rows) / 2.0f);

    // Рисуем ПЛЮС (+) в центре кадра
    int crossSize = 15;
    int lineThickness = 2;
    Scalar crossColor(0, 255, 255); // Желтый цвет

    // Горизонтальная линия плюса
    line(frame,
        Point(frameCenter.x - crossSize, frameCenter.y),
        Point(frameCenter.x + crossSize, frameCenter.y),
        crossColor, lineThickness);

    // Вертикальная линия плюса
    line(frame,
        Point(frameCenter.x, frameCenter.y - crossSize),
        Point(frameCenter.x, frameCenter.y + crossSize),
        crossColor, lineThickness);
}

bool ObjectTracker::isInitialized() const {
    return initialized;
}

void ObjectTracker::reset() {
    initialized = false;
    trackedObject = TrackedObject();
    updateCounter = 0;
    positionHistory.clear();
    velocityHistory.clear();
    currentVelocity = cv::Point2f(0, 0);
    trackedObject.type = faceTrackingMode ? "face" : "object";
}
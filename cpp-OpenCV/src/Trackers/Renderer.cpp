#include "Renderer.h"
#include <algorithm>
#include "FaceTracker.h"

Renderer::Renderer()
    : faceColor(0, 255, 0)           // ������
    , innerFaceColor(0, 200, 0)      // ������-������
    , lineColor(0, 255, 0)           // ������
    , textColor(0, 255, 0)           // ������
    , predictionColor(0, 255, 0)     // ������
    , lostColor(0, 0, 255)          // �������
    , PREDICTION_RADIUS(26)
    , BORDER_THICKNESS(3)
    , INNER_BORDER_THICKNESS(1)
    , PREDICTION_THICKNESS(3)
    , LINE_THICKNESS(2)
{
}

void Renderer::draw(cv::Mat& frame, const std::vector<TrackedFace>& faces, bool isInitialized) const {
    if (faces.empty()) {
        // �������������� ������, ����� ��� ���
        cv::Rect infoPanel(10, 10, 300, 90);
        cv::rectangle(frame, infoPanel, cv::Scalar(50, 50, 50), -1);
        cv::rectangle(frame, infoPanel, cv::Scalar(200, 200, 200), 1);

        cv::putText(frame, "Face Tracker Mode", cv::Point(20, 35),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, "Status: NO FACES", cv::Point(20, 65),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        return;
    }

    // ������ ������ ����
    for (const auto& face : faces) {
        switch (face.currentStatus) {
        case TargetStatus::find:
            drawTargetFind(frame, face);
            break;
        case TargetStatus::lost:
            drawTargetLost(frame, face);
            break;
        case TargetStatus::softlock:
            drawTargetSoftLock(frame, face);
            break;
        case TargetStatus::lock:
            drawTargetLock(frame, face);
            break;
        }
        
    }

    // �������������� ������ �� �����������
    drawInfoPanel(frame, faces);
}

void Renderer::drawTargetLock(cv::Mat& frame, const TrackedFace& face) const {

    // �������� ���� � ������ �����
    cv::rectangle(frame, face.boundingBox, faceColor, BORDER_THICKNESS);
    cv::rectangle(frame, face.boundingBox, innerFaceColor, INNER_BORDER_THICKNESS);
    cv::putText(frame, "FACE",
        cv::Point(face.boundingBox.x, face.boundingBox.y - 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2);

    // �������������� ������� (����� ����)
    cv::Point2f prediction = face.predictedCenter;
    prediction.x = std::max(0.0f, std::min(prediction.x, (float)frame.cols));
    prediction.y = std::max(0.0f, std::min(prediction.y, (float)frame.rows));
    cv::circle(frame, prediction, PREDICTION_RADIUS, predictionColor, PREDICTION_THICKNESS);

    // ����� ���������� (������ ��� � ObjectTracker)
    bool circleInsideRect = prediction.x >= face.boundingBox.x &&
        prediction.x <= face.boundingBox.x + face.boundingBox.width &&
        prediction.y >= face.boundingBox.y &&
        prediction.y <= face.boundingBox.y + face.boundingBox.height;

    if (circleInsideRect) {
        cv::Point2f rectCenter = face.center;
        cv::Point2f direction = prediction - rectCenter;
        float dist = cv::norm(direction);
        if (dist > 0) {
            direction *= (dist - PREDICTION_RADIUS) / dist; // ����� �� ������� �����
            cv::Point2f circleBoundary = rectCenter + direction;
            cv::line(frame, rectCenter, circleBoundary, lineColor, LINE_THICKNESS);
        }
    }
    else {
        cv::Point2f rectBoundary = getClosestPointOnRect(face.boundingBox, prediction);
        cv::Point2f direction = rectBoundary - prediction;
        float dist = cv::norm(direction);
        if (dist > 0) {
            direction *= PREDICTION_RADIUS / dist;
            cv::Point2f circleBoundary = prediction + direction;
            cv::line(frame, rectBoundary, circleBoundary, lineColor, LINE_THICKNESS);
        }
    }

    // ������� "prediction"
    cv::putText(frame, "prediction",
        cv::Point(prediction.x + PREDICTION_RADIUS + 5, prediction.y - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, predictionColor, 1);

    // ���������� ��� ������
    std::string info = "ID: " + std::to_string(face.id) + " Age: " + std::to_string(face.age);
    cv::putText(frame, info,
        cv::Point(face.boundingBox.x, face.boundingBox.y + face.boundingBox.height + 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
}

void Renderer::drawTargetSoftLock(cv::Mat& frame, const TrackedFace& face) const {

    // �������� ���� � ������ �����
    cv::rectangle(frame, face.boundingBox, faceColor, BORDER_THICKNESS);
    cv::rectangle(frame, face.boundingBox, innerFaceColor, INNER_BORDER_THICKNESS);
    cv::putText(frame, "FACE",
        cv::Point(face.boundingBox.x, face.boundingBox.y - 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2);


    // ���������� ��� ������
    std::string info = "ID: " + std::to_string(face.id) + " Age: " + std::to_string(face.age);
    cv::putText(frame, info,
        cv::Point(face.boundingBox.x, face.boundingBox.y + face.boundingBox.height + 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
}

void Renderer::drawTargetFind(cv::Mat& frame, const TrackedFace& face) const {
    
    // ----- NEW: draw only four corners (L‑shaped) -----
    int x = face.boundingBox.x;
    int y = face.boundingBox.y;
    int w = face.boundingBox.width;
    int h = face.boundingBox.height;
    int l = CalculateFindConorLength(w, h);   // length of each corner arm

    // Top‑left
    cv::line(frame, cv::Point(x, y), cv::Point(x + l, y), faceColor, BORDER_THICKNESS);
    cv::line(frame, cv::Point(x, y), cv::Point(x, y + l), faceColor, BORDER_THICKNESS);
    // Top‑right
    cv::line(frame, cv::Point(x + w, y), cv::Point(x + w - l, y), faceColor, BORDER_THICKNESS);
    cv::line(frame, cv::Point(x + w, y), cv::Point(x + w, y + l), faceColor, BORDER_THICKNESS);
    // Bottom‑left
    cv::line(frame, cv::Point(x, y + h), cv::Point(x + l, y + h), faceColor, BORDER_THICKNESS);
    cv::line(frame, cv::Point(x, y + h), cv::Point(x, y + h - l), faceColor, BORDER_THICKNESS);
    // Bottom‑right
    cv::line(frame, cv::Point(x + w, y + h), cv::Point(x + w - l, y + h), faceColor, BORDER_THICKNESS);
    cv::line(frame, cv::Point(x + w, y + h), cv::Point(x + w, y + h - l), faceColor, BORDER_THICKNESS);

    // ----- unchanged: info text below the box -----
    std::string info = "ID: " + std::to_string(face.id) + " Age: " + std::to_string(face.age);
    cv::putText(frame, info,
        cv::Point(face.boundingBox.x, face.boundingBox.y + face.boundingBox.height + 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
}

void Renderer::drawTargetLost(cv::Mat& frame, const TrackedFace& face) const{
    // ----- unchanged: dashed red full rectangle -----
    int dashLength = 10, gapLength = 5;
    cv::Scalar red = lostColor;
    auto drawDashedLine = [&](cv::Point p1, cv::Point p2, bool horizontal) {
        int length = horizontal ? (p2.x - p1.x) : (p2.y - p1.y);
        for (int pos = 0; pos < length; pos += dashLength + gapLength) {
            int endPos = std::min(pos + dashLength, length);
            if (horizontal)
                cv::line(frame, cv::Point(p1.x + pos, p1.y), cv::Point(p1.x + endPos, p1.y), red, BORDER_THICKNESS);
            else
                cv::line(frame, cv::Point(p1.x, p1.y + pos), cv::Point(p1.x, p1.y + endPos), red, BORDER_THICKNESS);
        }
        };

    drawDashedLine(cv::Point(face.boundingBox.x, face.boundingBox.y),
        cv::Point(face.boundingBox.x + face.boundingBox.width, face.boundingBox.y), true);
    drawDashedLine(cv::Point(face.boundingBox.x, face.boundingBox.y + face.boundingBox.height),
        cv::Point(face.boundingBox.x + face.boundingBox.width, face.boundingBox.y + face.boundingBox.height), true);
    drawDashedLine(cv::Point(face.boundingBox.x, face.boundingBox.y),
        cv::Point(face.boundingBox.x, face.boundingBox.y + face.boundingBox.height), false);
    drawDashedLine(cv::Point(face.boundingBox.x + face.boundingBox.width, face.boundingBox.y),
        cv::Point(face.boundingBox.x + face.boundingBox.width, face.boundingBox.y + face.boundingBox.height), false);

    cv::putText(frame, "LOST",
        cv::Point(face.boundingBox.x, face.boundingBox.y - 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, red, 2);
    return;
}

void Renderer::drawInfoPanel(cv::Mat& frame, const std::vector<TrackedFace>& faces) const {
    cv::Rect infoPanel(10, 10, 300, 140);
    cv::rectangle(frame, infoPanel, cv::Scalar(50, 50, 50), -1);
    cv::rectangle(frame, infoPanel, cv::Scalar(200, 200, 200), 1);

    cv::putText(frame, "Face Tracker Mode", cv::Point(20, 35),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    int activeFaces = 0;
    for (const auto& f : faces)
        if (!f.IsLost()) activeFaces++;

    std::string status = "Status: " + std::to_string(activeFaces) + " active / " + std::to_string(faces.size()) + " total";
    cv::Scalar statusColor = (activeFaces > 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::putText(frame, status, cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX, 0.6, statusColor, 2);

    // ����� ������ �������� ����
    if (!faces.empty()) {
        auto largestIt = std::max_element(faces.begin(), faces.end(),
            [](const TrackedFace& a, const TrackedFace& b) {
                return a.boundingBox.area() < b.boundingBox.area();
            });

        cv::putText(frame, "Largest: " + std::to_string(largestIt->boundingBox.width) + "x" + std::to_string(largestIt->boundingBox.height),
            cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        cv::putText(frame, "Velocity: " + std::to_string((int)largestIt->velocity.x) + ", " + std::to_string((int)largestIt->velocity.y) + " px/s",
            cv::Point(20, 115), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        cv::putText(frame, "Prediction: " + std::to_string((int)largestIt->predictedCenter.x) + ", " + std::to_string((int)largestIt->predictedCenter.y),
            cv::Point(20, 135), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    }
}

cv::Point2f Renderer::getClosestPointOnRect(const cv::Rect& rect, const cv::Point2f& point) const {
    float closestX = std::max((float)rect.x, std::min(point.x, (float)(rect.x + rect.width)));
    float closestY = std::max((float)rect.y, std::min(point.y, (float)(rect.y + rect.height)));

    // ���� ����� ������ �������������� � ������� ��������� �������
    if (closestX > rect.x && closestX < rect.x + rect.width &&
        closestY > rect.y && closestY < rect.y + rect.height) {
        float dLeft = closestX - rect.x;
        float dRight = (rect.x + rect.width) - closestX;
        float dTop = closestY - rect.y;
        float dBottom = (rect.y + rect.height) - closestY;
        float minDist = std::min({ dLeft, dRight, dTop, dBottom });

        if (minDist == dLeft)      closestX = rect.x;
        else if (minDist == dRight) closestX = rect.x + rect.width;
        else if (minDist == dTop)  closestY = rect.y;
        else                       closestY = rect.y + rect.height;
    }
    return cv::Point2f(closestX, closestY);
}

cv::Point2f Renderer::getPointOnCircle(const cv::Point2f& circleCenter,
    const cv::Point2f& targetPoint,
    float radius) const {
    cv::Point2f dir = targetPoint - circleCenter;
    float len = cv::norm(dir);
    if (len > 0)
        dir = dir * (radius / len);
    return circleCenter + dir;
}

int Renderer::CalculateFindConorLength(int w, int h) const {
    return w * 0.20;
}

// ������� ��� ������
void Renderer::setFaceColor(const cv::Scalar& color) { faceColor = color; }
void Renderer::setPredictionColor(const cv::Scalar& color) { predictionColor = color; }
void Renderer::setLostColor(const cv::Scalar& color) { lostColor = color; }
// ... ��������� ������� ����������
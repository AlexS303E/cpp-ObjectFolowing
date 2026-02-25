#include "LineModTracker.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

//=============================================================================
// Constructor / Destructor
//=============================================================================

LineModTracker::LineModTracker()
    : nextObjectId_(1)
    , initialized_(false)
    , hasPreviousTime_(false)
    , maxLostFrames_(15)
    , predictionTime_(1.0f)
    , matchThreshold_(0.8f)
{
    // Create the LINE-MOD detector with default modalities (color gradient)
    std::vector<cv::Ptr<cv::linemod::Modality>> modalities;
    modalities.push_back(cv::makePtr<cv::linemod::ColorGradient>());

    // The second parameter is a vector of modality thresholds (one per modality)
    std::vector<int> modalityThresholds(modalities.size(), 64);

    // Create detector using makePtr (OpenCV style)
    detector_ = cv::makePtr<cv::linemod::Detector>(modalities, modalityThresholds);

    // Configure renderer colors
    renderer_.setFaceColor(cv::Scalar(255, 0, 0));      // active objects
    renderer_.setPredictionColor(cv::Scalar(255, 255, 0)); // predicted position
    renderer_.setLostColor(cv::Scalar(128, 128, 128));  // lost objects
}

//=============================================================================
// Public methods
//=============================================================================

bool LineModTracker::loadTemplates(const std::vector<std::string>& templatePaths)
{
    if (templatePaths.empty()) {
        std::cerr << "[LineModTracker] No template paths provided." << std::endl;
        return false;
    }

    bool anyLoaded = false;
    for (const auto& path : templatePaths) {
        // Load template image (grayscale is recommended for LINE-MOD)
        cv::Mat templ = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (templ.empty()) {
            std::cerr << "[LineModTracker] Failed to load template: " << path << std::endl;
            continue;
        }

        // Create a simple mask (all white, i.e., entire image is used)
        cv::Mat mask = cv::Mat::ones(templ.size(), CV_8U) * 255;

        // Generate a class ID for this template
        std::string classId = "obj_" + std::to_string(classIds_.size());
        classIds_.push_back(classId);

        // Add the template to the detector
        std::vector<cv::Mat> sources = { templ };
        cv::Rect rect(0, 0, templ.cols, templ.rows);
        int templateId = detector_->addTemplate(sources, classId, mask, &rect);

        if (templateId == -1) {
            std::cerr << "[LineModTracker] Failed to add template: " << path << std::endl;
            classIds_.pop_back();  // remove the unused class ID
            continue;
        }

        // Store the template and its size for later use in detection
        templates_.push_back(templ);
        templateSizes_.push_back(templ.size());
        anyLoaded = true;

        std::cout << "[LineModTracker] Loaded template '" << path << "' as class '"
            << classId << "' (template ID " << templateId << ")" << std::endl;
    }

    if (!anyLoaded) {
        std::cerr << "[LineModTracker] No templates could be loaded." << std::endl;
        return false;
    }

    std::cout << "[LineModTracker] Successfully loaded " << templates_.size() << " templates." << std::endl;
    return true;
}

bool LineModTracker::initialize(const cv::Mat& frame)
{
    if (templates_.empty()) {
        std::cerr << "[LineModTracker] ERROR: No templates loaded. "
            << "Call loadTemplates() first or provide template images."
            << std::endl;
        return false;
    }
    if (frame.empty()) {
        std::cerr << "[LineModTracker] initialize: empty frame." << std::endl;
        return false;
    }

    // Detect objects on the first frame
    std::vector<cv::Rect> objects = detectObjects(frame);
    if (objects.empty()) {
        std::cout << "[LineModTracker] initialize: no objects detected." << std::endl;
        return false;
    }

    auto currentTime = std::chrono::steady_clock::now();

    // Add all detected objects to the TargetManager
    for (const auto& rect : objects) {
        TrackedFace obj;
        obj.id = nextObjectId_++;
        obj.boundingBox = rect;
        obj.center = cv::Point2f(rect.x + rect.width / 2.0f,
            rect.y + rect.height / 2.0f);
        obj.age = 1;
        obj.currentStatus = TargetStatus::find;
        obj.positionHistory.push_back(obj.center);
        obj.previousPosition = obj.center;
        obj.velocity = cv::Point2f(0, 0);
        obj.predictedCenter = obj.center;
        obj.hasPreviousPosition = false;
        obj.lostFrames = 0;
        obj.lostTimeSet = false;
        targetManager_.addFace(obj);
    }

    // Select the first detected object and set its status to softlock
    if (!targetManager_.getFaces().empty()) {
        targetManager_.selectNext();
        int selectedId = targetManager_.getSelectedId();
        if (selectedId != -1) {
            TrackedFace* objPtr = targetManager_.getFaceById(selectedId);
            if (objPtr) {
                objPtr->currentStatus = TargetStatus::softlock;
                targetManager_.updateFace(selectedId, *objPtr);
                std::cout << "[LineModTracker] Selected object ID " << selectedId
                    << " set to softlock." << std::endl;
            }
        }
    }

    initialized_ = true;
    previousTime_ = currentTime;
    hasPreviousTime_ = true;

    std::cout << "[LineModTracker] Initialized with " << targetManager_.getFaces().size()
        << " objects." << std::endl;
    return true;
}

bool LineModTracker::update(const cv::Mat& frame)
{
    if (!initialized_) {
        std::cerr << "[LineModTracker] update: tracker not initialized." << std::endl;
        return false;
    }

    // Dispatch according to current mode
    switch (trackerMode_) {
    case TrackerMode::src:
        return updateSrc(frame);
    case TrackerMode::trc:
        return updateTrc(frame);
    default:
        return false;
    }
}

bool LineModTracker::updateSrc(const cv::Mat& frame)
{
    // --- Time delta calculation ---
    auto currentTime = std::chrono::steady_clock::now();
    float deltaTime = 0.033f; // default ~30 fps
    if (hasPreviousTime_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            currentTime - previousTime_);
        deltaTime = elapsed.count() / 1000000.0f;
        deltaTime = std::clamp(deltaTime, 0.001f, 0.1f);
    }
    hasPreviousTime_ = true;
    previousTime_ = currentTime;

    // --- Detection ---
    std::vector<cv::Rect> detectedRects = detectObjects(frame);
    // Apply non‑maximum suppression to avoid duplicate detections
    detectedRects = nonMaximumSuppression(detectedRects, 0.3f);

    const auto& currentObjects = targetManager_.getFaces();
    int selectedId = targetManager_.getSelectedId();

    std::vector<bool> used(detectedRects.size(), false);

    // --- Step 1: Update existing objects with new detections ---
    for (const auto& obj : currentObjects) {
        if (obj.currentStatus == TargetStatus::lost)
            continue;

        float bestIoU = 0.3f;
        int bestIdx = -1;
        for (size_t i = 0; i < detectedRects.size(); ++i) {
            if (used[i]) continue;
            float iou = computeIOU(obj.boundingBox, detectedRects[i]);
            if (iou > bestIoU) {
                bestIoU = iou;
                bestIdx = static_cast<int>(i);
            }
        }

        TrackedFace updatedObj = obj;

        if (bestIdx != -1) {
            used[bestIdx] = true;
            updateObjectPosition(updatedObj, detectedRects[bestIdx], deltaTime);
            updatedObj.lostFrames = 0;
            if (updatedObj.currentStatus == TargetStatus::lost) {
                updatedObj.currentStatus = TargetStatus::find;
            }
        }
        else {
            // No matching detection: increment lost counter
            updatedObj.lostFrames++;
            if (updatedObj.lostFrames > maxLostFrames_) {
                updatedObj.currentStatus = TargetStatus::lost;
                updatedObj.lostTime = currentTime;
                updatedObj.lostTimeSet = true;
            }
            else {
                // Predict future position using velocity
                updatedObj.predictedCenter = getPredictedPosition(updatedObj);
            }
        }

        // Ensure the selected object remains in softlock (if not lost)
        if (obj.id == selectedId && updatedObj.currentStatus != TargetStatus::lost) {
            updatedObj.currentStatus = TargetStatus::softlock;
        }

        targetManager_.updateFace(obj.id, updatedObj);
    }

    // --- Step 2: Add new objects from unmatched detections ---
    for (size_t i = 0; i < detectedRects.size(); ++i) {
        if (!used[i]) {
            TrackedFace newObj;
            newObj.id = nextObjectId_++;
            newObj.boundingBox = detectedRects[i];
            newObj.center = cv::Point2f(
                detectedRects[i].x + detectedRects[i].width / 2.0f,
                detectedRects[i].y + detectedRects[i].height / 2.0f);
            newObj.previousPosition = newObj.center;
            newObj.velocity = cv::Point2f(0, 0);
            newObj.predictedCenter = newObj.center;
            newObj.age = 0;
            newObj.lostFrames = 0;
            newObj.currentStatus = TargetStatus::find;
            newObj.hasPreviousPosition = false;
            newObj.positionHistory.push_back(newObj.center);
            targetManager_.addFace(newObj);
        }
    }

    // --- Step 3: Remove objects that have been lost for too long ---
    targetManager_.removeLostFaces(1.5f);  // 1.5 seconds timeout

    // --- Step 4: Ensure the selected object (if any) has the proper status ---
    syncSelectedStatus();

    return !targetManager_.getFaces().empty();
}

bool LineModTracker::updateTrc(const cv::Mat& frame)
{
    // Single‑target mode: we only care about the currently selected object.
    int selectedId = targetManager_.getSelectedId();
    if (selectedId == -1) {
        // No target selected – fall back to SRC behaviour or just return false.
        // Here we simply return false to indicate no active target.
        return false;
    }

    TrackedFace* selectedObj = targetManager_.getFaceById(selectedId);
    if (!selectedObj) {
        return false;
    }

    // --- Time delta ---
    auto currentTime = std::chrono::steady_clock::now();
    float deltaTime = 0.033f;
    if (hasPreviousTime_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            currentTime - previousTime_);
        deltaTime = elapsed.count() / 1000000.0f;
        deltaTime = std::clamp(deltaTime, 0.001f, 0.1f);
    }
    hasPreviousTime_ = true;
    previousTime_ = currentTime;

    // --- Detection (only to locate the target) ---
    std::vector<cv::Rect> detectedRects = detectObjects(frame);
    detectedRects = nonMaximumSuppression(detectedRects, 0.3f);

    // Find the detection closest to the predicted position of the selected object
    cv::Point2f predicted = getPredictedPosition(*selectedObj);
    float minDist = std::numeric_limits<float>::max();
    int bestIdx = -1;
    for (size_t i = 0; i < detectedRects.size(); ++i) {
        cv::Point2f center(detectedRects[i].x + detectedRects[i].width / 2.0f,
            detectedRects[i].y + detectedRects[i].height / 2.0f);
        float dist = calculateDistance(predicted, center);
        // Also require a minimum IoU with the previous bounding box to avoid jumps
        float iou = computeIOU(selectedObj->boundingBox, detectedRects[i]);
        if (dist < minDist && iou > 0.2f) {
            minDist = dist;
            bestIdx = static_cast<int>(i);
        }
    }

    TrackedFace updatedObj = *selectedObj;

    if (bestIdx != -1) {
        // Update with the chosen detection
        updateObjectPosition(updatedObj, detectedRects[bestIdx], deltaTime);
        updatedObj.lostFrames = 0;
        if (updatedObj.currentStatus == TargetStatus::lost) {
            updatedObj.currentStatus = TargetStatus::softlock; // recovered
        }
    }
    else {
        // No suitable detection: count lost frames
        updatedObj.lostFrames++;
        if (updatedObj.lostFrames > maxLostFrames_) {
            updatedObj.currentStatus = TargetStatus::lost;
            updatedObj.lostTime = currentTime;
            updatedObj.lostTimeSet = true;
        }
        else {
            updatedObj.predictedCenter = predicted;
        }
    }

    // The selected object always stays in softlock unless lost
    if (updatedObj.currentStatus != TargetStatus::lost) {
        updatedObj.currentStatus = TargetStatus::softlock;
    }

    targetManager_.updateFace(selectedId, updatedObj);

    return (updatedObj.currentStatus != TargetStatus::lost);
}

void LineModTracker::drawTrackingInfo(cv::Mat& frame) const
{
    if (!initialized_) return;

    if (trackerMode_ == TrackerMode::src) {
        // Draw all objects
        renderer_.draw(frame, targetManager_.getFaces(), initialized_);
    }
    else {
        // Draw only the selected object
        int selectedId = targetManager_.getSelectedId();
        if (selectedId == -1) return;
        const auto& faces = targetManager_.getFaces();
        auto it = std::find_if(faces.begin(), faces.end(),
            [selectedId](const TrackedFace& f) { return f.id == selectedId; });
        if (it == faces.end()) return;
        std::vector<TrackedFace> single = { *it };
        renderer_.draw(frame, single, initialized_);
    }
}

void LineModTracker::reset() {
    targetManager_.clear();
    nextObjectId_ = 1;
    initialized_ = false;
    hasPreviousTime_ = false;
}

bool LineModTracker::isInitialized() const
{
    return initialized_;
}

std::vector<TrackedFace> LineModTracker::getTrackedObjects() const
{
    return targetManager_.getFaces();
}

void LineModTracker::selectNextTrg()
{
    if (trackerMode_ == TrackerMode::trc) return; // not allowed in TRC mode
    MasterTracker::selectNextTrg();  // changes selection in TargetManager
    syncSelectedStatus();
}

void LineModTracker::selectPrevTrg()
{
    if (trackerMode_ == TrackerMode::trc) return;
    MasterTracker::selectPrevTrg();
    syncSelectedStatus();
}

//=============================================================================
// Private helpers
//=============================================================================

std::vector<cv::Rect> LineModTracker::detectObjects(const cv::Mat& frame)
{
    std::vector<cv::Rect> results;
    if (!detector_ || frame.empty()) return results;

    // Prepare input sources (only the frame itself)
    std::vector<cv::Mat> sources = { frame };

    // Run detection
    std::vector<cv::linemod::Match> matches;
    detector_->match(sources, static_cast<float>(matchThreshold_), matches);

    // Convert each match to a bounding rectangle
    for (const auto& match : matches) {
        int templateId = match.template_id;
        if (templateId < 0 || templateId >= static_cast<int>(templateSizes_.size())) {
            // This should not happen if templates were loaded correctly
            continue;
        }
        cv::Size sz = templateSizes_[templateId];
        // The match provides the top‑left corner (x, y)
        cv::Rect rect(match.x, match.y, sz.width, sz.height);
        results.push_back(rect);
    }

    return results;
}

void LineModTracker::updateObjectPosition(TrackedFace& obj,
    const cv::Rect& newRect,
    float deltaTime)
{
    // Exponential smoothing of bounding box
    obj.boundingBox.x = static_cast<int>(0.7f * obj.boundingBox.x + 0.3f * newRect.x);
    obj.boundingBox.y = static_cast<int>(0.7f * obj.boundingBox.y + 0.3f * newRect.y);
    obj.boundingBox.width = static_cast<int>(0.7f * obj.boundingBox.width + 0.3f * newRect.width);
    obj.boundingBox.height = static_cast<int>(0.7f * obj.boundingBox.height + 0.3f * newRect.height);

    cv::Point2f newCenter(obj.boundingBox.x + obj.boundingBox.width / 2.0f,
        obj.boundingBox.y + obj.boundingBox.height / 2.0f);

    updateVelocity(obj, newCenter, deltaTime);
    updatePositionHistory(obj, newCenter);

    obj.center = newCenter;
    obj.predictedCenter = getPredictedPosition(obj);
}

void LineModTracker::updateVelocity(TrackedFace& obj,
    const cv::Point2f& newPosition,
    float deltaTime)
{
    if (obj.positionHistory.size() >= 2) {
        cv::Point2f last = obj.positionHistory.back();
        cv::Point2f displacement = newPosition - last;
        if (deltaTime > 0.0f) {
            cv::Point2f newVel = displacement / deltaTime;
            // Low‑pass filter
            const float alpha = 0.3f;
            obj.velocity = (1.0f - alpha) * obj.velocity + alpha * newVel;
        }
    }
}

void LineModTracker::updatePositionHistory(TrackedFace& obj,
    const cv::Point2f& newPosition)
{
    obj.positionHistory.push_back(newPosition);
    if (obj.positionHistory.size() > maxHistorySize_)
        obj.positionHistory.pop_front();
}

cv::Point2f LineModTracker::getPredictedPosition(const TrackedFace& obj) const
{
    return obj.center + obj.velocity * predictionTime_;
}

float LineModTracker::calculateDistance(const cv::Point2f& p1,
    const cv::Point2f& p2) const
{
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

float LineModTracker::computeIOU(const cv::Rect& a, const cv::Rect& b)
{
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int interWidth = std::max(0, x2 - x1);
    int interHeight = std::max(0, y2 - y1);
    int interArea = interWidth * interHeight;

    int unionArea = a.area() + b.area() - interArea;
    return (unionArea > 0) ? static_cast<float>(interArea) / unionArea : 0.0f;
}

std::vector<cv::Rect> LineModTracker::nonMaximumSuppression(
    const std::vector<cv::Rect>& rects, float threshold)
{
    std::vector<cv::Rect> result;
    if (rects.empty()) return result;

    // Sort rectangles by area (descending) – larger boxes are kept first
    std::vector<size_t> indices(rects.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&rects](size_t i, size_t j) {
            return rects[i].area() > rects[j].area();
        });

    std::vector<bool> suppressed(rects.size(), false);

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        if (suppressed[idx]) continue;

        result.push_back(rects[idx]);

        for (size_t j = i + 1; j < indices.size(); ++j) {
            size_t other = indices[j];
            if (suppressed[other]) continue;

            float iou = computeIOU(rects[idx], rects[other]);
            if (iou > threshold) {
                suppressed[other] = true;
            }
        }
    }

    return result;
}

void LineModTracker::syncSelectedStatus()
{
    int selectedId = targetManager_.getSelectedId();
    if (selectedId == -1) return;

    const auto& objects = targetManager_.getFaces();
    auto it = std::find_if(objects.begin(), objects.end(),
        [selectedId](const TrackedFace& f) { return f.id == selectedId; });

    if (it != objects.end() &&
        it->currentStatus != TargetStatus::softlock &&
        it->currentStatus != TargetStatus::lost)
    {
        TrackedFace updated = *it;
        updated.currentStatus = TargetStatus::softlock;
        targetManager_.updateFace(selectedId, updated);
    }
}
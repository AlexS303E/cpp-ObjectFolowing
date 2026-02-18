#pragma once

#include <opencv2/opencv.hpp>

#include "TargetManager.h"




class MasterTracker {
public:
    virtual ~MasterTracker() = default;

    virtual bool update(const cv::Mat& frame) = 0;

    virtual bool isInitialized() const = 0;

    // Переключиться на следующее лицо в списке (циклически)
    virtual void selectNextTrg() {
        targetManager_.selectNext();
    }
    virtual void selectPrevTrg() {
        targetManager_.selectPrev();
    }
    //virtual void ChangeMode() = 0;
    virtual void reset() = 0;
    virtual void drawTrackingInfo(cv::Mat& frame) const = 0;

protected:
    TargetManager targetManager_;

private:

    std::string name_;
};
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


protected:


private:
    TargetManager targetManager_;

    std::string name_;
};
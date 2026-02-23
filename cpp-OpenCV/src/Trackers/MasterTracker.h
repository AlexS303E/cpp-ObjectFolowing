#pragma once

#include <opencv2/opencv.hpp>

#include "TargetManager.h"




class MasterTracker {
public:
    MasterTracker()
        :trackerMode_(TrackerMode::src)
    {};

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

    // Переключение режим трекера
    virtual void selectPrevTrkMode() {
        TrackerMode oldMode = trackerMode_;
        trackerMode_ = TrackerMode::src;
        std::cout << "Src\n";
        if (oldMode != trackerMode_) {
            setSelectedTargetSoftlock();   // при переходе в SRC
        }
    }
    virtual void selectNextTrkMode() {
        TrackerMode oldMode = trackerMode_;
        trackerMode_ = TrackerMode::trc;
        std::cout << "Trc\n";
        if (oldMode != trackerMode_) {
            setSelectedTargetLock();       // при переходе в TRC
        }
    }

    //virtual void ChangeMode() = 0;
    virtual void reset() = 0;
    virtual void drawTrackingInfo(cv::Mat& frame) const = 0;

protected:
    void setSelectedTargetLock() {
        int selectedId = targetManager_.getSelectedId();
        if (selectedId == -1) return;
        TrackedFace* face = targetManager_.getFaceById(selectedId);
        
        face->currentStatus = TargetStatus::lock;

        std::cout << "id - " << selectedId << " status - " << face->currentStatus << "\n";
    }

    void setSelectedTargetSoftlock() {
        int selectedId = targetManager_.getSelectedId();
        if (selectedId == -1) return;
        TrackedFace* face = targetManager_.getFaceById(selectedId);

        face->currentStatus = TargetStatus::softlock;

        std::cout << "id - " << selectedId << " status - " << face->currentStatus << "\n";

    }

    TargetManager targetManager_;

    std::string name_;

    TrackerMode trackerMode_;

private:

};
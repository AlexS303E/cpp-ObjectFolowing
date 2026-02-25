#include "TargetManager.h"
#include <algorithm>

TargetManager::TargetManager() : selectedId_(-1), mode_(TrackerMode::src) {}

void TargetManager::addFace(const TrackedFace& face) {
    if (mode_ == TrackerMode::src) {
        faces_.push_back(face);
    }
    else {
        // В режиме TRC просто заменяем trackedFace_
        trackedFace_ = face;
    }
}

void TargetManager::updateFace(int id, const TrackedFace& newData) {
    if (mode_ == TrackerMode::src) {
        auto it = std::find_if(faces_.begin(), faces_.end(),
            [id](const TrackedFace& f) { return f.id == id; });
        if (it != faces_.end()) {
            *it = newData;
        }
    }
    else {
        // В режиме TRC обновляем trackedFace_, только если id совпадает
        if (trackedFace_.id == id) {
            trackedFace_ = newData;
        }
    }
}

void TargetManager::removeFace(int id) {
    if (mode_ == TrackerMode::src) {
        auto it = std::remove_if(faces_.begin(), faces_.end(),
            [id](const TrackedFace& f) { return f.id == id; });
        if (it != faces_.end()) {
            faces_.erase(it, faces_.end());
            if (id == selectedId_) {
                selectedId_ = -1;
            }
        }
    }
    else {
        if (trackedFace_.id == id) {
            trackedFace_ = TrackedFace(); // сброс
        }
    }
}

void TargetManager::removeLostFaces(float maxLostTimeSec) {
    if (mode_ == TrackerMode::src) {
        auto now = std::chrono::steady_clock::now();
        auto it = std::remove_if(faces_.begin(), faces_.end(),
            [&](const TrackedFace& f) {
                if (f.currentStatus == TargetStatus::lost && f.lostTimeSet) {
                    auto lostDuration = std::chrono::duration<float>(now - f.lostTime).count();
                    return lostDuration > maxLostTimeSec;
                }
                return false;
            });
        if (it != faces_.end()) {
            for (auto iter = it; iter != faces_.end(); ++iter) {
                if (iter->id == selectedId_) {
                    selectedId_ = -1;
                    break;
                }
            }
            faces_.erase(it, faces_.end());
        }
    }
    else {
        // В режиме TRC проверяем только trackedFace_
        if (trackedFace_.currentStatus == TargetStatus::lost && trackedFace_.lostTimeSet) {
            auto now = std::chrono::steady_clock::now();
            auto lostDuration = std::chrono::duration<float>(now - trackedFace_.lostTime).count();
            if (lostDuration > maxLostTimeSec) {
                trackedFace_ = TrackedFace();
            }
        }
    }
}

void TargetManager::clear() {
    faces_.clear();
    selectedId_ = -1;
    trackedFace_ = TrackedFace();
    // Режим не сбрасываем, чтобы сохранить текущий режим
}

const std::vector<TrackedFace>& TargetManager::getFaces() const {
    // Для совместимости: если режим TRC, возвращаем пустой вектор или вектор с trackedFace_?
    // Лучше возвращать вектор с одним элементом, чтобы код отрисовки работал без изменений.
    static std::vector<TrackedFace> singleFaceVector;
    if (mode_ == TrackerMode::src) {
        return faces_;
    }
    else {
        singleFaceVector.clear();
        if (trackedFace_.id != 0) { // предполагаем, что id=0 не используется
            singleFaceVector.push_back(trackedFace_);
        }
        return singleFaceVector;
    }
}

int TargetManager::getSelectedId() const {
    if (mode_ == TrackerMode::src) {
        return selectedId_;
    }
    else {
        return trackedFace_.id;
    }
}

const TrackedFace* TargetManager::getSelectedFace() const {
    if (mode_ == TrackerMode::src) {
        if (selectedId_ == -1) return nullptr;
        auto it = std::find_if(faces_.begin(), faces_.end(),
            [this](const TrackedFace& f) { return f.id == selectedId_; });
        return (it != faces_.end()) ? &(*it) : nullptr;
    }
    else {
        return (trackedFace_.id != 0) ? &trackedFace_ : nullptr;
    }
}

void TargetManager::selectNext() {
    if (mode_ != TrackerMode::src) return; // в TRC переключение не работает
    if (faces_.empty()) {
        selectedId_ = -1;
        return;
    }
    int currentIdx = -1;
    if (selectedId_ != -1) {
        for (size_t i = 0; i < faces_.size(); ++i) {
            if (faces_[i].id == selectedId_) {
                currentIdx = static_cast<int>(i);
                break;
            }
        }
    }
    int nextIdx = (currentIdx + 1) % static_cast<int>(faces_.size());
    setSelectedFace(faces_[nextIdx].id);
}

void TargetManager::selectPrev() {
    if (mode_ != TrackerMode::src) return;
    if (faces_.empty()) {
        selectedId_ = -1;
        return;
    }
    int currentIdx = -1;
    if (selectedId_ != -1) {
        for (size_t i = 0; i < faces_.size(); ++i) {
            if (faces_[i].id == selectedId_) {
                currentIdx = static_cast<int>(i);
                break;
            }
        }
    }
    int prevIdx = (currentIdx - 1 + static_cast<int>(faces_.size())) % static_cast<int>(faces_.size());
    setSelectedFace(faces_[prevIdx].id);
}

TrackedFace TargetManager::getFaceByIndex(int i) {
    return faces_[i];
}

TrackedFace* TargetManager::getFaceById(int id) {
    if (mode_ == TrackerMode::src) {
        for (auto& face : faces_) {
            if (face.id == id) return &face;
        }
        return nullptr;
    }
    else {
        return (trackedFace_.id == id) ? &trackedFace_ : nullptr;
    }
}

void TargetManager::setSelectedFace(int id) {
    if (mode_ != TrackerMode::src) return;
    // Сброс статуса предыдущего выбранного лица
    if (selectedId_ != -1) {
        for (auto& f : faces_) {
            if (f.id == selectedId_) {
                f.currentStatus = TargetStatus::find;
                break;
            }
        }
    }
    selectedId_ = id;
    if (selectedId_ != -1) {
        for (auto& f : faces_) {
            if (f.id == selectedId_) {
                f.currentStatus = TargetStatus::softlock;
                break;
            }
        }
    }
}

// === Новые методы ===

void TargetManager::setMode(TrackerMode newMode, int targetId) {
    if (mode_ == newMode) return;

    if (newMode == TrackerMode::trc) {
        // Переход в TRC
        if (targetId != -1) {
            auto it = std::find_if(faces_.begin(), faces_.end(),
                [targetId](const TrackedFace& f) { return f.id == targetId; });
            if (it != faces_.end()) {
                trackedFace_ = *it;
            }
            else if (!faces_.empty()) {
                trackedFace_ = faces_[0];
            }
        }
        else {
            const TrackedFace* selected = getSelectedFace();
            if (selected) {
                trackedFace_ = *selected;
            }
            else if (!faces_.empty()) {
                trackedFace_ = faces_[0];
            }
        }
        faces_.clear();
        selectedId_ = -1;
    }
    else { // переход в SRC
        faces_.clear();
        if (trackedFace_.id != 0) { // если есть отслеживаемое лицо
            TrackedFace restored = trackedFace_;
            if (restored.currentStatus != TargetStatus::lost) {
                restored.currentStatus = TargetStatus::softlock;
                selectedId_ = restored.id;
            }
            else {
                selectedId_ = -1;
            }
            faces_.push_back(restored);
        }
        else {
            selectedId_ = -1;
        }
        trackedFace_ = TrackedFace(); // сбрасываем
    }
    mode_ = newMode;
}

void TargetManager::setTrackedFace(const TrackedFace& face) {
    if (mode_ == TrackerMode::trc) {
        trackedFace_ = face;
    }
}

void TargetManager::updateTrackedFace(const TrackedFace& newData) {
    if (mode_ == TrackerMode::trc && trackedFace_.id == newData.id) {
        trackedFace_ = newData;
    }
}

bool TargetManager::hasTrackedFace() const {
    return mode_ == TrackerMode::trc && trackedFace_.id != 0;
}

void TargetManager::setFaces(const std::vector<TrackedFace>& newFaces) {
    faces_ = newFaces;
    // Проверяем, существует ли выбранное лицо в новом списке
    if (selectedId_ != -1) {
        auto it = std::find_if(faces_.begin(), faces_.end(),
            [this](const TrackedFace& f) { return f.id == selectedId_; });
        if (it == faces_.end()) {
            selectedId_ = -1; // выбранное лицо было удалено
        }
    }
    // trackedFace_ не трогаем, так как метод используется только в режиме SRC
}
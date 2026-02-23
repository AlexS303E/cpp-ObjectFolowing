#include "TargetManager.h"

TargetManager::TargetManager() : selectedId_(-1) {}

void TargetManager::addFace(const TrackedFace& face) {
    faces_.push_back(face);
}

void TargetManager::updateFace(int id, const TrackedFace& newData) {
    auto it = std::find_if(faces_.begin(), faces_.end(),
        [id](const TrackedFace& f) { return f.id == id; });
    if (it != faces_.end()) {
        *it = newData;
    }
}

void TargetManager::removeFace(int id) {
    auto it = std::remove_if(faces_.begin(), faces_.end(),
        [id](const TrackedFace& f) { return f.id == id; });
    if (it != faces_.end()) {
        faces_.erase(it, faces_.end());
        if (id == selectedId_) {
            selectedId_ = -1;
        }
    }
}

void TargetManager::removeLostFaces(float maxLostTimeSec) {
    auto now = std::chrono::steady_clock::now();
    auto it = std::remove_if(faces_.begin(), faces_.end(),
        [&](const TrackedFace& f) {
            if (f.currentStatus == TargetStatus::lost && f.lostTimeSet) {
                auto lostDuration = std::chrono::duration_cast<std::chrono::milliseconds>(now - f.lostTime).count() / 1000.0f;
                return lostDuration > maxLostTimeSec;
            }
            return false;
        });
    if (it != faces_.end()) {
        // Проверим, не удаляем ли мы выбранное лицо
        for (auto iter = it; iter != faces_.end(); ++iter) {
            if (iter->id == selectedId_) {
                selectedId_ = -1;
                break;
            }
        }
        faces_.erase(it, faces_.end());
    }
}

void TargetManager::clear() {
    faces_.clear();
    selectedId_ = -1;
}

const std::vector<TrackedFace>& TargetManager::getFaces() const {
    return faces_;
}

int TargetManager::getSelectedId() const {
    return selectedId_;
}

const TrackedFace* TargetManager::getSelectedFace() const {
    if (selectedId_ == -1) return nullptr;
    auto it = std::find_if(faces_.begin(), faces_.end(),
        [this](const TrackedFace& f) { return f.id == selectedId_; });
    return (it != faces_.end()) ? &(*it) : nullptr;
}

void TargetManager::selectPrev() {
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

void TargetManager::selectNext() {
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

TrackedFace TargetManager::getFaceByIndex(int i) {
    return faces_[i];
}

TrackedFace* TargetManager::getFaceById(int id) {
    for (auto& face : faces_) {
        if (face.id == id) return &face;
    }
    return nullptr;
}

void TargetManager::setSelectedFace(int id) {
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
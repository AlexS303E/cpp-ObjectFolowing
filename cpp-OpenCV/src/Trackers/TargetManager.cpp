#include "TargetManager.h"
#include <algorithm>

TargetManager::TargetManager() : selectedId_(-1) {}

void TargetManager::update(const std::vector<TrackedFace>& faces) {
    faces_ = faces;  // полная копия

    // Проверить, не пропало ли выбранное лицо из нового списка
    if (selectedId_ != -1) {
        auto it = std::find_if(faces_.begin(), faces_.end(),
            [this](const TrackedFace& f) { return f.id == selectedId_; });
        if (it == faces_.end()) {
            // Лицо больше не отслеживается – сбрасываем выбор
            selectedId_ = -1;
        }
    }
}

void TargetManager::selectNext() {
    if (faces_.empty()) {
        selectedId_ = -1;
        return;
    }

    // Найти индекс текущего выбранного лица (если есть)
    int currentIdx = -1;
    if (selectedId_ != -1) {
        for (size_t i = 0; i < faces_.size(); ++i) {
            if (faces_[i].id == selectedId_) {
                currentIdx = static_cast<int>(i);
                break;
            }
        }
    }

    // Вычислить индекс следующего лица
    int nextIdx;
    if (currentIdx == -1) {
        nextIdx = 0;                           // нет выбора → берём первое
    }
    else {
        nextIdx = (currentIdx + 1) % static_cast<int>(faces_.size());
    }

    setSelectedFace(faces_[nextIdx].id);
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

    int prevIdx;
    if (currentIdx == -1) {
        prevIdx = static_cast<int>(faces_.size()) - 1; // последнее
    }
    else {
        prevIdx = (currentIdx - 1 + static_cast<int>(faces_.size())) % static_cast<int>(faces_.size());
    }

    setSelectedFace(faces_[prevIdx].id);
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

const std::vector<TrackedFace>& TargetManager::getFaces() const {
    return faces_;
}

void TargetManager::setSelectedFace(int id) {
    // Сбросить статус предыдущего выбранного лица (если оно было)
    if (selectedId_ != -1) {
        for (auto& f : faces_) {
            if (f.id == selectedId_) {
                // Возвращаем обычный статус (предполагаем, что до выделения был find)
                // Если лицо было потеряно, статус lost останется – мы не меняем его,
                // но для простоты ставим find, так как при следующем update трекер всё равно скорректирует.
                f.currentStatus = TargetStatus::find;
                break;
            }
        }
    }

    // Установить новое выделение
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
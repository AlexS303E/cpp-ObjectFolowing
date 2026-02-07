#include "yolo.h"

namespace fs = std::filesystem;


bool IsFileExist(const std::string& file) {
    if (!fs::exists(file)) {
        std::cerr << "\nФАЙЛ НЕ НАЙДЕН: " << fs::absolute(file) << std::endl;
        std::cerr << "Текущая рабочая директория: " << fs::current_path() << std::endl;
        return false;
    }
    return true;
}



YOLOv3Detector::YOLOv3Detector(float confThreshold, float nmsThresh, int width, int height)
: confidenceThreshold(confThreshold), nmsThreshold(nmsThresh),
inputWidth(width), inputHeight(height) {

    if (!IsFileExist(YOLOv3CONF)) {
        return;
    }
    if (!IsFileExist(YOLOv3WEIGHT)) {
        return;
    }
    if (!IsFileExist(YOLOv3WEIGHT)) {
        return;
    }


    // Загрузка классов
    loadClasses(CLASSES);

    // Генерация цветов для каждого класса
    generateColors();

    // Загрузка сети YOLO
    net = cv::dnn::readNetFromDarknet(YOLOv3CONF, YOLOv3WEIGHT);

    if (net.empty()) {
        throw std::runtime_error("Ошибка: не удалось загрузить модель YOLO!");
    }

    // Настройка вычислений (CPU/OpenCV, можно заменить на CUDA)
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::cout << "YOLOv3 модель загружена успешно!" << std::endl;
    std::cout << "Классов загружено: " << classNames.size() << std::endl;
}

void YOLOv3Detector::loadClasses(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Ошибка: не удалось открыть файл классов: " + filename);
    }

    std::string className;
    while (std::getline(file, className)) {
        if (!className.empty()) {
            classNames.push_back(className);
        }
    }
}

void YOLOv3Detector::generateColors() {
    // Генерация уникальных цветов для каждого класса
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100, 255);

    for (size_t i = 0; i < classNames.size(); ++i) {
        colors.push_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
    }
}

void YOLOv3Detector::detectAndDraw(cv::Mat& frame) {
    // Создание blob из изображения
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(inputWidth, inputHeight),
        cv::Scalar(0, 0, 0), true, false);

    // Установка blob как входа сети
    net.setInput(blob);

    // Получение имен выходных слоев
    std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();

    // Прямой проход
    std::vector<cv::Mat> outs;
    net.forward(outs, outNames);

    // Обработка выходов
    postprocess(frame, outs);

    // Отображение FPS
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time: %.2f ms", t);
    cv::putText(frame, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
        0.7, cv::Scalar(0, 255, 0), 2);
}

void YOLOv3Detector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        const float* data = reinterpret_cast<const float*>(outs[i].data);

        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;

            // Получение класса с максимальной уверенностью
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > confidenceThreshold) {
                int centerX = static_cast<int>(data[0] * frame.cols);
                int centerY = static_cast<int>(data[1] * frame.rows);
                int width = static_cast<int>(data[2] * frame.cols);
                int height = static_cast<int>(data[3] * frame.rows);

                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back(static_cast<float>(confidence));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Применение Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

    // Отрисовка результатов
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box, frame);
    }
}

void YOLOv3Detector::drawPred(int classId, float conf, const cv::Rect& box, cv::Mat& frame) {
    // Отрисовка рамки
    cv::rectangle(frame, box, colors[classId % colors.size()], 2);

    // Формирование текста
    std::string label = cv::format("%.2f", conf);
    if (classId < static_cast<int>(classNames.size())) {
        label = classNames[classId] + ": " + label;
    }

    // Фон для текста
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int top = std::max(box.y, labelSize.height);
    cv::rectangle(frame, cv::Point(box.x, top - labelSize.height),
        cv::Point(box.x + labelSize.width, top + baseLine),
        colors[classId % colors.size()], cv::FILLED);

    // Текст
    cv::putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX,
        0.5, cv::Scalar(0, 0, 0), 1);
}



YOLOv8Detector::YOLOv8Detector(float confThreshold, float nmsThresh, int width, int height)
    : confidenceThreshold(confThreshold), nmsThreshold(nmsThresh), inputWidth(width), inputHeight(height) {

    // Загрузка классов
    loadClassNames(CLASSES);

    // Генерация цветов
    generateColors();

    // Загрузка ONNX модели
    net = cv::dnn::readNetFromONNX(YOLOv8n);

    if (net.empty()) {
        throw std::runtime_error("Ошибка: не удалось загрузить модель YOLOv8 из " + YOLOv8n);
    }

    // Настройка вычислений
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

    std::cout << "YOLOv8 модель загружена успешно!" << std::endl;
    std::cout << "Размер входа: " << inputWidth << "x" << inputHeight << std::endl;
    std::cout << "Классов загружено: " << classNames.size() << std::endl;
}

void YOLOv8Detector::loadClassNames(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл классов: " + filename);
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) classNames.push_back(line);
    }
}

void YOLOv8Detector::generateColors() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100, 255);
    for (size_t i = 0; i < classNames.size(); ++i) {
        colors.push_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
    }
}

void YOLOv8Detector::detectAndDraw(cv::Mat& frame) {
    // Создание blob
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(inputWidth, inputHeight),
        cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // Прямой проход
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Постобработка
    postprocess(frame, outputs);

    // Отображение FPS
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    cv::putText(frame, cv::format("Inference: %.2f ms", t),
        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

void YOLOv8Detector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs) {
    // YOLOv8 output format: [1, 84, 8400] (xywh + conf + 80 classes)
    const float* data = reinterpret_cast<const float*>(outs[0].data);
    const int numChannels = outs[0].size[1]; // 84
    const int numAnchors = outs[0].size[2];  // 8400

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < numAnchors; ++i) {
        float objectConfidence = data[4 * numAnchors + i];

        if (objectConfidence >= confidenceThreshold) {
            // Поиск класса с максимальным score
            float maxClassScore = 0;
            int maxClassId = 0;

            for (int c = 0; c < numChannels - 5; ++c) {
                float score = data[(5 + c) * numAnchors + i];
                if (score > maxClassScore) {
                    maxClassScore = score;
                    maxClassId = c;
                }
            }

            float totalConfidence = objectConfidence * maxClassScore;

            if (totalConfidence >= confidenceThreshold) {
                // Координаты bbox
                float cx = data[i];
                float cy = data[numAnchors + i];
                float w = data[2 * numAnchors + i];
                float h = data[3 * numAnchors + i];

                // Конвертация в пиксели
                int left = static_cast<int>((cx - w / 2) * frame.cols);
                int top = static_cast<int>((cy - h / 2) * frame.rows);
                int width = static_cast<int>(w * frame.cols);
                int height = static_cast<int>(h * frame.rows);

                // Клиппинг
                left = std::max(0, std::min(left, frame.cols - 1));
                top = std::max(0, std::min(top, frame.rows - 1));
                width = std::max(1, std::min(width, frame.cols - left));
                height = std::max(1, std::min(height, frame.rows - top));

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(totalConfidence);
                classIds.push_back(maxClassId);
            }
        }
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

    // Отрисовка
    for (int idx : indices) {
        drawBox(classIds[idx], confidences[idx], boxes[idx], frame);
    }

    // Статистика
    cv::putText(frame, "Objects: " + std::to_string(indices.size()),
        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

void YOLOv8Detector::drawBox(int classId, float conf, const cv::Rect& box, cv::Mat& frame) {
    cv::rectangle(frame, box, colors[classId % colors.size()], 2);

    std::string label = cv::format("%.2f", conf);
    if (classId < static_cast<int>(classNames.size())) {
        label = classNames[classId] + ": " + label;
    }

    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int top = std::max(box.y, labelSize.height);

    cv::rectangle(frame, cv::Point(box.x, top - labelSize.height),
        cv::Point(box.x + labelSize.width, top + baseLine),
        colors[classId % colors.size()], cv::FILLED);

    cv::putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX,
        0.5, cv::Scalar(0, 0, 0), 1);
}


YOLOv26Detector::YOLOv26Detector(float confThreshold, float nmsThresh, int width, int height)
    : confidenceThreshold(confThreshold), nmsThreshold(nmsThresh),
    inputWidth(width), inputHeight(height) {

    // Загрузка классов
    loadClassNames(CLASSES);

    // Генерация цветов
    generateColors();

    // Загрузка ONNX модели YOLOv26
    net = cv::dnn::readNetFromONNX(YOLOv26n);


    // Настройка вычислений
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::cout << "YOLOv26 модель загружена успешно!" << std::endl;
    std::cout << "Классов: " << classNames.size() << std::endl;
}

void YOLOv26Detector::loadClassNames(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл классов: " + filename);
    }
    std::string line;
    while (getline(file, line)) {
        if (!line.empty()) classNames.push_back(line);
    }
}

void YOLOv26Detector::generateColors() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100, 255);
    for (size_t i = 0; i < classNames.size(); ++i) {
        colors.push_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
    }
}

void YOLOv26Detector::detectAndDraw(cv::Mat& frame) {
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(inputWidth, inputHeight),
        cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // Выполнение прямого прохода
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Постобработка для YOLOv26
    postprocess(frame, outputs);

    // Отображение FPS
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    cv::putText(frame, cv::format("YOLOv26 Inference: %.2f ms", t),
        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

void YOLOv26Detector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outputs) {
    // YOLOv26 output format: [1, 84, 8400] (xywh + conf + 80 classes)
    const float* data = reinterpret_cast<const float*>(outputs[0].data);

    // Получаем размеры вывода
    const int numChannels = outputs[0].size[1]; // 84
    const int numAnchors = outputs[0].size[2];  // 8400

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Парсинг детекций
    for (int i = 0; i < numAnchors; ++i) {
        // Получаем уверенность объекта
        float objectConfidence = data[4 * numAnchors + i];

        if (objectConfidence >= confidenceThreshold) {
            // Получаем оценки классов
            const float* classScores = data + 5 * numAnchors + i;

            // Находим класс с максимальной оценкой
            float maxClassScore = 0;
            int maxClassId = 0;
            for (int c = 0; c < (numChannels - 5); ++c) {
                float score = classScores[c * numAnchors];
                if (score > maxClassScore) {
                    maxClassScore = score;
                    maxClassId = c;
                }
            }

            // Общая уверенность = уверенность объекта * уверенность класса
            float totalConfidence = objectConfidence * maxClassScore;

            if (totalConfidence >= confidenceThreshold) {
                // Координаты bbox (нормализованные)
                float cx = data[i];
                float cy = data[numAnchors + i];
                float w = data[2 * numAnchors + i];
                float h = data[3 * numAnchors + i];

                // Преобразование в пиксели
                int left = static_cast<int>((cx - w / 2) * frame.cols);
                int top = static_cast<int>((cy - h / 2) * frame.rows);
                int width = static_cast<int>(w * frame.cols);
                int height = static_cast<int>(h * frame.rows);

                // Корректировка координат
                left = std::max(0, std::min(left, frame.cols - 1));
                top = std::max(0, std::min(top, frame.rows - 1));
                width = std::max(1, std::min(width, frame.cols - left));
                height = std::max(1, std::min(height, frame.rows - top));

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(totalConfidence);
                classIds.push_back(maxClassId);
            }
        }
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

    // Отрисовка
    for (int idx : indices) {
        drawBox(classIds[idx], confidences[idx], boxes[idx], frame);
    }

    // Статистика
    cv::putText(frame, "YOLOv26 Objects: " + std::to_string(indices.size()),
        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

void YOLOv26Detector::drawBox(int classId, float conf, const cv::Rect& box, cv::Mat& frame) {
    cv::rectangle(frame, box, colors[classId % colors.size()], 2);

    std::string label = cv::format("%.2f", conf);
    if (classId < (int)classNames.size()) {
        label = classNames[classId] + ": " + label;
    }

    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int top = std::max(box.y, labelSize.height);
    cv::rectangle(frame, cv::Point(box.x, top - labelSize.height),
        cv::Point(box.x + labelSize.width, top + baseLine),
        colors[classId % colors.size()], cv::FILLED);
    cv::putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX,
        0.5, cv::Scalar(0, 0, 0), 1);
}
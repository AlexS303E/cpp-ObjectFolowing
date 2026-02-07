#include "TFLiteDetector.h"
#include <fstream>
#include <iostream>

TFLiteDetector::TFLiteDetector(const std::string& modelPath,
    const std::string& labelsPath,
    int inputWidth, int inputHeight)
    : modelPath_(modelPath), labelsPath_(labelsPath),
    inputWidth_(inputWidth), inputHeight_(inputHeight) {
}

bool TFLiteDetector::initialize() {
    // 1. Загрузка модели
    model_ = tflite::FlatBufferModel::BuildFromFile(modelPath_.c_str());
    if (!model_) {
        std::cerr << "❌ Не удалось загрузить модель: " << modelPath_ << std::endl;
        return false;
    }

    // 2. Создание интерпретатора
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
    if (!interpreter_) {
        std::cerr << "❌ Не удалось создать интерпретатор TFLite." << std::endl;
        return false;
    }

    // 3. (ОПЦИОНАЛЬНО) Настройка GPU делегата для ускорения
    if (useGPUDelegate_) {
        // Попробуйте создать и применить GPU Delegate (OpenCL)
        // Важно: это экспериментальный шаг для AMD
        // Вам может потребоваться TfLiteGpuDelegateV2Create
        // delegate_.reset(TfLiteGpuDelegateV2Create(&options));
        // if (interpreter_->ModifyGraphWithDelegate(delegate_.get()) != kTfLiteOk) {
        //     std::cerr << "⚠️ GPU делегат не сработал, используется CPU." << std::endl;
        //     delegate_.reset();
        // }
    }

    // 4. Выделение памяти для тензоров
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        std::cerr << "❌ Ошибка выделения памяти для тензоров." << std::endl;
        return false;
    }

    // 5. Загрузка меток классов
    loadLabels(labelsPath_);

    std::cout << "✅ TFLite детектор инициализирован." << std::endl;
    return true;
}

void TFLiteDetector::detectAndDraw(cv::Mat& frame) {
    cv::Mat processed;
    preprocess(frame, processed); // Масштабирование, нормализация

    // 1. Копирование данных в входной тензор
    float* inputTensor = interpreter_->typed_input_tensor<float>(0);
    // Пример для модели с входом [1, height, width, 3]
    // Необходимо скопировать данные processed.data в inputTensor
    // Возможно, потребуется преобразование формата (например, BGR -> RGB)

    // 2. Запуск инференса
    if (interpreter_->Invoke() != kTfLiteOk) {
        std::cerr << "❌ Ошибка при выполнении модели." << std::endl;
        return;
    }

    // 3. Получение выходных данных
    float* outputTensor = interpreter_->typed_output_tensor<float>(0);
    postprocess(frame, outputTensor);
}

void TFLiteDetector::preprocess(const cv::Mat& input, cv::Mat& output) {
    // Преобразование входного кадра под формат модели:
    // 1. Изменение размера до inputWidth_ x inputHeight_
    // 2. Нормализация значений пикселей (например, в диапазон [0,1] или [-1,1])
    // 3. Преобразование цветового пространства (BGR -> RGB)
}
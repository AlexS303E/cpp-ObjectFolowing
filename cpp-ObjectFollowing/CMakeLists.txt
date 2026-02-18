cmake_minimum_required(VERSION 3.20)
project(MyOpenCVProject VERSION 1.0.0 LANGUAGES CXX)

# === Общие настройки ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Включаем поддержку filesystem (C++17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
elseif(MSVC)
    add_compile_definitions(_SILENCE_CXX17_FILESYSTEM_DEPRECATION_WARNING)
endif()

# === Поддержка vcpkg ===
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# === Поиск OpenCV ===
find_package(OpenCV REQUIRED COMPONENTS core highgui videoio imgproc dnn imgcodecs objdetect)

if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV or set OpenCV_DIR")
endif()

# === Список файлов проекта ===
set(SOURCE_FILES
    src/main.cpp
    src/global.cpp
    src/WebcamViewer.cpp
    src/Trackers/ObjectTracker.cpp
    src/Trackers/FaceTracker.cpp
    src/Trackers/EdgeDetector.cpp
    src/Trackers/TrackerManager.cpp
    src/Trackers/TargetManager.cpp
)

set(HEADER_FILES
    src/global.h
    src/WebcamViewer.h
    src/Trackers/ObjectTracker.h
    src/Trackers/FaceTracker.h
    src/Trackers/EdgeDetector.h
    src/Trackers/TrackerManager.h
    src/Trackers/TargetManager.h
    src/Trackers/MasterTracker.h
)

# === Создание исполняемого файла ===
add_executable(WebcamViewer ${SOURCE_FILES} ${HEADER_FILES} "src/Trackers/Renderer.h" "src/Trackers/Renderer.cpp" "src/Trackers/TargetManager.h" "src/Trackers/TargetManager.cpp")

# === Настройки цели ===
set_target_properties(WebcamViewer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    DEBUG_POSTFIX "_d"
)

# === Подключение библиотек ===
target_link_libraries(WebcamViewer PRIVATE ${OpenCV_LIBS})

# Директории включения
target_include_directories(WebcamViewer PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

# === Специфичные настройки для Windows ===
if(WIN32)
    target_compile_definitions(WebcamViewer PRIVATE 
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

# === Копирование папки models ===
# Создаем папку models рядом с исполняемым файлом
set(MODELS_TARGET_DIR ${CMAKE_BINARY_DIR}/bin/models)

# Создаем директорию
add_custom_command(TARGET WebcamViewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${MODELS_TARGET_DIR}
    COMMENT "Creating models directory..."
)

# Копируем файлы, если они есть в исходниках
if(EXISTS ${CMAKE_SOURCE_DIR}/models AND IS_DIRECTORY ${CMAKE_SOURCE_DIR}/models)
    file(GLOB MODEL_FILES "${CMAKE_SOURCE_DIR}/models/*")
    if(MODEL_FILES)
        add_custom_command(TARGET WebcamViewer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${MODEL_FILES} ${MODELS_TARGET_DIR}/
            COMMENT "Copying model files..."
        )
    endif()
else()
    message(STATUS "Models folder not found. Will create empty directory.")
endif()

# === Настройка для Visual Studio (опционально) ===
if(MSVC)
    # Включаем предупреждения
    target_compile_options(WebcamViewer PRIVATE /W4)
endif()

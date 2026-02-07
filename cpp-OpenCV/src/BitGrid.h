#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>
#include <string>

// Перечисление методов сжатия
enum CompressionMethod {
    COMPRESSION_NONE = 0,     // Без сжатия
    COMPRESSION_RLE = 1,      // Run-Length Encoding
    COMPRESSION_LZ4 = 2,      // LZ4 (быстрое сжатие)
    COMPRESSION_HUFFMAN = 3   // Кодирование Хаффмана
};

class BitGrid {
public:
    // Конструкторы
    BitGrid();
    BitGrid(int width, int height);
    BitGrid(const cv::Mat& edgeImage);

    // Основные методы
    bool get(int x, int y) const;
    void set(int x, int y, bool value);
    void clear();

    // Конвертация
    cv::Mat toImage() const;
    std::vector<uint8_t> toBytes() const;
    void fromBytes(const std::vector<uint8_t>& data, int width, int height);

    // Информация
    int width() const { return m_width; }
    int height() const { return m_height; }
    int size() const { return m_width * m_height; }
    int byteSize() const { return (size() + 7) / 8; }

    // Операции
    void resize(int width, int height);
    BitGrid operator&(const BitGrid& other) const;
    BitGrid operator|(const BitGrid& other) const;
    BitGrid operator~() const;

    // Статистика
    int countTrue() const;
    float density() const;

    // Утилиты
    void save(const std::string& filename, CompressionMethod method = COMPRESSION_RLE) const;
    void load(const std::string& filename);

    // Методы сжатия
    std::vector<uint8_t> compress(CompressionMethod method = COMPRESSION_RLE) const;
    bool decompress(const std::vector<uint8_t>& compressedData);

    // Информация о сжатии
    struct CompressionInfo {
        int originalSize;
        int compressedSize;
        float ratio;
        CompressionMethod method;
    };

    CompressionInfo getCompressionInfo(const std::vector<uint8_t>& compressedData) const;

private:
    int m_width;
    int m_height;
    std::vector<uint8_t> m_data;

    // Методы сжатия (приватные реализации)
    std::vector<uint8_t> compressRLE() const;
    std::vector<uint8_t> compressLZ4() const;
    std::vector<uint8_t> compressHuffman() const;

    // Методы распаковки
    bool decompressRLE(const std::vector<uint8_t>& data);
    bool decompressLZ4(const std::vector<uint8_t>& data);
    bool decompressHuffman(const std::vector<uint8_t>& data);

    // Вспомогательные методы
    void setInternal(int index, bool value);
    bool getInternal(int index) const;
    int calculateByteIndex(int bitIndex) const;
    uint8_t calculateBitMask(int bitIndex) const;

    // Статистика для оптимизации сжатия
    void analyzeCompressionStats() const;
};
#include "BitGrid.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <map>
#include <queue>
#include <algorithm>

using namespace std;

// Простая реализация LZ4-подобного сжатия
namespace LZ4Simple {
    vector<uint8_t> compress(const vector<uint8_t>& input) {
        vector<uint8_t> output;
        size_t i = 0;
        const size_t maxOffset = 1024; // Максимальное смещение для поиска

        while (i < input.size()) {
            size_t bestMatchLen = 0;
            size_t bestMatchPos = 0;

            // Ищем совпадения в предыдущих данных
            size_t searchStart = (i > maxOffset) ? i - maxOffset : 0;

            for (size_t j = searchStart; j < i; ++j) {
                size_t matchLen = 0;
                while (i + matchLen < input.size() &&
                    j + matchLen < i &&
                    input[j + matchLen] == input[i + matchLen] &&
                    matchLen < 255) {
                    ++matchLen;
                }

                if (matchLen > bestMatchLen && matchLen >= 4) {
                    bestMatchLen = matchLen;
                    bestMatchPos = j;
                }
            }

            if (bestMatchLen >= 4) {
                // Кодируем ссылку
                uint16_t offset = i - bestMatchPos;
                output.push_back(0xFF); // Маркер ссылки
                output.push_back(static_cast<uint8_t>(offset & 0xFF));
                output.push_back(static_cast<uint8_t>((offset >> 8) & 0xFF));
                output.push_back(static_cast<uint8_t>(bestMatchLen));
                i += bestMatchLen;
            }
            else {
                // Кодируем литерал
                output.push_back(input[i]);
                ++i;
            }
        }

        return output;
    }

    vector<uint8_t> decompress(const vector<uint8_t>& input) {
        vector<uint8_t> output;
        size_t i = 0;

        while (i < input.size()) {
            if (input[i] == 0xFF && i + 3 < input.size()) {
                // Это ссылка
                uint16_t offset = (input[i + 2] << 8) | input[i + 1];
                uint8_t length = input[i + 3];

                size_t refPos = output.size() - offset;
                for (int j = 0; j < length; ++j) {
                    output.push_back(output[refPos + j]);
                }

                i += 4;
            }
            else {
                // Это литерал
                output.push_back(input[i]);
                ++i;
            }
        }

        return output;
    }
}

// Узел дерева Хаффмана
struct HuffmanNode {
    uint8_t symbol;
    int frequency;
    HuffmanNode* left;
    HuffmanNode* right;

    HuffmanNode(uint8_t s, int f) : symbol(s), frequency(f), left(nullptr), right(nullptr) {}

    bool isLeaf() const { return left == nullptr && right == nullptr; }
};

// Компаратор для приоритетной очереди Хаффмана
struct CompareNodes {
    bool operator()(HuffmanNode* a, HuffmanNode* b) {
        return a->frequency > b->frequency;
    }
};

// Класс для кодирования Хаффмана
class HuffmanCoder {
private:
    map<uint8_t, string> codeTable;
    map<string, uint8_t> decodeTable;

    void buildTree(const map<uint8_t, int>& frequencies) {
        priority_queue<HuffmanNode*, vector<HuffmanNode*>, CompareNodes> pq;

        // Создаем узлы для каждого символа
        for (const auto& pair : frequencies) {
            pq.push(new HuffmanNode(pair.first, pair.second));
        }

        // Строим дерево
        while (pq.size() > 1) {
            HuffmanNode* left = pq.top(); pq.pop();
            HuffmanNode* right = pq.top(); pq.pop();

            HuffmanNode* parent = new HuffmanNode(0, left->frequency + right->frequency);
            parent->left = left;
            parent->right = right;

            pq.push(parent);
        }

        // Строим таблицу кодов
        if (!pq.empty()) {
            HuffmanNode* root = pq.top();
            buildCodeTable(root, "");
            // Удаляем дерево
            deleteTree(root);
        }
    }

    void deleteTree(HuffmanNode* node) {
        if (node) {
            deleteTree(node->left);
            deleteTree(node->right);
            delete node;
        }
    }

    void buildCodeTable(HuffmanNode* node, string code) {
        if (node == nullptr) return;

        if (node->isLeaf()) {
            codeTable[node->symbol] = code;
            decodeTable[code] = node->symbol;
        }
        else {
            buildCodeTable(node->left, code + "0");
            buildCodeTable(node->right, code + "1");
        }
    }

public:
    ~HuffmanCoder() {
        // Очищаем таблицы
        codeTable.clear();
        decodeTable.clear();
    }

    vector<uint8_t> encode(const vector<uint8_t>& data) {
        // Считаем частоты
        map<uint8_t, int> frequencies;
        for (uint8_t byte : data) {
            frequencies[byte]++;
        }

        // Строим дерево и таблицу кодов
        buildTree(frequencies);

        // Кодируем данные
        string bitString;
        for (uint8_t byte : data) {
            bitString += codeTable[byte];
        }

        // Преобразуем битовую строку в байты
        vector<uint8_t> result;

        // Сохраняем таблицу частот (первые 256 int для частот)
        for (int i = 0; i < 256; ++i) {
            int freq = (frequencies.find(i) != frequencies.end()) ? frequencies[i] : 0;
            result.push_back((freq >> 24) & 0xFF);
            result.push_back((freq >> 16) & 0xFF);
            result.push_back((freq >> 8) & 0xFF);
            result.push_back(freq & 0xFF);
        }

        // Добавляем длину битовой строки
        uint32_t bitLength = bitString.length();
        result.push_back((bitLength >> 24) & 0xFF);
        result.push_back((bitLength >> 16) & 0xFF);
        result.push_back((bitLength >> 8) & 0xFF);
        result.push_back(bitLength & 0xFF);

        // Добавляем закодированные данные
        uint8_t currentByte = 0;
        int bitCount = 0;

        for (char bit : bitString) {
            currentByte = (currentByte << 1) | (bit == '1' ? 1 : 0);
            bitCount++;

            if (bitCount == 8) {
                result.push_back(currentByte);
                currentByte = 0;
                bitCount = 0;
            }
        }

        // Добавляем последний неполный байт
        if (bitCount > 0) {
            currentByte <<= (8 - bitCount);
            result.push_back(currentByte);
        }

        return result;
    }

    vector<uint8_t> decode(const vector<uint8_t>& encodedData) {
        vector<uint8_t> result;

        if (encodedData.size() < 256 * 4 + 4) {
            return result;
        }

        // Читаем таблицу частот
        map<uint8_t, int> frequencies;
        size_t idx = 0;
        for (int i = 0; i < 256; ++i) {
            int freq = (encodedData[idx] << 24) | (encodedData[idx + 1] << 16) |
                (encodedData[idx + 2] << 8) | encodedData[idx + 3];
            idx += 4;
            if (freq > 0) {
                frequencies[i] = freq;
            }
        }

        // Читаем длину битовой строки
        uint32_t bitLength = (encodedData[idx] << 24) | (encodedData[idx + 1] << 16) |
            (encodedData[idx + 2] << 8) | encodedData[idx + 3];
        idx += 4;

        // Восстанавливаем дерево
        buildTree(frequencies);

        // Декодируем данные
        string currentCode;
        size_t bitIndex = 0;

        while (bitIndex < bitLength) {
            // Получаем очередной бит
            size_t byteIndex = idx + (bitIndex / 8);
            if (byteIndex >= encodedData.size()) break;

            int bitPos = 7 - (bitIndex % 8);
            char bit = ((encodedData[byteIndex] >> bitPos) & 1) ? '1' : '0';
            currentCode += bit;
            bitIndex++;

            // Проверяем, есть ли такой код в таблице
            auto it = decodeTable.find(currentCode);
            if (it != decodeTable.end()) {
                result.push_back(it->second);
                currentCode.clear();
            }
        }

        return result;
    }
};

// Реализация методов BitGrid

BitGrid::BitGrid() : m_width(0), m_height(0) {}

BitGrid::BitGrid(int width, int height)
    : m_width(width), m_height(height) {
    m_data.resize(byteSize(), 0);
}

BitGrid::BitGrid(const cv::Mat& edgeImage) {
    if (edgeImage.empty()) {
        m_width = 0;
        m_height = 0;
        return;
    }

    cv::Mat binary;
    if (edgeImage.channels() == 3) {
        cv::cvtColor(edgeImage, binary, cv::COLOR_BGR2GRAY);
    }
    else {
        binary = edgeImage.clone();
    }

    cv::threshold(binary, binary, 127, 255, cv::THRESH_BINARY);

    m_width = binary.cols;
    m_height = binary.rows;
    m_data.resize(byteSize(), 0);

    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            bool value = binary.at<uint8_t>(y, x) > 0;
            set(x, y, value);
        }
    }
}

bool BitGrid::get(int x, int y) const {
    if (x < 0 || x >= m_width || y < 0 || y >= m_height) {
        return false;
    }
    int index = y * m_width + x;
    return getInternal(index);
}

void BitGrid::set(int x, int y, bool value) {
    if (x < 0 || x >= m_width || y < 0 || y >= m_height) {
        return;
    }
    int index = y * m_width + x;
    setInternal(index, value);
}

void BitGrid::clear() {
    fill(m_data.begin(), m_data.end(), 0);
}

cv::Mat BitGrid::toImage() const {
    cv::Mat image(m_height, m_width, CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            if (get(x, y)) {
                image.at<uint8_t>(y, x) = 255;
            }
        }
    }

    return image;
}

vector<uint8_t> BitGrid::toBytes() const {
    vector<uint8_t> result;

    int dataSize = byteSize();
    result.resize(8 + dataSize);

    result[0] = (m_width >> 0) & 0xFF;
    result[1] = (m_width >> 8) & 0xFF;
    result[2] = (m_width >> 16) & 0xFF;
    result[3] = (m_width >> 24) & 0xFF;

    result[4] = (m_height >> 0) & 0xFF;
    result[5] = (m_height >> 8) & 0xFF;
    result[6] = (m_height >> 16) & 0xFF;
    result[7] = (m_height >> 24) & 0xFF;

    copy(m_data.begin(), m_data.end(), result.begin() + 8);

    return result;
}

void BitGrid::fromBytes(const vector<uint8_t>& data, int width, int height) {
    m_width = width;
    m_height = height;

    int expectedSize = byteSize();

    if (static_cast<int>(data.size()) < expectedSize) {
        cerr << "Error: Data size mismatch!" << endl;
        return;
    }

    m_data.assign(data.begin(), data.begin() + expectedSize);
}

void BitGrid::resize(int width, int height) {
    if (width <= 0 || height <= 0) {
        return;
    }

    BitGrid newGrid(width, height);

    int minWidth = min(m_width, width);
    int minHeight = min(m_height, height);

    for (int y = 0; y < minHeight; ++y) {
        for (int x = 0; x < minWidth; ++x) {
            newGrid.set(x, y, get(x, y));
        }
    }

    m_width = width;
    m_height = height;
    m_data = newGrid.m_data;
}

BitGrid BitGrid::operator&(const BitGrid& other) const {
    if (m_width != other.m_width || m_height != other.m_height) {
        return BitGrid();
    }

    BitGrid result(m_width, m_height);

    for (size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] = m_data[i] & other.m_data[i];
    }

    return result;
}

BitGrid BitGrid::operator|(const BitGrid& other) const {
    if (m_width != other.m_width || m_height != other.m_height) {
        return BitGrid();
    }

    BitGrid result(m_width, m_height);

    for (size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] = m_data[i] | other.m_data[i];
    }

    return result;
}

BitGrid BitGrid::operator~() const {
    BitGrid result(m_width, m_height);

    for (size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] = ~m_data[i];
    }

    return result;
}

int BitGrid::countTrue() const {
    int count = 0;

    for (int i = 0; i < size(); ++i) {
        if (getInternal(i)) {
            ++count;
        }
    }

    return count;
}

float BitGrid::density() const {
    if (size() == 0) {
        return 0.0f;
    }
    return static_cast<float>(countTrue()) / size();
}

vector<uint8_t> BitGrid::compress(CompressionMethod method) const {
    switch (method) {
    case COMPRESSION_RLE:
        return compressRLE();
    case COMPRESSION_LZ4:
        return compressLZ4();
    case COMPRESSION_HUFFMAN:
        return compressHuffman();
    default:
        return toBytes();
    }
}

bool BitGrid::decompress(const vector<uint8_t>& compressedData) {
    if (compressedData.size() < 1) {
        return false;
    }

    CompressionMethod method = static_cast<CompressionMethod>(compressedData[0]);
    vector<uint8_t> data(compressedData.begin() + 1, compressedData.end());

    switch (method) {
    case COMPRESSION_RLE:
        return decompressRLE(data);
    case COMPRESSION_LZ4:
        return decompressLZ4(data);
    case COMPRESSION_HUFFMAN:
        return decompressHuffman(data);
    default:
        if (data.size() >= 8) {
            int width = (data[3] << 24) | (data[2] << 16) | (data[1] << 8) | data[0];
            int height = (data[7] << 24) | (data[6] << 16) | (data[5] << 8) | data[4];
            fromBytes(vector<uint8_t>(data.begin() + 8, data.end()), width, height);
            return true;
        }
        return false;
    }
}

vector<uint8_t> BitGrid::compressRLE() const {
    vector<uint8_t> result;

    result.push_back(COMPRESSION_RLE);

    result.push_back((m_width >> 0) & 0xFF);
    result.push_back((m_width >> 8) & 0xFF);
    result.push_back((m_width >> 16) & 0xFF);
    result.push_back((m_width >> 24) & 0xFF);
    result.push_back((m_height >> 0) & 0xFF);
    result.push_back((m_height >> 8) & 0xFF);
    result.push_back((m_height >> 16) & 0xFF);
    result.push_back((m_height >> 24) & 0xFF);

    int totalBits = size();
    int i = 0;

    while (i < totalBits) {
        bool currentBit = getInternal(i);
        int count = 1;

        while (i + count < totalBits &&
            getInternal(i + count) == currentBit &&
            count < 255) {
            count++;
        }

        result.push_back(currentBit ? 1 : 0);
        result.push_back(count);

        i += count;
    }

    return result;
}

bool BitGrid::decompressRLE(const vector<uint8_t>& data) {
    if (data.size() < 8) {
        return false;
    }

    int width = (data[3] << 24) | (data[2] << 16) | (data[1] << 8) | data[0];
    int height = (data[7] << 24) | (data[6] << 16) | (data[5] << 8) | data[4];

    m_width = width;
    m_height = height;
    m_data.resize(byteSize(), 0);

    int bitIndex = 0;
    size_t dataIndex = 8;

    while (dataIndex + 1 < data.size() && bitIndex < size()) {
        bool value = data[dataIndex] != 0;
        int count = data[dataIndex + 1];

        for (int i = 0; i < count && bitIndex < size(); ++i) {
            setInternal(bitIndex, value);
            bitIndex++;
        }

        dataIndex += 2;
    }

    return true;
}

vector<uint8_t> BitGrid::compressLZ4() const {
    vector<uint8_t> result;

    result.push_back(COMPRESSION_LZ4);

    auto bytes = toBytes();
    auto compressed = LZ4Simple::compress(bytes);

    result.push_back((m_width >> 0) & 0xFF);
    result.push_back((m_width >> 8) & 0xFF);
    result.push_back((m_width >> 16) & 0xFF);
    result.push_back((m_width >> 24) & 0xFF);
    result.push_back((m_height >> 0) & 0xFF);
    result.push_back((m_height >> 8) & 0xFF);
    result.push_back((m_height >> 16) & 0xFF);
    result.push_back((m_height >> 24) & 0xFF);

    result.insert(result.end(), compressed.begin(), compressed.end());

    return result;
}

bool BitGrid::decompressLZ4(const vector<uint8_t>& data) {
    if (data.size() < 8) {
        return false;
    }

    int width = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
    int height = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7];

    vector<uint8_t> compressed(data.begin() + 8, data.end());
    auto decompressed = LZ4Simple::decompress(compressed);

    m_width = width;
    m_height = height;
    m_data = decompressed;

    return true;
}

vector<uint8_t> BitGrid::compressHuffman() const {
    vector<uint8_t> result;

    result.push_back(COMPRESSION_HUFFMAN);

    auto bytes = toBytes();
    HuffmanCoder coder;
    auto compressed = coder.encode(bytes);

    result.push_back((m_width >> 0) & 0xFF);
    result.push_back((m_width >> 8) & 0xFF);
    result.push_back((m_width >> 16) & 0xFF);
    result.push_back((m_width >> 24) & 0xFF);
    result.push_back((m_height >> 0) & 0xFF);
    result.push_back((m_height >> 8) & 0xFF);
    result.push_back((m_height >> 16) & 0xFF);
    result.push_back((m_height >> 24) & 0xFF);

    result.insert(result.end(), compressed.begin(), compressed.end());

    return result;
}

bool BitGrid::decompressHuffman(const vector<uint8_t>& data) {
    if (data.size() < 8) {
        return false;
    }

    int width = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
    int height = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7];

    vector<uint8_t> compressed(data.begin() + 8, data.end());
    HuffmanCoder coder;
    auto decompressed = coder.decode(compressed);

    m_width = width;
    m_height = height;
    m_data = decompressed;

    return true;
}

BitGrid::CompressionInfo BitGrid::getCompressionInfo(const vector<uint8_t>& compressedData) const {
    CompressionInfo info;
    info.originalSize = byteSize();
    info.compressedSize = compressedData.size();
    info.ratio = (info.originalSize > 0) ?
        (float)info.compressedSize / info.originalSize : 0.0f;

    if (!compressedData.empty()) {
        info.method = static_cast<CompressionMethod>(compressedData[0]);
    }
    else {
        info.method = COMPRESSION_NONE;
    }

    return info;
}

void BitGrid::save(const string& filename, CompressionMethod method) const {
    auto compressed = compress(method);
    ofstream file(filename, ios::binary);

    if (file) {
        file.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
        file.close();
    }
}

void BitGrid::load(const string& filename) {
    ifstream file(filename, ios::binary | ios::ate);

    if (!file) {
        return;
    }

    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    vector<uint8_t> buffer(size);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        decompress(buffer);
    }

    file.close();
}

int BitGrid::calculateByteIndex(int bitIndex) const {
    return bitIndex / 8;
}

uint8_t BitGrid::calculateBitMask(int bitIndex) const {
    return 1 << (bitIndex % 8);
}

void BitGrid::setInternal(int index, bool value) {
    int byteIndex = calculateByteIndex(index);
    uint8_t bitMask = calculateBitMask(index);

    if (value) {
        m_data[byteIndex] |= bitMask;
    }
    else {
        m_data[byteIndex] &= ~bitMask;
    }
}

bool BitGrid::getInternal(int index) const {
    int byteIndex = calculateByteIndex(index);
    if (byteIndex >= static_cast<int>(m_data.size())) {
        return false;
    }
    uint8_t bitMask = calculateBitMask(index);
    return (m_data[byteIndex] & bitMask) != 0;
}

void BitGrid::analyzeCompressionStats() const {
    // Реализация пуста, так как метод не используется
}
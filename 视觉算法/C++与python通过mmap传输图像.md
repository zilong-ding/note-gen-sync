# C++与python通过mmap机制传输图像

测试环境：同一台电脑上，ubuntu24

## C++端代码实现

### 1.定义数据结构

```C++
// 帧头结构（固定在共享内存开头）
struct FrameHeader {
    int width;
    int height;
    int channels;
    int64_t timestamp_ms;
    int frame_valid;  // 必须是最后一个字段，用于同步
} __attribute__((packed)); // 强制无填充，保证结构体大小精确为 24 字节
```

### 2.确定数据大小

```c++
    auto frame = cv::imread(imagePath);
    auto width = frame.cols;
    auto height = frame.rows;
    std::cout << "width: " << width << std::endl;
    std::cout << "height: " << height << std::endl;

    size_t frame_size_ = width * height * 3 + sizeof(FrameHeader);;
    std::cout << "frame_size_: " << frame_size_ << std::endl;
```

### 3.创建共享内存对象

```c++
    int shm_fd_;
    std::string shm_name_ = "/mmaptest_shm";
    // ✅ 关键修复：确保没有残留的旧文件
    shm_unlink(shm_name_.c_str()); // 删除旧的（即使不存在也不报错）
    shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_ == -1) {
        std::cerr << "shm_open failed: " << strerror(errno) << "\n";
        return false;
    }
    void* shm_ptr_;
    ftruncate(shm_fd_, frame_size_);


    shm_ptr_ = mmap(nullptr, frame_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (shm_ptr_ == MAP_FAILED) {
        std::cerr << " mmap failed: " << strerror(errno) << "\n";
        return false;
    }
```


### 4.传输数据

```c++
    auto* header = static_cast<FrameHeader*>(shm_ptr_);
    char* data_start = static_cast<char*>(shm_ptr_) + sizeof(FrameHeader);
    while (1) {
        // Step 1: 标记为“正在写入” → 防止 Python 读到一半
        header->frame_valid = 0;

        // Step 2: 更新元信息
        header->width = frame.cols;
        header->height = frame.rows;
        header->channels = frame.channels();
        // header->timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        //     std::chrono::system_clock::now().time_since_epoch()).count();
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        std::cout << timestamp_ms << std::endl;
        header->timestamp_ms = timestamp_ms;
        std::cout << header->timestamp_ms << " ms" << std::endl;

        // Step 3: 写入图像数据（可能耗时几微秒）
        memcpy(data_start, frame.data, frame.total() * frame.elemSize());

        // Step 4: 标记为“已完成” → Python 可安全读取
        __sync_synchronize(); // 内存屏障，防止指令重排序（重要！）
        header->frame_valid = 1; // ✅ 最后一步！原子性标志

        cv::imshow("Result of Detection", frame);
        cv::waitKey(1);
    }
```


## python端代码实现

### 1.创建共享内存对象

```python
        # 计算总大小：头 + 图像数据
        header_size = 4 * 4 + 8 # 5 ints = 24 bytes
        total_size = width * height * channels + header_size # 775704

        fd = os.open(shm_name, os.O_RDONLY)
        mm = mmap.mmap(fd, total_size, mmap.MAP_SHARED, mmap.PROT_READ)
```

### 2.解析数据

```python
            header = np.frombuffer(mm[:12], dtype=np.int32, count=3)
            receivetime = np.frombuffer(mm[12:20], dtype=np.int64, count=1)[0]
            frame_valid = np.frombuffer(mm[20:24], dtype=np.int32, count=1)
        
            # 解析其他字段
            w, h, c = int(header[0]), int(header[1]), int(header[2])

            # 验证分辨率一致性
            if w != width or h != height or c != channels:
                print(f"[WARN] Resolution mismatch: expected {width}x{height}, got {w}x{h}")
                mm.close()
                os.close(fd)
                return None

            # 读取图像数据（从第20字节开始）
            img_data = np.frombuffer(mm[24:], dtype=np.uint8).reshape((h, w, c))
```

## 关键点

要清楚数据结构中每一部分所占多少字节好方便解析

## 完整代码

### C++

```c++
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <opencv2/opencv.hpp>

// 帧头结构（固定在共享内存开头）
struct FrameHeader {
    int width;
    int height;
    int channels;
    int64_t timestamp_ms;
    int frame_valid;  // 必须是最后一个字段，用于同步
} __attribute__((packed)); // 强制无填充，保证结构体大小精确为 20 字节


int main(int argc, char** argv) {
    std::cout << "hello mmaptest" << std::endl;
    std::string imagePath = "/home/dzl/CLionProjects/onnxtest/Images/000000014439.jpg";
    auto frame = cv::imread(imagePath);
    auto width = frame.cols;
    auto height = frame.rows;
    std::cout << "width: " << width << std::endl;
    std::cout << "height: " << height << std::endl;

    size_t frame_size_ = width * height * 3 + sizeof(FrameHeader);;
    std::cout << "frame_size_: " << frame_size_ << std::endl;
    int shm_fd_;
    std::string shm_name_ = "/mmaptest_shm";
    // ✅ 关键修复：确保没有残留的旧文件
    shm_unlink(shm_name_.c_str()); // 删除旧的（即使不存在也不报错）
    shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_ == -1) {
        std::cerr << "shm_open failed: " << strerror(errno) << "\n";
        return false;
    }
    void* shm_ptr_;
    ftruncate(shm_fd_, frame_size_);


    shm_ptr_ = mmap(nullptr, frame_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (shm_ptr_ == MAP_FAILED) {
        std::cerr << " mmap failed: " << strerror(errno) << "\n";
        return false;
    }

    auto* header = static_cast<FrameHeader*>(shm_ptr_);
    char* data_start = static_cast<char*>(shm_ptr_) + sizeof(FrameHeader);
    while (1) {
        // Step 1: 标记为“正在写入” → 防止 Python 读到一半
        header->frame_valid = 0;

        // Step 2: 更新元信息
        header->width = frame.cols;
        header->height = frame.rows;
        header->channels = frame.channels();
        // header->timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        //     std::chrono::system_clock::now().time_since_epoch()).count();
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        std::cout << timestamp_ms << std::endl;
        header->timestamp_ms = timestamp_ms;
        std::cout << header->timestamp_ms << " ms" << std::endl;

        // Step 3: 写入图像数据（可能耗时几微秒）
        memcpy(data_start, frame.data, frame.total() * frame.elemSize());

        // Step 4: 标记为“已完成” → Python 可安全读取
        __sync_synchronize(); // 内存屏障，防止指令重排序（重要！）
        header->frame_valid = 1; // ✅ 最后一步！原子性标志

        cv::imshow("Result of Detection", frame);
        cv::waitKey(1);
    }
    cv::destroyAllWindows();
    return 0;
}
```



### python

```python
import numpy as np
import cv2
import mmap
import os
from typing import Optional
import time

def read_frame_from_shm(shm_name: str, width: int, height: int, channels: int = 3) -> Optional[np.ndarray]:
    """
    从共享内存读取一帧图像，零拷贝，带 frame_valid 同步
    """
    try:
        # 计算总大小：头 + 图像数据
        header_size = 4 * 4 + 8 # 5 ints = 24 bytes
        total_size = width * height * channels + header_size # 775704

        fd = os.open(shm_name, os.O_RDONLY)
        mm = mmap.mmap(fd, total_size, mmap.MAP_SHARED, mmap.PROT_READ)

        # 轮询直到 frame_valid == 1
        while True:
            header = np.frombuffer(mm[:12], dtype=np.int32, count=3)
            receivetime = np.frombuffer(mm[12:20], dtype=np.int64, count=1)[0]
            frame_valid = np.frombuffer(mm[20:24], dtype=np.int32, count=1)
        
            # 解析其他字段
            w, h, c = int(header[0]), int(header[1]), int(header[2])

            # 验证分辨率一致性
            if w != width or h != height or c != channels:
                print(f"[WARN] Resolution mismatch: expected {width}x{height}, got {w}x{h}")
                mm.close()
                os.close(fd)
                return None

            # 读取图像数据（从第20字节开始）
            img_data = np.frombuffer(mm[24:], dtype=np.uint8).reshape((h, w, c))
            # ✅ 当前系统时间（秒）转成毫秒，与 C++ 对齐
            current_ms = time.time() * 1000 // 1  # 转为毫秒整数
            timestamp_ms = receivetime
            # ✅ 计算延迟：毫秒级
            delay_ms = current_ms - timestamp_ms

            # ✅ 格式化输出：清晰标注单位
            print(f"[INFO] Received frame at {current_ms:,} ms, "
                  f"[INFO] Sent at {timestamp_ms:,} ms, "
                  f"delay: {delay_ms:,} ms")
            time.sleep(0.001)  # 等待 1ms，避免忙等待
            mm.close()
            os.close(fd)
            return img_data

    except Exception as e:
        print(f"[ERROR] Read from {shm_name}: {e}")
        return None

if __name__ == "__main__":
    while True:
        img = read_frame_from_shm("/dev/shm/mmaptest_shm", 640, 404)
        if img is not None:
            cv2.imshow("frame", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            time.sleep(0.1)  # 如果没读到，稍等再试
    cv2.destroyAllWindows()
```

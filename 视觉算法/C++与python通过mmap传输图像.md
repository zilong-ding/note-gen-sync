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



## python端代码实现

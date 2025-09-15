# 使用onnxruntime-gpu C++部署

参考文章：https://jinscott.medium.com/onnx-runtime-on-c-67f69de9b95c

## 查看推理引擎

```C++
#include "onnxruntime_cxx_api.h"
#include <iostream>

void check_providers() {
    //check providers
    auto providers = Ort::GetAvailableProviders();
    for (const auto& provider : providers) {
        std::cout << provider << std::endl;
    }
}
```

* **作用**：定义一个名为 `check_providers` 的函数，用于检测并打印当前系统中 ONNX Runtime 可用的执行提供程序。
* **为什么需要这个函数？**
  * ONNX Runtime 支持多种后端执行引擎（如 CPU、CUDA、DirectML、TensorRT、OpenVINO 等），但并非所有都默认启用。
  * 了解哪些提供程序可用，有助于你：
    * 选择最优推理加速方式（如 GPU 加速）
    * 排查模型无法在特定设备上运行的问题
    * 验证是否成功安装了 GPU 版本的 ONNX Runtime


## 加载ONNX模型

```C++
int init_onnx(const std::string& model_path = "/home/dzl/CLionProjects/onnxtest/models/Detect/yolov8s.onnx") {
    // 1. 创建环境
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "YOLOv8");

    // 2. 设置会话选项
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetInterOpNumThreads(0);   // 自动选择
    sessionOptions.SetIntraOpNumThreads(0);   // 自动选择
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); // 启用优化！

    // 3. 尝试添加 CUDA 提供程序
    bool use_cuda = true;
    if (use_cuda) {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic; // 生产推荐
        cuda_options.arena_extend_strategy = 0;
        cuda_options.do_copy_in_default_stream = 0;

        bool cuda_added = sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        if (!cuda_added) {
            std::cerr << "[WARNING] Failed to initialize CUDA provider. Using CPU only.\n";
        } else {
            std::cout << "[INFO] CUDA execution provider successfully added.\n";
        }
    }

    // 4. 加载模型
    try {
        Ort::Session session(env, model_path.c_str(), sessionOptions);
        std::cout << "[SUCCESS] ONNX model loaded successfully.\n";
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Exception: " << e.what() << " (Code: " << e.GetOrtErrorCode() << ")\n";
        return -1;
    }

    // 5. （可选）创建 MemoryInfo —— 如果后续要用，现在创建
    // Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // ... 保存到全局或返回 ...

    return 0;
}
```

# 使用onnxruntime-gpu C++部署

参考文章：https://jinscott.medium.com/onnx-runtime-on-c-67f69de9b95c

## 查看推理引擎

```C++

```



* **作用**：定义一个名为 `check_providers` 的函数，用于检测并打印当前系统中 ONNX Runtime 可用的执行提供程序。
* **为什么需要这个函数？**
  * ONNX Runtime 支持多种后端执行引擎（如 CPU、CUDA、DirectML、TensorRT、OpenVINO 等），但并非所有都默认启用。
  * 了解哪些提供程序可用，有助于你：
    * 选择最优推理加速方式（如 GPU 加速）
    * 排查模型无法在特定设备上运行的问题
    * 验证是否成功安装了 GPU 版本的 ONNX Runtime

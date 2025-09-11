# 🤖 ROS2 Jazzy 开发环境搭建全攻略：Ubuntu 24.10 + PyCharm/CLion 高效开发

**副标题：** 告别环境配置烦恼，手把手教你搭建 Python 和 C++ 双语言 ROS2 开发环境。

对于机器人开发者而言，一个顺手的开发环境是效率的基石。本文将基于最新的 **Ubuntu 24.10** 和 **ROS2 Jazzy**，详细介绍如何使用 **PyCharm** 和 **CLion** 这两款强大的 IDE 进行 ROS2 开发，助你事半功倍！

---

## 一、系统与环境准备

**1. 安装 ROS2 Jazzy**

我们推荐使用国内镜像源进行安装，速度更快更稳定。打开终端，执行以下命令：

```bash
wget http://fishros.com/install -O fishros && . fishros
```

> **小贴士**：如果你的默认 Shell 是 `zsh`，请先下载脚本，然后在 `bash` 环境下运行：
>
> ```bash
> bash fishros
> ```

安装完成后，记得在终端中执行 `source /opt/ros/jazzy/setup.bash` 来激活环境。为了方便，可以将此命令添加到你的 `~/.bashrc` 或 `~/.zshrc` 文件中。

---

## 二、Python 开发环境：PyCharm 配置

**1. 创建 ROS2 Python 包**

首先，按照官方教程创建一个简单的 Publisher/Subscriber 包。

```bash
# 进入你的工作空间（假设为 ~/ros2_ws）
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python --license Apache-2.0 py_pubsub
```

接着，下载官方示例代码：

```bash
cd ~/ros2_ws/src/py_pubsub/py_pubsub
wget https://raw.githubusercontent.com/ros2/examples/jazzy/rclpy/topics/minimal_publisher/examples_rclpy_minimal_publisher/publisher_member_function.py
wget https://raw.githubusercontent.com/ros2/examples/jazzy/rclpy/topics/minimal_subscriber/examples_rclpy_minimal_subscriber/subscriber_member_function.py
```

然后，按照教程修改 `package.xml` 和 `setup.py` 文件，添加依赖和入口点。

**2. PyCharm 项目配置**

* **打开项目**：启动 PyCharm，选择 “Open” 或 “Open Folder”，定位到你的工作空间 `src` 目录 (`~/ros2_ws/src`)。
* **配置 Python 解释器**：
  1. 进入 `File > Settings > Project: src > Python Interpreter`。
  2. 点击右上角的齿轮图标，选择 “Add...”。
  3. 选择 “System Interpreter”，然后找到你的系统 Python (通常是 `/usr/bin/python3`)。**关键一步：不要使用 Conda 或 Virtualenv 环境！** ROS2 的系统包与这些环境可能存在兼容性问题。

     ![2025-09-10_16-20.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/853e1cea-aec5-4fe2-bf47-13e9aeac0953.jpeg)
* **添加 ROS2 库路径**：
  1. 在同一个 “Python Interpreter” 页面，点击右下角的 “Show All...”。
  2. 选中你刚添加的解释器，点击右边的文件夹图标。
  3. 点击 “+” 号，添加 ROS2 的 Python 库路径：`/opt/ros/jazzy/lib/python3.12/site-packages`。

     ![2025-09-10_16-23.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/134c3c4a-bef0-4896-89f0-bf405b93de42.jpeg)
* **设置环境变量**：
  1. 进入 `Run > Edit Configurations...`。
  2. 点击 “+” 号，选择 “Python”。
  3. 在 “Script path” 中选择你的 `publisher_member_function.py` 或 `subscriber_member_function.py`。
  4. 在 “Environment variables” 中，点击文件夹图标，添加一个新的环境变量：![2025-09-11_09-29.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/ce444fba-6fe7-42cc-8944-a13a803cfce7.jpeg)

完成以上配置后，你就可以直接在 PyCharm 里点击运行按钮，像运行普通 Python 脚本一样启动你的 ROS2 节点了！

---

## 三、C++ 开发环境：CLion 配置

**1. 创建 ROS2 C++ 包**

同样，我们先创建一个 C++ 包。

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake cpp_pubsub
```

下载官方 C++ 示例代码：

```bash
cd ~/ros2_ws/src/cpp_pubsub/src
wget -O publisher_lambda_function.cpp https://raw.githubusercontent.com/ros2/examples/jazzy/rclcpp/topics/minimal_publisher/lambda.cpp
```

确保 `CMakeLists.txt` 内容如下：

```cmake
cmake_minimum_required(VERSION 3.5)
project(cpp_pubsub)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(talker src/publisher_lambda_function.cpp)
ament_target_dependencies(talker rclcpp std_msgs)

install(TARGETS
  talker
  DESTINATION lib/${PROJECT_NAME})

ament_package()
```

**2. 解决常见编译错误**

在终端中尝试编译：

```bash
cd ~/ros2_ws
colcon build --packages-select cpp_pubsub
```

你可能会遇到类似 `No module named 'catkin_pkg'` 的错误。这是因为 ROS2 的构建工具依赖于系统 Python 的一些包，而你可能不小心在 Conda 环境中执行了命令。

**解决方法**：

```bash
# 1. 退出所有 Conda 环境
conda deactivate

# 2. 删除之前失败的构建产物
rm -rf build/ install/ log/

# 3. 安装缺失的系统依赖
sudo apt update && sudo apt install python3-catkin-pkg python3-lxml python3-pkg-resources

# 4. 重新编译
colcon build --packages-select cpp_pubsub
```

**3. CLion 项目配置**

为了让 CLion 能完美理解 ROS2 的 `colcon` 工作空间，我们需要一个顶层的 `CMakeLists.txt`。

* **创建顶层 CMakeLists.txt**：

  ```bash
  # 下载一个社区维护的顶层 CMakeLists.txt 脚本
  sudo mkdir -p /opt/ros/scripts/cmake
  sudo git clone https://github.com/kai-waang/colcon-toplevel-cmake /opt/ros/scripts/cmake

  # 将其复制到你的工作空间根目录
  cd ~/ros2_ws
  cp /opt/ros/scripts/cmake/toplevel.cmake ./CMakeLists.txt
  ```
* **从终端启动 CLion**：

  ```bash
  # 确保在工作空间根目录，并已 source 环境
  cd ~/ros2_ws
  source install/setup.bash # 或 source install/setup.zsh
  clion .
  ```
* **配置 CLion 项目**：

  1. 首次打开，CLion 会提示加载 CMake 项目。在 “Build directory” 中，将其设置为 `build`。

     ![2025-09-11_09-13.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/043a758a-2079-4fd7-a466-ff44f23af457.jpeg)
  2. 进入 `File > Settings > Build, Execution, Deployment > Toolchains`。
  3. 在 “Environment” 选项卡下，勾选 “Load from file”，然后选择你工作空间下的环境配置文件：`~/ros2_ws/install/setup.bash` (或 `.zsh`)。

     ![2025-09-11_09-14.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/5463c0e3-7fb7-4a65-aaec-4f9ea6445c12.jpeg)
  4. **（可选）精细化编译控制**：你可以编辑顶层的 `CMakeLists.txt` 文件，通过设置 `COLCON_PACKAGES` 变量来指定只编译特定的包，效果等同于 `colcon build --packages-select`。

完成配置后，你就可以在 CLion 中享受代码高亮、智能补全、一键编译和图形化调试的畅快体验了！

---

## 四、总结

至此，一个功能完备的 ROS2 Jazzy 开发环境就搭建完成了！无论是偏爱 Python 的快速原型开发，还是追求 C++ 的极致性能，PyCharm 和 CLion 都能为你提供强大的支持。

**关键要点回顾**：

* **环境隔离**：尽量使用系统 Python，避免 Conda/Venv 与 ROS2 的潜在冲突。
* **路径配置**：为 IDE 正确配置 ROS2 的库路径和环境变量是成功的关键。
* **工具链集成**：利用顶层 `CMakeLists.txt` 让 CLion 无缝集成 `colcon` 工作流。

赶紧动手试试吧！让高效的工具助你在机器人开发的道路上加速前行！🚀

---

**欢迎关注本公众号，获取更多 ROS2、机器人和 AI 领域的前沿技术与实战教程！**

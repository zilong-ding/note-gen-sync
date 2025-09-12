# ROS2安装和IDE设置

ubuntu24.10

ROS版本 jazzy

python版本 3.12

## ROS2安装

```bash
wget http://fishros.com/install -O fishros && . fishros

```

如果是zsh下载之后再执行`bash fishros`

## IDE设置

### PyCharm

参考文章：https://zhuanlan.zhihu.com/p/552211633

首先根据网页https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html

创建ros2 python包，然后使用pycharm打开 src文件夹选择python解释器

![2025-09-10_16-20.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/853e1cea-aec5-4fe2-bf47-13e9aeac0953.jpeg)

然后设置ROS2 python库路径

![2025-09-10_16-23.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/134c3c4a-bef0-4896-89f0-bf405b93de42.jpeg)

然后添加环境变量

![2025-09-11_09-29.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/ce444fba-6fe7-42cc-8944-a13a803cfce7.jpeg)

然后就可以运行了

![2025-09-10_16-41.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/704c8fb7-07e2-46a7-89f0-66122896700a.jpeg)

对应python代码如下

```python
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    try:
        rclpy.init(args=args)
        minimal_publisher = MinimalPublisher()
        rclpy.spin(minimal_publisher)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
```

### Clion

参考文章：https://www.jetbrains.com/help/clion/ros2-tutorial.html

参考文章：https://zhuanlan.zhihu.com/p/693626476

首先根据网页https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html创建`ament_cmake`类型的功能包

然后使用clion打开src文件夹

下面是cmakelists文件中的内容

```cmake
cmake_minimum_required(VERSION 3.5)
project(cpp_pubsub)

# Default to C++14
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

下面是publisher_lambda_function.cpp中的内容

```cpp
//
// Created by dzl on 9/10/25.
//
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalSubscriber : public rclcpp::Node
{
public:
    MinimalSubscriber()
    : Node("minimal_subscriber")
    {
        auto topic_callback =
          [this](std_msgs::msg::String::UniquePtr msg) -> void {
              RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
        };
        subscription_ =
          this->create_subscription<std_msgs::msg::String>("topic", 10, topic_callback);
    }

private:
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalSubscriber>());
    rclcpp::shutdown();
    return 0;
}
```

关闭之后编译

遇到下面问题

```bash
    ~/ros2_ws  colcon build                                                           ✔  at 19:09:26  
Starting >>> cpp_pubsub
Starting >>> py_pubsub
--- stderr: cpp_pubsub                                                
Traceback (most recent call last):
  File "/opt/ros/jazzy/share/ament_cmake_core/cmake/core/package_xml_2_cmake.py", line 22, in <module>
    from catkin_pkg.package import parse_package_string
ModuleNotFoundError: No module named 'catkin_pkg'
CMake Error at /opt/ros/jazzy/share/ament_cmake_core/cmake/core/ament_package_xml.cmake:95 (message):
  execute_process(/home/dzl/anaconda3/bin/python3
  /opt/ros/jazzy/share/ament_cmake_core/cmake/core/package_xml_2_cmake.py
  /home/dzl/ros2_ws/src/cpp_pubsub/package.xml
  /home/dzl/ros2_ws/build/cpp_pubsub/ament_cmake_core/package.cmake) returned
  error code 1
Call Stack (most recent call first):
  /opt/ros/jazzy/share/ament_cmake_core/cmake/core/ament_package_xml.cmake:49 (_ament_package_xml)
  /opt/ros/jazzy/share/ament_cmake_core/cmake/core/ament_package.cmake:63 (ament_package_xml)
  CMakeLists.txt:24 (ament_package)


---
Failed   <<< cpp_pubsub [0.29s, exited with code 1]
Aborted  <<< py_pubsub [0.33s]             

Summary: 0 packages finished [0.41s]
  1 package failed: cpp_pubsub
  1 package aborted: py_pubsub
  1 package had stderr output: cpp_pubsub

```

这里主要原因是之前构建py_pubsub的时候实在conda环境下构建的，然后ros2对conda支持还是有些问题，这里`conda deactivate`，然后删除`build/`,`install/`,`log/`目录。同时安装`sudo apt install python3-catkin-pkg python3-lxml python3-pkg-resources`，安装完成后`colcon build`成功。

在终端中编译成功后，参考https://blog.csdn.net/qq_45709806/article/details/149025458这边文章，首先是下载安装colcon 顶层cmake

```bash
sudo git clone https://github.com/kai-waang/colcon-toplevel-cmake /opt/ros/scripts/cmake

```

然后切换到工作目录下

```bash
cd ~/ros2_ws # 此处替换为自己的工作空间

```

然后拷贝文件到当前目录

```bash
cp /opt/ros/scripts/cmake/toplevel.cmake ./CMakeLists.txt

```

编译工作空间，从命令行启动CLion

```bash
colcon build	# 这一步是为了生成 install/setup.zsh
clion .

```

第一次用CLion打开这个工作空间，会弹出下面的窗口，选择对应配置，输出编译路径设置为build

![2025-09-11_09-13.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/043a758a-2079-4fd7-a466-ff44f23af457.jpeg)

进入 `Settings > Build, Execution, Deployment > ToolChains`，在Environment file中选择当前工作空间下的`install/setup.zsh`

![2025-09-11_09-14.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/5463c0e3-7fb7-4a65-aaec-4f9ea6445c12.jpeg)

通过修改顶层的`CMakeLists.txt`，可以控制只编译某几个包，类似于`colcon build --packages-select xxx<span> </span>`，我的顶层的`CMakeLists.txt`修改如下：

![2025-09-11_09-15.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/b500daf2-1c12-4df8-90cd-2e7629e1ed3a.jpeg)

接下来就可以正常编译、运行和调试了

### QT-creator

参考文章：https://blog.csdn.net/qq_44940689/article/details/138165085

参考文章：https://blog.csdn.net/qq_40326539/article/details/134578275

参考文章：https://github.com/ros-industrial/ros_qtc_plugin

#### 首先安装qt6.7以上版本

这里是使用的qt6.9.0

安装好之后查看qt_creator的版本，点击help->about qt

![2025-09-12_09-34.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/7bb0051f-4041-422e-814e-dae619933cd2.jpeg)

然后安装插件，注意这里是默认[Qt Creator](https://so.csdn.net/so/search?q=Qt%20Creator&spm=1001.2101.3001.7020)的安装路径为`~/Qt/Tools/QtCreator`，如果你的路径不是这里，请将下面语句`-C`后改为你自己的安装路径。

```bash
curl -SL $(curl -s https://api.github.com/repos/ros-industrial/ros_qtc_plugin/releases/latest | grep -E 'browser_download_url.*qtcreator-plugin-ros-.*-Linux-'$(arch)'.zip' | cut -d'"' -f 4) | bsdtar -xzf - -C ~/Qt/Tools/QtCreator
```

然后在git上面查到对应版本的[release](https://github.com/ros-industrial/ros_qtc_plugin/releases)

![2025-09-12_09-38.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/ae6bfc12-7bd5-4577-888f-b1c03cc7162c.jpeg)


这里我是要找17.0的版本

下载之后解压

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

以上两篇只是参考，实际这次并没有用到


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


#### 遇到的问题

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


这里主要原因是之前构建py_pubsub的时候实在conda环境下构建的，然后ros2对conda支持还是有些问题，这里`conda deactivate`，然后删除`build/`,`install/`,`log/`目录。同时安装``

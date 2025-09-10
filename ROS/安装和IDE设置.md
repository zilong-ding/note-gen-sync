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

```

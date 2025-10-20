# coppeliasim中进行自动手眼标定--眼在手外

## 思路

1. 搭建仿真环境，主要是包含机器人、标定板与相机。将标定板固连到机器人末端
2. 生成标定板在相机下的位姿：斐波那契网格采样，将标定板启动到对应位姿保存对应图像与机器人位姿
3. 标定相机内参、标定相机与机器人之间的相关位姿


## 第一步：搭建仿真环境

### 设置机器人和相机

这里选择的是ABB4600的机器人，导入后删除其自带的示例脚本

![2025-10-19_08-41.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/e678e9ec-ec34-4e8f-a7d2-b6d8c71f9afd.jpeg)

相机设置了两个一个专门负责彩色图，另一个专门负责深度图，当然这里我们只使用彩色图

![2025-10-19_08-46.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/d772614a-b747-4fb5-bf80-770f04d07aab.jpeg)

彩色相机设置

![2025-10-19_08-46_1.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/dd31d755-bd30-44d0-8e22-5476f8e5aa23.jpeg)

深度相机设置

![2025-10-19_08-47.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/8050d62d-06eb-422e-9732-1efe2b4f9805.jpeg)

### 设置标定板

先进入标定板生成网站生成标定板https://calib.io/

![2025-10-20_08-50.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/d01042bd-47fd-4191-be8e-57bc31ee70f4.jpeg)

下载之后转为图片格式。在coppeliasim中设置一个长方体，设置纹理

![2025-10-20_08-52.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/3f6e8626-b865-4b89-a5a0-49f0b56fa7a0.jpeg)

![2025-10-20_08-53.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/abfa8cb5-b191-4ab3-9804-cc2ea6a004fd.jpeg)

这里选择导入纹理，导入后设置AlongU,AlongV为棋盘格整体的长宽

## 第二步：使用斐波那契网格采样的方法生成标定板位姿

$$
z_{n}=(2n-1)/N-1\text{(1)}\\x_{n}=\sqrt{1-z_{n}^{2}}\cdot\cos(2\pi n\phi)\quad(2)\\y_{n}=\sqrt{1-z_{n}^{2}}\cdot\sin(2\pi n\phi)\quad(3)
$$

$$
\text{其中常数}\phi=(\sqrt5-1)/2\approx0.618\text{正是黄金分割比}^+。
$$

![2025-10-19_08-39.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/853c6e52-212b-4b02-965a-ae5411c8cb40.jpeg)

## 第三步：标定相机内参




## 第四步：手眼标定

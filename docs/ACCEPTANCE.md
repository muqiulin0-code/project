# 阶段 0 验收说明

## 验收目标

阶段 0 不是要求记住所有 API，而是证明你能够从零配置环境、阅读文档、编写代码、定位错误，并把结果可复现地提交到 Git。

完成本仓库后，应能独立回答和操作以下内容：

- 使用 Python 虚拟环境隔离项目依赖，并知道 `PYTHONPATH`、`pip`、`venv` 的作用。
- 使用 NumPy 表示矩阵、理解广播和矩阵乘法，并实现旋转矩阵与四元数之间的数学转换。
- 使用 OpenCV 读取摄像头/视频、转换颜色空间、调用 Haar 或 DNN 检测器、绘制结果。
- 使用 PyTorch 构建 `Dataset`/`DataLoader`、搭建 CNN、完成训练/验证/测试、保存最佳权重和训练曲线。
- 使用 C++ 的 RAII、模板、STL、`std::thread`、`std::mutex`、`std::condition_variable` 实现线程安全队列。
- 使用 CMake 的 `add_library`、`add_executable`、`target_link_libraries`、`find_package`、`install` 构建多文件项目。
- 使用 Git 的 `add/commit/branch/push/pull` 管理代码，并通过 README、测试和提交记录展示工作。

## 五项练习及验收方法

| 练习 | 在本仓库中的实现 | 验收命令 | 通过条件 |
| --- | --- | --- | --- |
| PyTorch CIFAR-10 分类器 | `python/cifar10_classifier/` | `python -m cifar10_classifier.train ...` | 官方测试集准确率严格大于 70%，生成 `metrics.json`、`history.csv`、`curves.png` 和最佳权重 |
| OpenCV 实时人脸/物体检测 | `python/opencv_detection/` | 人脸：`python -m opencv_detection --source 0 --mode face`；物体：先下载模型，再用 `--mode object` | 摄像头或视频窗口实时显示带框结果，`q`/`Esc` 可退出，可选输出 MP4 |
| C++ 多线程生产者-消费者 | `cpp/include/producer_consumer/bounded_queue.hpp`，`cpp/src/pipeline.cpp` | `./build/cpp/pc_demo 4 3 5000 16` | 产出数等于消费数，校验和正确，满队列阻塞、关闭队列唤醒等测试通过 |
| CMake 多文件 C++ 项目 | `cpp/CMakeLists.txt` | `cmake -S cpp -B build/cpp && cmake --build build/cpp` | 成功生成静态库、可执行文件、测试程序和安装规则 |
| NumPy 旋转矩阵与四元数转换 | `python/rotation_conversions/` | `python -m rotation_conversions --axis 0 0 1 --angle 90` | 单位四元数、正交旋转矩阵、双向转换和组合运算的随机测试全部通过 |

## 真正需要掌握的内容

### 1. PyTorch 与 CIFAR-10

重点不是某个模型能到多少分，而是掌握以下闭环：

1. 数据集划分：训练集、验证集、官方测试集不能混用。
2. 预处理：通道归一化、随机裁剪、随机翻转只用于训练。
3. 模型结构：卷积、BatchNorm、残差连接、池化、全连接层。
4. 训练循环：前向传播、损失、反向传播、优化器更新。
5. 验证与保存：按验证集选择最佳模型，最后只评估一次测试集。
6. 结果证明：准确率、损失曲线、超参数和环境版本都有记录。

> “准确率 > 70%”应理解为官方 CIFAR-10 测试集上的准确率，而不是训练集准确率。

### 2. OpenCV 检测

需要理解视频流本质上是一帧一帧的图片。检测流程是：

`VideoCapture -> 预处理 -> 检测器 -> 非极大值抑制（如适用）-> 绘制 -> imshow/VideoWriter`

人脸练习使用 OpenCV 自带的 Haar 模型，适合理解流程；物体检测使用 MobileNet-SSD 和 `cv2.dnn`，更接近实际推理流程。无需摄像头时，可传视频路径进行验收。

### 3. C++ 并发

生产者-消费者问题的核心是共享缓冲区。必须保证：

- 缓冲区满时生产者等待，空时消费者等待。
- 条件变量必须在循环谓词下等待，避免虚假唤醒。
- 唤醒发生在释放锁之后或由 `notify_*` 完成，避免不必要的阻塞。
- `close()` 后不再接收新数据，但可以让消费者消费完剩余数据再退出。
- 每个线程只写自己的局部结果，主线程 `join` 后再汇总，避免数据竞争。

### 4. CMake

至少能独立写出：

```cmake
cmake_minimum_required(VERSION 3.22)
project(demo LANGUAGES CXX)
find_package(Threads REQUIRED)
add_library(core ...)
add_executable(demo ...)
target_link_libraries(demo PRIVATE core Threads::Threads)
install(TARGETS demo RUNTIME DESTINATION bin)
```

### 5. NumPy 与 3D 旋转

需要明确记号后再写代码：

- 本仓库使用标量在前的四元数 `[w, x, y, z]`。
- 本仓库使用列向量约定 `v_rotated = R @ v`。
- 四元数 `q` 与 `-q` 表示同一个旋转，因此输出规范化为 `w >= 0`。
- 旋转矩阵必须满足 `R.T @ R = I` 且 `det(R) = 1`。
- 矩阵与四元数之间的结果应通过随机往返测试，而不是只看一个示例。

## 最终产出物验收

GitHub 仓库需要至少包含：

- 五项练习源码及清晰的运行说明。
- 可重复执行的单元测试或 CTest。
- CIFAR-10 实测指标、训练日志和曲线。
- C++ 多文件 CMake 工程。
- Git 提交历史。
- 根目录 README，说明环境、命令、结果和已知限制。

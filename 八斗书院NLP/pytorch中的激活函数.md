# pytorch中的激活函数

# 📊 激活函数详解与适用场景

---

## ✅ 1. `ReLU` — `nn.ReLU()`

![2025-09-08_09-40.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/87feb02c-c2ce-4228-ad0f-b7a60b666ded.jpeg)

- **公式**：`f(x) = max(0, x)`
- **特点**：
  - 计算简单，收敛快。
  - 缓解梯度消失（正区间梯度=1）。
  - 存在“神经元死亡”问题（负值梯度=0）。
- **适用场景**：
  - ✅ 默认首选，适用于大多数 CNN、MLP、Transformer 等。
  - 尤其适合图像、语音、NLP 等深层网络。
- **缺点**：
  - Dead ReLU 问题（负值永久失活）。

> 🚀 **推荐指数：⭐⭐⭐⭐⭐（通用首选）**

---

## ✅ 2. `ReLU6` — `nn.ReLU6()`

![2025-09-08_09-53.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/d147eb7c-de6d-41db-924c-5b28610886ee.jpeg)

- **公式**：`f(x) = min(max(0, x), 6)`
- **特点**：
  - ReLU 的“截断版”，上限为6。
  - 在移动端/量化模型中表现稳定（如 MobileNet）。
- **适用场景**：
  - ✅ 移动端模型、量化感知训练（QAT）、边缘设备部署。
  - ✅ MobileNet、EfficientNet 等轻量模型常用。
- **优点**：
  - 数值范围受限 → 更适合低精度计算。

> 📱 **推荐指数：⭐⭐⭐⭐（移动端/量化首选）**

---

## ✅ 3. `LeakyReLU` — `nn.LeakyReLU(negative_slope=0.1)`

![2025-09-08_09-54.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/e8177bf5-41f0-419a-958b-a6503b63d773.jpeg)

- **公式**：
  `f(x) = x if x > 0 else 0.1 * x`
- **特点**：
  - 负值区域有小斜率 → 缓解神经元死亡。
- **适用场景**：
  - ✅ GANs（生成对抗网络）中常用，防止生成器/判别器“死掉”。
  - ✅ 当 ReLU 表现不佳时的替代方案。
- **缺点**：
  - 负斜率是超参数，需调优（常用 0.01~0.2）。

> 🎭 **推荐指数：⭐⭐⭐⭐（GANs/ReLU 替代）**

---

## ✅ 4. `PReLU` — `nn.PReLU()`

![2025-09-08_09-54_1.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/b90b9ea3-59cd-4d01-90d1-a8b6108549db.jpeg)

- **公式**：`f(x) = x if x > 0 else a * x`，其中 `a` 是可学习参数。
- **特点**：
  - LeakyReLU 的“可学习版”，每个通道/神经元有自己的负斜率。
- **适用场景**：
  - ✅ 高性能模型（如人脸识别、目标检测）中表现优异（如 PReLU 在 PReLUNet、InsightFace 中使用）。
  - ✅ 当你愿意增加少量参数换取性能提升时。
- **缺点**：
  - 增加参数量，训练稍慢。

> 🧬 **推荐指数：⭐⭐⭐（追求极致性能时使用）**

---

## ⚠️ 5. `RReLU` — `nn.RReLU()`

![2025-09-08_09-56.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/bd91f849-e691-4223-bfdb-61c50981d70d.jpeg)

- **公式**：负斜率在训练时随机采样（如 [0.1, 0.3]），测试时取均值。
- **特点**：
  - 类似 Dropout 的正则化效果。
- **适用场景**：
  - ❗️较少使用，主要用于研究或特定正则化需求。
  - 曾用于早期 Kaggle 比赛（如 NDSB）。
- **缺点**：
  - 不稳定，工业界很少用。

> 🧪 **推荐指数：⭐（研究/实验用）**

---

## ✅ 6. `ELU` — `nn.ELU()`

![2025-09-08_09-56_1.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/5392a72a-19f7-49eb-a009-9eb99cfeec6b.jpeg)

- **公式**：
  `f(x) = x if x > 0 else α*(exp(x)-1)`（默认 α=1）
- **特点**：
  - 负值区域平滑 → 均值更接近零 → 加速收敛。
  - 比 ReLU 更鲁棒，但计算稍慢。
- **适用场景**：
  - ✅ 自编码器、无监督学习、需要平滑梯度的场景。
  - ✅ 当 ReLU 收敛慢或不稳定时尝试。

> 🌀 **推荐指数：⭐⭐⭐⭐（平滑/无监督任务）**

---

## ✅ 7. `SELU` — `nn.SELU()`

![2025-09-08_09-57.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/c6da4225-682f-46a0-833c-283d6be02797.jpeg)

- **公式**：自归一化激活函数，有特定 α, λ 值（≈1.05, 1.67）。
- **特点**：
  - 如果网络满足特定条件（如 LeCun 正态初始化、每层均值方差稳定），可实现**自归一化**（无需 BatchNorm）。
- **适用场景**：
  - ✅ 全连接网络（MLP）中替代 BatchNorm。
  - ✅ 当你想简化网络结构（去掉 BN 层）时。
- **⚠️ 限制**：
  - 必须配合特定初始化和架构，否则可能爆炸或失效。

> 🧮 **推荐指数：⭐⭐⭐（特定 MLP 架构）**

---

## ⚠️ 8. `CELU` — `nn.CELU()`



- **公式**：`f(x) = max(0, x) + min(0, α*(exp(x/α)-1))`
- **特点**：
  - ELU 的连续可微版本（在0点更平滑）。
- **适用场景**：
  - ❗️研究用，工业界极少使用。
  - 在需要高阶导数（如物理模拟、微分方程网络）时可能有用。

> 🔬 **推荐指数：⭐（特殊需求）**

---

## ✅ 9. `GELU` — `nn.GELU()`

- **公式**：`f(x) = x * Φ(x)`，其中 Φ 是标准正态累积分布函数。
- **近似实现**：`0.5 * x * (1 + tanh[√(2/π)(x + 0.044715x³)])`
- **特点**：
  - 平滑、非单调，结合了 Dropout、Zoneout 的思想。
  - 在 Transformer 中表现极佳。
- **适用场景**：
  - ✅ **Transformer 架构首选**（BERT, GPT, ViT, LLaMA 等）。
  - ✅ NLP、大模型、注意力机制模型。
- **优点**：
  - 实验表现优于 ReLU 和 ELU。

> 🤖 **推荐指数：⭐⭐⭐⭐⭐（Transformer 必选）**

---

## ✅ 10. `Sigmoid` — `nn.Sigmoid()`

- **公式**：`f(x) = 1 / (1 + exp(-x))`
- **特点**：
  - 输出在 (0,1)，可解释为概率。
  - 梯度消失严重（两端梯度≈0）。
- **适用场景**：
  - ✅ 二分类任务的**最后一层输出**。
  - ✅ 注意力机制中的门控（如 LSTM 中的门）。
  - ❌ 不推荐用于隐藏层。

> 🎯 **推荐指数：⭐⭐⭐（仅用于输出层/门控）**

---

## ⚠️ 11. `LogSigmoid` — `nn.LogSigmoid()`

- **公式**：`f(x) = log(1 / (1 + exp(-x)))`
- **特点**：
  - 数值稳定性更好（避免 exp 溢出）。
- **适用场景**：
  - ✅ 作为损失函数的一部分（如 BCEWithLogitsLoss 内部使用）。
  - ❌ 一般不直接用于网络激活。

> 🧮 **推荐指数：⭐⭐（配合损失函数使用）**

---

## ✅ 12. `Tanh` — `nn.Tanh()`

- **公式**：`f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
- **特点**：
  - 输出在 (-1, 1)，均值为0 → 比 Sigmoid 更适合隐藏层。
  - 仍有梯度消失问题。
- **适用场景**：
  - ✅ RNN、LSTM、GRU 的隐藏状态激活。
  - ✅ 早期 CNN/MLP（现多被 ReLU 替代）。
  - ✅ 强化学习中策略网络输出（需负值）。

> 🔄 **推荐指数：⭐⭐⭐（RNN/历史模型）**

---

## ⚠️ 13. `Softplus` — `nn.Softplus()`

- **公式**：`f(x) = log(1 + exp(x))`
- **特点**：
  - ReLU 的平滑近似，处处可导。
- **适用场景**：
  - ✅ 需要 ReLU 但要求可导的场景（如某些概率模型、变分推断）。
  - ❌ 一般性能不如 ReLU。

> 📐 **推荐指数：⭐⭐（平滑 ReLU 替代）**

---

## ⚠️ 14. `Softsign` — `nn.Softsign()`

- **公式**：`f(x) = x / (1 + |x|)`
- **特点**：
  - 类似 Tanh，但衰减更慢。
- **适用场景**：
  - ❗️极少使用，可被 Tanh 或 ReLU 替代。

> 🚫 **推荐指数：⭐（不推荐）**

---

## ✅ 15. `SiLU (Swish)` — `nn.SiLU()`

- **公式**：`f(x) = x * sigmoid(x)`
- **特点**：
  - 平滑、非单调，实验表现优异。
  - Swish 是其别名（Google 提出）。
- **适用场景**：
  - ✅ 替代 ReLU，尤其在图像分类、EfficientNet 中表现好。
  - ✅ 当 GELU 不适用时（如非 Transformer 架构）的优秀选择。

> 🚀 **推荐指数：⭐⭐⭐⭐（ReLU 强力替代）**

---

## ✅ 16. `Mish` — `nn.Mish()`

- **公式**：`f(x) = x * tanh(softplus(x)) = x * tanh(ln(1+e^x))`
- **特点**：
  - 平滑、非单调、无上界、有下界。
  - 实验在 CV 任务中优于 Swish/GELU。
- **适用场景**：
  - ✅ 计算机视觉任务（YOLOv4, YOLOv7 使用）。
  - ✅ 当你想尝试超越 ReLU/Swish 的激活函数时。
- **缺点**：
  - 计算开销略大。

> 🖼️ **推荐指数：⭐⭐⭐⭐（CV 任务优选）**

---

## ⚠️ 17. `Hardtanh` — `nn.Hardtanh()`

- **公式**：`f(x) = -1 if x < -1, x if -1<=x<=1, 1 if x>1`
- **特点**：
  - Tanh 的分段线性近似。
- **适用场景**：
  - ✅ 量化/低精度训练中替代 Tanh。
  - ✅ RNN 隐藏层（节省计算）。

> 📉 **推荐指数：⭐⭐（量化/Tanh 替代）**

---

## ✅ 18. `Hardswish` — `nn.Hardswish()`

- **公式**：分段线性近似 Swish，计算更快。
- **特点**：
  - Swish 的高效版本，适合移动端。
- **适用场景**：
  - ✅ **MobileNetV3、EfficientNet-Lite 等移动端模型首选**。
  - ✅ 替代 ReLU6 或 Swish 以获得更好精度+速度平衡。

> 📱 **推荐指数：⭐⭐⭐⭐⭐（移动端模型首选）**

---

## ✅ 19. `Hardsigmoid` — `nn.Hardsigmoid()`

- **公式**：Sigmoid 的分段线性近似。
- **特点**：
  - 计算快，适合移动端。
- **适用场景**：
  - ✅ MobileNetV3 中的 SE 模块、注意力门控。
  - ✅ 替代 Sigmoid 用于门控机制（节省计算）。

> 📱 **推荐指数：⭐⭐⭐⭐（移动端门控首选）**

---

## ⚠️ 20~22. `Hardshrink`, `Softshrink`, `Tanhshrink`

- **共同点**：都是“收缩函数”，用于稀疏化或去噪。
- **适用场景**：
  - ❗️极少用于 DNN 激活层。
  - 可能用于：稀疏编码、自编码器去噪、信号处理。
- **工业界基本不用作隐藏层激活函数。**

> 🧪 **推荐指数：⭐（研究/特殊用途）**

---

## ⚠️ 23. `Threshold` — `nn.Threshold(threshold=0.5, value=0.1)`

- **公式**：`f(x) = x if x > threshold else value`
- **适用场景**：
  - ❗️自定义阈值激活，极少见于现代网络。
  - 可能用于：二值化网络、硬件模拟。

> 🚫 **推荐指数：⭐（不推荐常规使用）**

---

## ⚠️ 24. `GLU` — `nn.GLU()`

- **公式**：Gated Linear Unit，`GLU(x) = a ⊗ σ(b)`，其中 x 被分成 a, b 两部分。
- **特点**：
  - 引入门控机制，类似 LSTM。
- **适用场景**：
  - ✅ **Transformer 的变体（如 GLU Variants in LLaMA, PaLM）**。
  - ✅ 替代 FFN 中的 ReLU，提升性能。
- **注意**：
  - 输入维度必须是偶数（自动 split）。

> 🤖 **推荐指数：⭐⭐⭐⭐（Transformer 进阶替换）**

---

# 🎯 总结推荐表（按任务类型）


| 任务类型              | 推荐激活函数                      |
| --------------------- | --------------------------------- |
| **通用 MLP/CNN**      | ReLU → Swish → Mish             |
| **Transformer (NLP)** | GELU                              |
| **移动端/轻量模型**   | Hardswish, ReLU6, Hardsigmoid     |
| **GANs**              | LeakyReLU                         |
| **RNN/LSTM**          | Tanh, Sigmoid (门控)              |
| **输出层（二分类）**  | Sigmoid                           |
| **输出层（多分类）**  | Softmax（不是激活函数，是输出层） |
| **自编码器/无监督**   | ELU, SELU                         |
| **Transformer 进阶**  | GLU, SwiGLU, GeGLU                |

---

# ✅ 最佳实践建议

1. **默认用 ReLU** — 简单、快、有效。
2. **做 NLP / Transformer → 用 GELU**。
3. **做 CV 且追求 SOTA → 试 Mish**。
4. **部署移动端 → 用 Hardswish + Hardsigmoid**。
5. **GANs → LeakyReLU**。
6. **不想调参 → 避免 PReLU/RReLU/Threshold**。
7. **需要平滑梯度 → ELU / Softplus**。

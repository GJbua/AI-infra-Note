# 大模型量化技术学习笔记：原理、算法与工程落地

> 学习目标：掌握大模型 W8A8、W4A16 量化原理，理解 AWQ、GPTQ、SmoothQuant 等量化算法，能够独立设计 Qwen-Image 的量化方案。
>
> 工程背景：vLLM-Omni v0.20.0、Qwen-Image-2512、Ascend NPU。

---

## 1. 量化基础

### 1.1 什么是量化？

量化（Quantization）是将高精度浮点数转换为低比特整数表示，以减少模型权重存储、显存占用和访存开销，并利用硬件低精度计算能力提高推理效率。

常见数据类型：

| 数据类型 | 位宽 | 说明 |
|---|---:|---|
| FP32 | 32 bit | 单精度浮点 |
| FP16 | 16 bit | 半精度浮点 |
| BF16 | 16 bit | Brain Floating Point |
| INT8 | 8 bit | 8 位整数 |
| INT4 | 4 bit | 4 位整数 |

以仿射量化（Affine Quantization）为例：

```math
q=\mathrm{clip}\left(
\mathrm{round}\left(\frac{x}{s}\right)+z,
q_{\min},q_{\max}
\right)
```

反量化：

```math
\hat{x}=s(q-z)
```

其中：

- $x$：原始浮点数。
- $q$：量化后的整数。
- $s$：量化 Scale。
- $z$：Zero Point。
- $q_{\min},q_{\max}$：整数表示范围。
- $\hat{x}$：反量化后的近似浮点数。

对于对称量化，通常令 $z=0$：

```math
s=\frac{\max(|x|)}{q_{\max}}
```

```math
q=\mathrm{clip}\left(
\mathrm{round}\left(\frac{x}{s}\right),
-q_{\max},q_{\max}
\right)
```

注意：具体整数范围取决于量化格式。例如对称 INT8 常采用 $[-127,127]$，对称 INT4 常采用 $[-7,7]$，也存在其他实现方式。

量化误差：

```math
E=x-\hat{x}
```

量化的核心目标是在降低数据表示精度的同时，尽可能减小量化误差对模型计算结果的影响。

### 1.2 GEMM 是什么？

GEMM（General Matrix Multiplication）是通用矩阵乘法，不等于 INT8 矩阵乘法。

以 Linear 为例：

```math
Y=XW
```

假设：

```math
X\in\mathbb{R}^{M\times K}
```

```math
W\in\mathbb{R}^{K\times N}
```

```math
Y\in\mathbb{R}^{M\times N}
```

其中：

- $M$：输入 Token 数量或 Batch 维度。
- $K$：输入特征维度。
- $N$：输出特征维度。

GEMM 可以使用 FP16、BF16、INT8 等不同数据类型。

对于 INT8 计算，假设采用对称量化：

```math
X\approx s_X X_q
```

```math
W\approx s_W W_q
```

因此：

```math
Y\approx s_Xs_W(X_qW_q)
```

其中 $X_q$ 和 $W_q$ 是 INT8 矩阵。

矩阵乘法通常先使用 INT32 累加，再根据 Scale 恢复到目标输出精度。

例如，假设激活使用 Per-Token Scale，权重使用 Per-Output-Channel Scale：

```math
Y_{m,n}\approx
s_{X,m}s_{W,n}
\sum_{k=1}^{K}X_{q,m,k}W_{q,k,n}
```

如果权重采用 Per-Group 量化，则每组具有独立 Scale，不能直接将一个统一的 $s_W$ 提取到整个求和式外面。

此时：

```math
Y_{m,n}\approx
s_{X,m}
\sum_g s_{W,g,n}
\sum_{k\in g}X_{q,m,k}W_{q,k,n}
```

**为什么 INT8 矩阵乘法通常使用 INT32 累加？**

INT8 的单个数值范围有限，但矩阵乘法需要对 K 个乘积进行累加，累加结果可能远超 INT8 的表示范围。

因此通常使用 INT32 保存中间累加结果，避免溢出。

### 1.3 量化为什么能够加速？

主要有三个方面：

1. **降低权重访存量**：INT8 权重理论上约为 FP16/BF16 的一半，INT4 约为四分之一。
2. **提高计算吞吐**：硬件支持专用低精度矩阵计算单元时，可以获得更高的计算吞吐。
3. **降低显存占用**：为更大的模型或 Batch Size 留出更多显存空间。

但是，低比特不一定意味着更快。

如果硬件没有对应的低精度 GEMM Kernel，或者动态量化、反量化、数据布局转换带来的额外开销较大，实际推理速度可能没有提升。

因此需要使用 Profiler 检查实际执行的算子。

---

## 2. 常见量化格式：W8A8 与 W4A16

### 2.1 W8A8

W8A8 表示：

- W（Weight）：权重使用 INT8。
- A（Activation）：激活使用 INT8。

典型计算流程：

1. 模型加载阶段或离线准备阶段，将权重转换为 INT8。
2. Linear 前向时，将浮点激活量化为 INT8。
3. 执行 INT8 GEMM。
4. 使用 INT32 累加。
5. 结合 Scale 恢复到目标输出精度。

```math
W_{\mathrm{BF16}}\rightarrow W_{\mathrm{INT8}}
```

```math
X_{\mathrm{BF16}}\rightarrow X_{\mathrm{INT8}}
```

```math
Y_{\mathrm{INT32}}=X_{\mathrm{INT8}}W_{\mathrm{INT8}}
```

优点：

- 权重和激活都可以降低位宽。
- 能够利用硬件 INT8 GEMM 获得计算加速。
- 权重和激活的访存量均可能降低。

缺点：

- 权重和激活都会引入量化误差。
- 激活离群值可能影响量化精度。
- 需要硬件提供高效的 INT8 GEMM Kernel。

适用场景：

硬件具备高效 INT8 计算能力，希望同时降低权重访存和计算开销。

### 2.2 W4A16

W4A16 表示：

- W：权重使用 INT4。
- A：激活使用 FP16/BF16。

典型计算流程：

1. 离线将权重转换为 INT4。
2. 保存 INT4 权重及对应的 Quantization Scale。
3. 推理时读取 INT4 权重。
4. Kernel 对 INT4 权重进行解包、反量化，或者使用融合反量化的矩阵乘法 Kernel。
5. 与 FP16/BF16 激活执行计算。

```math
W_{\mathrm{BF16}}\rightarrow W_{\mathrm{INT4}}
```

```math
Y=X_{\mathrm{BF16}}\cdot\hat{W}_{\mathrm{INT4}}
```

其中 $\hat{W}_{\mathrm{INT4}}$ 表示反量化后的近似权重。

注意：反量化不一定需要单独生成完整的 BF16 权重矩阵。高效的 Weight-only Kernel 可以在计算过程中融合解包、反量化和矩阵乘法。

优点：

- 权重压缩程度更高。
- 显著降低权重存储和访存开销。
- 不需要将激活量化成 INT8。

缺点：

- INT4 权重量化误差通常更大。
- 需要专用 INT4 权重布局和计算 Kernel。
- 反量化开销可能抵消部分加速收益。

适用场景：

模型权重显存或权重访存成为瓶颈，希望进一步压缩模型。

### 2.3 W8A8 与 W4A16 对比

| 维度 | W8A8 | W4A16 |
|---|---|---|
| 权重 | INT8 | INT4 |
| 激活 | INT8 | FP16/BF16 |
| 权重存储量 | FP16/BF16 的约 1/2 | FP16/BF16 的约 1/4 |
| 激活压缩 | 有 | 无 |
| 主要优化 | 计算与访存 | 权重存储与访存 |
| 主要误差来源 | 权重和激活量化 | 权重量化 |
| 关键硬件能力 | INT8 GEMM | INT4 Weight-only Kernel |

注意：上述压缩比例不包含 Scale、Zero Point、Padding 和其他元数据开销。

---

## 3. Online Quantization 与 Offline Quantization

### 3.1 在线权重量化

在当前讨论的 vLLM-Omni 量化路径中，Online Weight Quantization 可以理解为：

模型加载时读取原始浮点权重，计算量化参数并生成低比特权重。

```math
W_{\mathrm{BF16}}
\xrightarrow{\mathrm{Model\ Loading}}
(W_q,s_W)
```

之后将量化权重保存在设备内存中，推理时直接使用。

**在线权重量化不意味着每次请求都重新量化权重。**

优点：

- 可以直接加载原始 BF16/FP16 Checkpoint。
- 不需要提前制作专用量化 Checkpoint。
- 部署流程相对简单。

缺点：

- 每次模型启动可能都需要执行权重量化。
- 基础在线量化通常不会自动完成复杂的校准搜索和模型级误差补偿。

### 3.2 离线量化

离线量化在模型部署之前完成权重量化：

```math
W_{\mathrm{BF16}}
\xrightarrow{\mathrm{Offline\ Quantization}}
(W_q,s_W,\mathrm{Metadata})
```

保存：

- 量化权重。
- Weight Scale。
- Zero Point（如果使用非对称量化）。
- Group Size 等量化配置。
- 其他必要的量化元数据。

部署时直接加载量化 Checkpoint。

优点：

- 减少模型启动时的权重量化开销。
- 可以使用校准数据。
- 可以结合 AWQ、GPTQ 等算法进行误差优化。

### 3.3 在线量化与动态激活量化的区别

需要特别区分：

**权重量化：**

- 权重在模型加载阶段或离线阶段量化。
- 权重在推理期间通常保持不变。
- 不需要每个请求重复量化。

**激活动态量化：**

- 激活在模型运行时量化。
- 不同输入 Token 的激活分布可能不同。
- 可以根据当前输入动态计算 Scale。

例如，假设激活：

```math
X\in\mathbb{R}^{M\times K}
```

采用 Per-Token 动态量化：

```math
S_X\in\mathbb{R}^{M\times 1}
```

每一行对应一个 Token，每个 Token 拥有自己的 Scale。

因此：

```math
X_{q,m,k}=
\mathrm{clip}\left(
\mathrm{round}\left(\frac{X_{m,k}}{S_{X,m}}\right),
q_{\min},q_{\max}
\right)
```

**Online / Offline 描述的是量化发生的时机，不直接代表量化精度高低。**

---

## 4. Quantization Scale 的粒度

假设权重矩阵：

```math
W\in\mathbb{R}^{K\times N}
```

其中：

- K：输入通道数。
- N：输出通道数。

本文统一使用数学布局 $[K,N]$。实际框架可能使用 $[N,K]$ 存储权重，因此判断量化轴时必须结合实际权重布局。

### 4.1 Per-Tensor

整个权重矩阵共享一个 Scale：

```math
S_W\in\mathbb{R}
```

优点：

- 实现简单。
- Scale 元数据开销小。
- Kernel 实现相对容易。

缺点：

- 不同通道之间的数值范围可能差异很大。
- 一个极大值可能扩大整个矩阵的量化范围。
- 其他通道的有效表示精度可能下降。

### 4.2 Per-Channel

通常采用 Per-Output-Channel Weight Quantization，即每个输出通道单独计算 Scale。

对于 $W\in\mathbb{R}^{K\times N}$：

```math
S_W\in\mathbb{R}^{N}
```

每个输出通道对应一列权重，独立计算量化 Scale。

优点：

- 减少不同输出通道之间的数值范围干扰。
- 通常比 Per-Tensor 具有更好的量化精度。
- Scale 元数据开销相对可控。

注意：Per-Channel 必须明确具体量化轴。

### 4.3 Per-Input-Channel

也可以按照输入通道独立计算 Scale：

```math
S_W\in\mathbb{R}^{K}
```

每个输入通道对应 W 的一行。

这种方式与 AWQ 的输入通道缩放方向一致，因此需要特别注意两者之间的关系，后面会详细解释。

### 4.4 Per-Group

将每个输出通道沿输入维度 K 划分为多个 Group，每组独立计算 Scale。

假设 Group Size 为 G：

```math
S_W\in\mathbb{R}^{\lceil K/G\rceil\times N}
```

每个输出通道内部有 $\lceil K/G\rceil$ 个量化组。

优点：

- 能够适应更局部的权重分布。
- 减少不同数值区域之间的相互影响。
- 常用于 W4A16 权重量化。

缺点：

- Scale 元数据开销增加。
- Kernel 实现更加复杂。
- Group Size 会影响精度和计算效率。

通常 Group Size 越小，越容易适应局部权重分布，但元数据开销和 Kernel 处理成本也会增加。

### 4.5 三种主要粒度对比

| 粒度 | Scale 数量 | 特点 |
|---|---|---|
| Per-Tensor | 1 | 最简单，不同通道共享 Scale |
| Per-Output-Channel | 每个输出通道一个 | 减少输出通道之间的范围干扰 |
| Per-Group | 每个权重组一个 | 更细粒度，常用于 INT4 |

一般情况下，较细粒度的量化有利于降低量化误差，但并不保证所有实现都严格满足：

```math
E_{\mathrm{PerGroup}}
\leq E_{\mathrm{PerChannel}}
\leq E_{\mathrm{PerTensor}}
```

实际误差还取决于分组方向、权重分布和量化算法。

---

## 5. AWQ：Activation-aware Weight Quantization

### 5.1 AWQ 解决什么问题？

AWQ 是一种激活感知的权重量化算法，经典应用是 W4A16 的 Group-wise INT4 Quantization。

核心观察：

不同输入通道对模型输出的影响并不相同。一些激活幅度较大的输入通道通常更加重要，需要在量化过程中得到保护。

AWQ 利用校准数据统计激活分布，识别重要输入通道，并通过等价通道缩放降低重要通道的量化误差。

注意：

- AWQ 保护的是 Linear 内部的重要输入通道。
- 它不是判断整个 Transformer Block 是否重要。
- 它不会增加 INT4 的表示位数。
- 它也不是把重要通道直接保存为 FP16。

### 5.2 AWQ 的等价变换

假设 Linear：

```math
Y=XW
```

其中：

```math
X\in\mathbb{R}^{B\times K}
```

```math
W\in\mathbb{R}^{K\times N}
```

定义逐输入通道缩放向量：

```math
d=(d_1,d_2,\ldots,d_K)
```

构造对角矩阵：

```math
D=\mathrm{diag}(d_1,d_2,\ldots,d_K)
```

进行等价变换：

```math
\boxed{Y=(XD^{-1})(DW)}
```

也就是：

```math
X'=XD^{-1}
```

```math
W'=DW
```

具体含义：

- X 的每个输入通道除以对应的 $d_i$。
- W 的对应输入通道乘以 $d_i$。
- 原始矩阵乘法结果完全不变。

这里没有增加矩阵维度，也没有增加参数量，只是改变了输入和权重的数值分布。

### 5.3 AWQ 为什么可以降低量化误差？

定义 $Q(W')$ 表示经过量化再反量化后的权重近似值。

量化输出：

```math
Y_q=XD^{-1}Q(DW)
```

原始输出：

```math
Y=XW
```

两者的误差：

```math
\boxed{
Y_q-Y=XD^{-1}\left[Q(DW)-DW\right]
}
```

这个公式说明：

输出误差同时受到两个因素影响：

1. 缩放后的激活 $XD^{-1}$。
2. 缩放后的权重 $DW$ 所产生的量化误差。

假设某个重要输入通道的缩放因子 $d_i$ 增大：

- 对应权重被放大。
- 对应激活被缩小。
- 重要通道的权重在共享量化范围内可能获得更好的有效表示精度。
- 该通道的量化误差对最终输出的贡献也会受到激活缩放的影响。

但不能简单认为最终误差必然缩小 $1/d_i$。

因为权重放大后，量化 Scale 和 Group 内的数值范围也可能发生变化，进而影响其他权重的量化误差。

因此 AWQ 需要搜索合适的通道缩放因子。

### 5.4 AWQ 如何确定缩放因子？

首先使用少量校准数据运行模型，收集 Linear 输入激活。

对于第 i 个输入通道，统计平均绝对激活值：

```math
a_i=\frac{1}{T}\sum_{t=1}^{T}|X_{t,i}|
```

其中 T 表示校准样本和 Token 维度上的统计数量。

同时统计对应输入通道的权重幅度：

```math
w_i=\max_j|W_{i,j}|
```

AWQ 的典型缩放形式为：

```math
\boxed{
d_i=\frac{a_i^\alpha}{w_i^{1-\alpha}}
}
```

其中：

```math
\alpha\in[0,1]
```

$\alpha$ 用于平衡激活重要性与权重数值范围。

具体实现中可能还会进行数值稳定性处理，例如加入很小的正数，避免除零。

随后进行网格搜索：

1. 选择一个候选 $\alpha$。
2. 根据激活统计和权重统计计算缩放向量 d。
3. 执行等价变换 $X'=XD^{-1}$、$W'=DW$。
4. 对变换后的权重执行 Group-wise INT4 量化。
5. 使用校准激活计算量化输出。
6. 与原始浮点输出比较。
7. 选择输出误差较小的缩放参数。

典型重构目标：

```math
\boxed{
\min_D
\left\|
XW-XD^{-1}Q(DW)
\right\|_F^2
}
```

AWQ 不需要重新训练模型，也不需要反向传播。

### 5.5 为什么 AWQ 常与 Group-wise INT4 结合？

Group-wise Quantization 中，同一个 Group 内的权重共享一个量化 Scale。

假设某个 Group 内包含多个输入通道。

如果将重要通道对应的权重放大：

- 重要通道的权重可能获得更多有效量化表示空间。
- 但 Group 的最大值也可能增大。
- 其他通道的量化步长可能随之增大。
- 其他权重的量化误差可能上升。

因此 AWQ 需要在保护重要通道与避免恶化其他通道之间取得平衡。

### 5.6 AWQ 与 Per-Channel / Per-Group 的关系

AWQ 的通道缩放与量化 Scale 是两个不同对象。

| 参数 | 维度 | 作用 |
|---|---|---|
| AWQ 通道缩放 d | 输入通道 K | 调整输入通道和对应权重行的数值分布 |
| Per-Output-Channel Scale | 输出通道 N | 每个输出通道独立量化 |
| Per-Group Scale | Group 数量 × N | 每个权重组独立量化 |
| Per-Input-Channel Scale | 输入通道 K | 每个输入通道独立量化 |

**Per-Output-Channel 量化仍然可以使用 AWQ。**

因为每个输出通道虽然拥有独立 Scale，但该输出通道内部的不同输入通道仍然共享量化范围。

AWQ 对输入通道进行缩放，仍可能改变这些权重之间的相对分布。

但是，如果使用 Per-Input-Channel 量化，每个输入通道独立计算 Scale：

```math
\Delta_i=\frac{\max_j|W_{i,j}|}{q_{\max}}
```

AWQ 对该输入通道的权重乘以 $d_i$ 后，Scale 也会相应变化：

```math
\Delta_i'=d_i\Delta_i
```

于是：

```math
\frac{d_iW_{i,j}}{d_i\Delta_i}
=
\frac{W_{i,j}}{\Delta_i}
```

在这种理想的对称、基于最大绝对值计算 Scale 的条件下，量化整数值可能完全不变。

因此，如果每个输入通道独立量化，AWQ 的通道缩放可能变得冗余。

**核心结论：AWQ 是否有效，取决于通道缩放方向与量化 Scale 的分组方向，而不是简单取决于是否使用 Per-Channel。**

---

## 6. GPTQ 与 SmoothQuant

### 6.1 GPTQ

GPTQ 是一种基于近似二阶信息的权重量化算法。

基本思路：

1. 使用校准数据收集 Linear 输入。
2. 估计输入的二阶信息。
3. 逐步量化权重。
4. 利用近似二阶信息补偿权重量化误差。
5. 保存量化权重和对应元数据。

与 AWQ 的区别：

- AWQ：利用激活重要性进行等价缩放，保护重要输入通道。
- GPTQ：利用近似二阶信息，在逐步量化权重时补偿误差。

两者都属于 PTQ（Post-Training Quantization），不需要完整重新训练模型。

### 6.2 SmoothQuant

SmoothQuant 主要解决 W8A8 中激活离群值带来的量化困难。

通过等价变换：

```math
XW=(XD^{-1})(DW)
```

将部分激活侧的量化难度转移到权重侧，使激活和权重更加适合 INT8 量化。

典型缩放因子可以根据激活和权重的通道统计量构造：

```math
d_i=
\frac{
\left(\max_t|X_{t,i}|\right)^\alpha
}{
\left(\max_j|W_{i,j}|\right)^{1-\alpha}
}
```

与 AWQ 的区别：

| 算法 | 主要目标 | 典型应用 |
|---|---|---|
| AWQ | 保护重要输入通道，降低低比特权重量化误差 | W4A16 |
| GPTQ | 利用近似二阶信息补偿权重量化误差 | W4A16 等 |
| SmoothQuant | 缓解激活离群值，平滑权重和激活分布 | W8A8 |

虽然 AWQ 和 SmoothQuant 都涉及等价缩放，但它们的优化目标和缩放策略不同。

---

## 7. 层敏感性分析与混合精度量化

### 7.1 层敏感性分析解决什么问题？

AWQ、GPTQ、SmoothQuant 解决的是如何让一个模块量化得更好。

层敏感性分析解决的是：

**模型中的哪些模块对量化误差更加敏感，哪些模块应该保留 BF16？**

不同 Linear 和 Transformer Block 对量化误差的敏感程度可能不同。

因此可以让大部分模块使用 INT8，而少数敏感模块保留 BF16。

### 7.2 敏感性分析的统计粒度

建议分为三个层次。

**第一层：Linear 级**

统计：

- QKV Projection。
- Attention Output Projection。
- MLP Up Projection。
- MLP Down Projection。

目的：定位具体哪个矩阵量化误差较大。

**第二层：子模块级**

统计：

- Attention。
- MLP。

目的：判断哪个计算子模块更加敏感。

**第三层：Transformer Block 级**

统计完整 Block 的输入和输出。

目的：观察整个 Block 的量化误差传播。

注意：部分文献和代码中的 Transformer Layer 可能指完整 Transformer Block，使用时要明确术语。

### 7.3 量化误差指标

假设：

```math
Y_{\mathrm{ref}}=Y_{\mathrm{BF16}}
```

```math
Y_q=Y_{\mathrm{quant}}
```

#### MSE

```math
MSE=
\frac{1}{N}
\sum_{i=1}^{N}
(Y_{\mathrm{ref},i}-Y_{q,i})^2
```

关注绝对数值误差。

缺点是不同层输出数值范围可能不同，直接比较 MSE 不一定公平。

#### NMSE

```math
NMSE=
\frac{
\|Y_{\mathrm{ref}}-Y_q\|_F^2
}{
\|Y_{\mathrm{ref}}\|_F^2+\epsilon
}
```

NMSE 对数值误差进行归一化，更适合跨层比较。

#### Cosine Similarity

```math
CosSim=
\frac{
\langle Y_{\mathrm{ref}},Y_q\rangle
}{
\|Y_{\mathrm{ref}}\|_2\|Y_q\|_2+\epsilon
}
```

关注两个输出向量的方向是否一致。

例如：

```math
Y_{\mathrm{ref}}=[1,2]
```

```math
Y_q=[2,4]
```

此时：

```math
CosSim=1
```

但两个向量的数值幅度并不相同。

因此 Cosine Similarity 可能忽略整体幅度变化。

#### 指标选择

| 指标 | 关注内容 | 建议用途 |
|---|---|---|
| MSE | 绝对数值误差 | 基础误差评估 |
| NMSE | 相对数值误差 | 主要指标 |
| Cosine Similarity | 向量方向 | 辅助指标 |
| Relative L2 | 相对整体误差 | 辅助指标 |

建议使用 NMSE 或 Relative L2 作为主要指标，Cosine Similarity 作为辅助指标。

### 7.4 局部误差与端到端误差

需要区分两种实验。

**局部 Linear / Block 误差：**

使用 BF16 参考前向中缓存的相同输入，分别计算 BF16 和 INT8 输出。

这样能够测量模块本身的量化误差，避免上游量化误差改变输入分布。

**完整模型误差：**

直接运行 BF16 和量化模型，比较：

- DiT 噪声或速度预测误差。
- 不同去噪 Step 的 Latents 误差。
- 最终生成图像质量。

对于扩散模型，需要覆盖不同去噪阶段，因为激活分布和量化误差可能随时间步变化。

### 7.5 为什么不能只根据 Block 输出误差决定是否保留 BF16？

局部误差大不代表最终生成质量一定下降。

因此需要进一步执行消融实验：

1. 全部受支持 Linear 使用 INT8。
2. 选择一个 Linear 或 Block 恢复 BF16。
3. 使用相同 Prompt、Seed、分辨率、Scheduler 和去噪步数重新生成。
4. 比较恢复前后的生成质量。
5. 判断是否值得恢复该模块。

### 7.6 混合精度优化问题

可以抽象为：

```math
\min_{\boldsymbol{x}}T(\boldsymbol{x})
```

约束：

```math
L_{\mathrm{quality}}(\boldsymbol{x})\leq\varepsilon
```

其中：

- $x_i=1$：第 i 个模块使用 BF16。
- $x_i=0$：第 i 个模块使用 INT8。
- $T$：推理时间。
- $L_{\mathrm{quality}}$：最终生成质量损失。

可以将其近似抽象为背包或组合优化问题。

但真实模型中不同层的量化误差存在耦合，不能简单地把各层独立测量的误差直接相加作为整体误差。

最终需要通过联合消融实验验证。

---

## 8. Qwen-Image-2512 的量化工程落地

### 8.1 当前工程背景

- 框架：vLLM-Omni v0.20.0。
- 模型：Qwen-Image-2512。
- 硬件：Ascend 910B。
- 并行配置：TP=2。
- 推理方式：Diffusion Step Execution。

首先优先量化 DiT Transformer 中受支持的 Linear 层。

Text Encoder、VAE、Scheduler 暂时保持 BF16，减少实验变量。

具体量化支持、算子能力和 Scale 粒度，需要以本地 v0.20.0 代码及 Ascend 算子文档为准。

### 8.2 第一阶段：跑通 W8A8 INT8

先建立 BF16 Baseline，再运行现有 INT8 路径。

典型执行逻辑：

**权重侧：**

```math
W_{\mathrm{BF16}}
\xrightarrow{\mathrm{Load\ Time\ Quantization}}
(W_{\mathrm{INT8}},S_W)
```

**激活侧：**

```math
X_{\mathrm{BF16}}
\xrightarrow{\mathrm{Dynamic\ Quantization}}
(X_{\mathrm{INT8}},S_X)
```

**计算侧：**

```math
Y_{\mathrm{INT32}}=X_{\mathrm{INT8}}W_{\mathrm{INT8}}
```

**输出侧：**

```math
Y_{\mathrm{BF16}}\approx S_XS_WY_{\mathrm{INT32}}
```

最后一步是简化示意，实际 Scale 可能按照 Token、输出通道或 Group 进行广播，不能一概使用两个标量相乘。

需要确认：

- 权重是否真正转换成 INT8。
- 激活是否执行动态量化。
- 是否调用 Ascend INT8 GEMM。
- Weight Scale 的 Shape 和量化轴。
- Activation Scale 的 Shape 和广播方式。
- TP=2 下两个 Rank 是否正确处理本地权重分片。
- 输出是否恢复到预期精度。

不能仅凭程序启动成功就认为 INT8 加速已经生效，需要使用 Profiler 检查实际算子。

### 8.3 Online INT8 的局限

基础 Online INT8 路径主要负责：

- 加载阶段权重量化。
- 运行时激活动态量化。
- Scale 计算和传递。
- INT8 GEMM。
- 输出精度转换。

它不代表框架自动执行了 AWQ、GPTQ 或模型级敏感性分析。

基础在线量化不会自动：

- 使用校准数据搜索最优量化参数。
- 执行 AWQ 等价缩放。
- 使用 GPTQ 进行误差补偿。
- 自动识别敏感层。
- 自动将敏感层恢复为 BF16。

需要自行通过校准、消融实验和混合精度配置完成精度优化。

### 8.4 TP=2 下的权重量化

假设数学权重按照 TP=2 切分：

```math
W=[W_0,W_1]
```

两个 Rank 分别持有本地权重分片，可以对各自分片执行量化：

```math
W_{q,0}=Q(W_0)
```

```math
W_{q,1}=Q(W_1)
```

需要注意：

- Column Parallel 与 Row Parallel 的切分方向不同。
- Scale 必须与本地权重布局对应。
- 量化轴不能在 TP 切分后被错误改变。
- 量化前后必须保持原有 TP 通信语义。

对于 AWQ，缩放向量沿输入通道维度，因此还需要保证通道缩放与 TP 权重切分方式一致。

---

## 9. Qwen-Image 量化实验设计

### 9.1 第一阶段：建立 BF16 / INT8 Baseline

固定：

- Prompt。
- Random Seed。
- 分辨率。
- 去噪步数。
- Scheduler。
- Guidance Scale。
- Batch Size。
- TP 配置。

分别测试 BF16 和 INT8。

| 指标 | BF16 | INT8 |
|---|---|---|
| DiT 单步耗时 | 实测 | 实测 |
| 完整去噪耗时 | 实测 | 实测 |
| VAE Decode | 实测 | 实测 |
| 端到端延迟 | 实测 | 实测 |
| Throughput | 实测 | 实测 |
| 峰值显存 | 实测 | 实测 |
| 生成质量 | 评估 | 评估 |

### 9.2 第二阶段：Calibration & Profiling

准备具有代表性的 Prompt，覆盖不同去噪阶段：

- 前期。
- 中期。
- 后期。

收集：

- Linear 输入激活。
- Linear 输出。
- Block 输入。
- Block 输出。
- 不同 Step 的激活统计。

需要按模块和 Step 保存数据，不能简单地只保留最后一次 Hook 结果。

### 9.3 第三阶段：层敏感性分析

分别统计：

- Linear NMSE。
- Linear Cosine Similarity。
- Block NMSE。
- DiT Prediction Error。
- Latents NMSE。

先对误差较大的模块排序，再进行 BF16 恢复消融实验。

### 9.4 第四阶段：混合精度实验

| 实验 | 配置 |
|---|---|
| Baseline | 全 BF16 |
| INT8-1 | 全部受支持 Linear 使用 INT8 |
| INT8-2 | 敏感 Linear 恢复 BF16 |
| INT8-3 | 敏感 Block 恢复 BF16 |
| INT8-4 | 不同混合精度组合 |

最终比较：

- 图像生成质量。
- DiT 单步耗时。
- 端到端延迟。
- Throughput。
- 显存占用。

### 9.5 第五阶段：进一步优化

**W8A8 方向：**

研究 SmoothQuant 思路，降低激活离群值造成的量化误差。

**W4A16 方向：**

研究 AWQ、GPTQ、Group-wise INT4 和 Ascend INT4 Kernel 适配。

注意：算法原理不代表当前 Qwen-Image + Ascend 环境已经具备对应的完整工程支持。需要分别验证 Checkpoint 格式、量化算子和硬件 Kernel。

---

## 10. 面试总结：如何落地 Qwen-Image 量化？

如果让我负责将 Qwen-Image 部署到 Ascend NPU，我会按照以下步骤推进：

1. **分析模型与硬件**：Qwen-Image 的主要计算集中在 DiT Transformer，因此优先考虑量化 DiT 中受支持的 Linear 层。
2. **选择量化格式**：根据硬件 Kernel 能力选择 W8A8 或 W4A16。当前环境优先验证已有支持的 W8A8 INT8 路径。
3. **实现量化计算**：加载阶段量化权重，运行时动态量化激活，执行 INT8 GEMM，使用 INT32 累加并恢复输出精度。
4. **评估量化误差**：使用校准数据统计不同 Linear 和 Block 的误差，覆盖不同扩散时间步。
5. **实施混合精度**：通过消融实验识别敏感模块，对质量敏感的层保留 BF16，其余层使用 INT8。
6. **端到端验证**：固定 Prompt、Seed 和推理配置，对比生成质量、单步耗时、端到端延迟、吞吐和显存。
7. **进一步优化**：如果 W8A8 的激活离群值影响精度，研究 SmoothQuant；如果希望进一步压缩权重，研究 AWQ/GPTQ 与 INT4 Kernel 适配。

---

## 11. 核心知识框架总结

| 概念 | 解决的问题 |
|---|---|
| W8A8 / W4A16 | 决定权重和激活使用多少比特 |
| Per-Tensor / Per-Channel / Per-Group | 决定量化 Scale 如何分配 |
| AWQ / GPTQ / SmoothQuant | 决定如何降低量化误差 |
| GEMM Kernel | 决定如何执行量化后的矩阵计算 |
| Online / Offline | 决定量化发生的时机 |
| 层敏感性分析 | 决定哪些模块需要保留高精度 |
| 混合精度量化 | 在模型精度和推理性能之间进行权衡 |

**核心原则：量化格式决定使用多少比特，量化粒度决定 Scale 如何分配，量化算法决定如何降低误差，层敏感性分析决定哪些模块需要保留高精度，底层 Kernel 决定最终能否获得实际加速。**

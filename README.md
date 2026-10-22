# AI Infra 学习笔记

> 当前主题：LLM 推理基础 —— Prefill / Decode
>
> 复习方式：先独立回答问题，再对照答案查漏补缺，最后进行闭卷复述。
>
> 星级说明：
> - ★★★★★：核心必会
> - ★★★★☆：重要进阶
> - ★★★☆☆：拓展理解

---

## 一、问题索引

### 1. Prefill / Decode

| 编号 | 星级 | 难度 | 问题 |
|---|---|---|---|
| Q1 | ★★★★★ | 基础 | [什么是 Prefill 和 Decode 阶段？为什么 LLM 推理要分成两阶段？](#q1-prefill-and-decode) |
| Q2 | ★★★★★ | 基础 | [Prefill 和 Decode 各自的计算瓶颈是什么？为什么同一个模型在不同阶段的瓶颈不同？](#q2-performance-bottlenecks) |
| Q3 | ★★★★☆ | 进阶 | [Prompt 长度为 4096、生成 256 个 Token，Prefill 和 Decode 的计算量占比是多少？](#q3-computation-estimation) |
| Q4 | ★★★★★ | 基础 | [为什么 Decode 每次通常只处理一个 Token？能不能一次 Decode 多个 Token？](#q4-autoregressive-decode) |
| Q5 | ★★★★☆ | 进阶 | [Chunked Prefill 的工作原理是什么？它解决了什么问题，又带来了什么新问题？](#q5-chunked-prefill) |

---

## 二、问题与答案

### Q1. Prefill and Decode

**问题：什么是 Prefill 和 Decode 阶段？为什么 LLM 推理要分成两阶段？**

#### 核心概念

- **Prefill（提示词处理阶段）**：处理输入 Prompt 的所有 Token，计算各层中间表示，并建立对应的 KV Cache。
- **Decode（自回归生成阶段）**：基于已有上下文和 KV Cache，逐步生成后续 Token。标准自回归 Decode 中，每条请求通常每轮生成一个新 Token。

Prefill 完成后，最后一个输入位置的 logits 可用于采样第一个输出 Token。之后，模型进入常规 Decode，逐步生成后续 Token。

```text
输入 Prompt
    |
    v
Prefill：处理整个输入序列，建立 KV Cache
    |
    v
采样第一个输出 Token
    |
    v
Decode：生成后续 Token
    |
    v
直到生成结束或达到停止条件
```

#### 为什么要区分两阶段？

- Prefill 可以并行处理输入序列中的多个 Token，通常能形成较大的矩阵乘法。
- Decode 受自回归依赖约束，每条请求通常每轮只生成一个 Token，并反复读取历史 KV Cache。
- 两阶段的性能瓶颈、批处理方式和优化目标可能不同，因此推理系统通常会分别分析和优化它们。

#### 面试主答案

> Prefill 阶段处理完整的输入 Prompt，计算各层表示并建立 KV Cache；Decode 阶段基于历史 KV Cache 自回归地生成后续 Token。Prefill 能并行处理多个输入 Token，而标准 Decode 受 Token 之间的自回归依赖约束，每条请求每轮通常生成一个 Token。因此，两阶段的计算模式和性能瓶颈不同，需要采用不同的性能优化与调度策略。

---

### Q2. Performance Bottlenecks

**问题：Prefill 和 Decode 各自的计算瓶颈是什么？为什么同一个模型在不同阶段的瓶颈不同？**

#### 典型结论

- **Prefill 通常更偏计算密集型（Compute-bound）。**
- **Decode 通常更偏显存带宽受限（Memory-bandwidth-bound），尤其是小 Batch 时。**

这是常见趋势，并非所有模型、Batch Size 和上下文长度下都绝对成立。

#### 1. Prefill 为什么通常偏计算密集型？

以线性层为例：

```math
Y = XW
```

其中，$X$ 是输入特征，$W$ 是权重矩阵，$Y$ 是输出特征。

假设输入形状为 $X \in \mathbb{R}^{B \times S \times D}$，其中 $B$ 是 Batch Size，$S$ 是序列长度，$D$ 是隐藏维度。

Prefill 时，多个输入 Token 可以一起参与矩阵运算，形成规模较大的矩阵乘法，更容易利用 GPU Tensor Core 等计算单元。长 Prompt 还会带来较大的 Attention 计算量。

#### 2. Decode 为什么通常偏显存带宽受限？

小 Batch Decode 时，每条请求每轮只输入一个新 Token，但仍然需要读取模型权重，并读取历史 KV Cache 参与 Attention。

此时：

- 单次矩阵乘法规模相对较小，可能无法充分利用计算单元。
- 每轮需要读取模型权重，但单个请求对应的计算量有限。
- Attention 需要读取历史 KV Cache，上下文越长，相关访存压力通常越大。
- 权重和 KV Cache 的数据搬运成本可能成为性能瓶颈。

因此，Decode 容易受到显存带宽限制。

#### 3. 什么是算术强度（Arithmetic Intensity）？

算术强度衡量单位数据搬运量对应的计算量：

```math
\text{Arithmetic Intensity}
=
\frac{\text{FLOPs}}{\text{Bytes Transferred}}
```

其中：

- FLOPs：浮点运算次数。
- Bytes Transferred：数据搬运量，通常以字节为单位统计。

算术强度较高时，更可能受计算峰值限制；算术强度较低时，更可能受内存带宽限制。

这是基于 Roofline Model 的直观判断。实际瓶颈还取决于硬件的计算峰值与带宽、Kernel 效率和并行度。

#### 4. 哪些因素会改变结论？

- **Batch Size**：Decode 的 Batch Size 增大后，多个请求可以共享权重读取，提高计算利用率。
- **上下文长度**：长上下文会增加每轮 Decode 读取历史 KV Cache 的成本。
- **模型结构**：GQA、MQA、MLA 等结构会影响 KV Cache 的大小和访问方式。
- **Kernel 实现**：融合算子、访存效率和并行策略都会改变实际性能。
- **硬件特征**：不同 GPU 或 NPU 的计算峰值与内存带宽不同。

#### 面试主答案

> Prefill 通常偏计算密集型，因为它可以并行处理大量 Prompt Token，形成较大的矩阵乘法并充分利用 GPU 计算单元。Decode 通常偏显存带宽受限，因为每条请求每轮只处理一个新 Token，却仍需读取模型权重和历史 KV Cache，尤其在小 Batch 下算术强度较低。但实际瓶颈取决于 Batch Size、上下文长度、模型结构、硬件和 Kernel 实现，不能绝对化。

---

### Q3. Computation Estimation

**问题：Prompt 长度为 4096，生成 256 个 Token，Prefill 和 Decode 的计算量占比是多少？**

#### 先明确计算口径

题目没有指定模型结构和 FLOPs 统计口径，因此不能仅凭 Prompt 长度与生成长度，得到唯一精确的总计算量比例。

需要区分：

1. 主要线性层与 MLP 的计算量。
2. Attention 的计算量。
3. 整个模型的总 FLOPs。

#### 1. 简化估算：只看主要的逐 Token 计算

如果假设每个 Token 对线性层和 MLP 的计算成本近似相同，并暂时忽略 Attention 随序列长度变化的额外项：

```math
F_{\text{prefill}} \propto P
```

```math
F_{\text{decode}} \propto G
```

代入：

```math
P = 4096,\qquad G = 256
```

得到：

```math
F_{\text{prefill}}:F_{\text{decode}}
\approx 4096:256 = 16:1
```

因此，在这个简化模型下：

- Prefill：约占 $16/17 \approx 94.1\%$。
- Decode：约占 $1/17 \approx 5.9\%$。

**注意：这是忽略 Attention 序列长度影响后的简化估算，不是精确的总 FLOPs 占比。**

#### 2. 为什么精确计算更复杂？

线性层和 MLP 的计算量大致随 Token 数线性增长；Attention 的计算量还与序列长度有关。

设：

- $P$：Prompt 长度。
- $G$：生成 Token 数。
- $a$：近似表示线性层、MLP 等每 Token 对应的计算量系数。
- $b$：近似表示 Attention 计算量系数。

可以用下面的简化模型描述：

```math
F_{\text{prefill}} \approx aP+bP^2
```

```math
F_{\text{decode}}
\approx aG+b\left(GP+\frac{G(G-1)}{2}\right)
```

这里的 Decode Attention 项，近似表示生成阶段每一步对 Prompt 和此前生成 Token 进行 Attention 的累计计算量。

实际计算还会受到 Attention 类型、层数、隐藏维度、GQA/MQA、Prefill 与 Decode 的边界定义以及具体实现影响。

#### 面试主答案

> 如果只按主要线性层和 MLP 的逐 Token 计算量做简化估算，Prefill 与 Decode 的计算量约为 4096:256，也就是 16:1，对应约 94.1% 和 5.9%。但这不是精确的总 FLOPs 比例，因为 Prefill 的 Attention 计算与 Prompt 长度的平方相关，而 Decode 的 Attention 还会逐步读取历史上下文。要得到精确比例，需要明确模型结构和计算量统计口径。

---

### Q4. Autoregressive Decode

**问题：为什么 Decode 每次通常只处理一个 Token？能不能一次 Decode 多个 Token？**

#### 1. 为什么标准自回归 Decode 每轮通常生成一个 Token？

因为下一个 Token 的预测依赖之前已经生成的 Token。

自回归语言模型将输出序列的联合概率分解为：

```math
p(y_1,\ldots,y_G \mid x)
=
\prod_{t=1}^{G} p(y_t \mid x,y_{1:t-1})
```

其中：

- $x$：输入 Prompt。
- $y_t$：第 $t$ 个输出 Token。
- $y_{1:t-1}$：当前 Token 之前已经生成的 Token。

标准自回归生成必须先确定当前 Token，才能继续生成下一个 Token。

#### 2. 每条请求每轮生成一个 Token，不代表 GPU 每轮只能处理一个 Token

例如，Batch Size 为 32 时，Decode 可以在同一轮中为 32 条请求各生成一个 Token。

这是**跨请求并行**，而不是单条请求一次生成 32 个未知 Token。

#### 3. 能否一次推进多个 Token？可以

投机解码（Speculative Decoding）是一种常见方案：

1. Draft Model 或其他候选生成机制先提出多个候选 Token。
2. Target Model 对候选序列进行验证。
3. 在符合验证规则的前提下接受多个候选 Token。
4. 减少 Target 模型逐 Token 串行执行的次数。

MTP、Medusa、EAGLE 等方法也可以采用不同机制提出多个候选，但具体实现和模型要求有所不同。

#### 面试主答案

> 标准自回归 Decode 每轮通常只生成一个 Token，是因为下一个 Token 的预测依赖之前已经生成的 Token。但这不代表 GPU 每轮只能处理一个 Token：连续批处理中可以同时为多个请求生成 Token。此外，投机解码可以先提出候选序列，再由 Target 模型并行验证，从而在一次验证中接受多个 Token，减少串行 Decode 步数。

---

### Q5. Chunked Prefill

**问题：Chunked Prefill 的工作原理是什么？它解决了什么问题，又带来了什么新问题？**

#### 1. 问题背景：长 Prefill 可能干扰 Decode

假设一个很长的 Prompt 请求 A 与多个 Decode 请求 B、C、D 同时运行。如果 A 的 Prefill 一次占用较大的执行预算，其他请求可能需要等待，导致 Decode 的 Token 间延迟（TPOT）升高。

```text
不分块（示意）：

A：████████████████████  长 Prefill
B：                     等待 / Decode 延后
C：                     等待 / Decode 延后
D：                     等待 / Decode 延后
```

#### 2. Chunked Prefill 的核心机制

把长 Prompt 切成多个较小的 Token Chunk，由调度器分轮安排执行。这样 Prefill 不必一次消耗全部 Token 预算，可以与 Decode 请求更灵活地共享执行机会。

```text
分块后（示意）：

第 1 轮：A 的 Chunk 1 + 已调度的 Decode 请求
第 2 轮：A 的 Chunk 2 + 已调度的 Decode 请求
第 3 轮：A 的 Chunk 3 + 已调度的 Decode 请求
...
```

这只是概念示意，实际是否同轮执行、优先级如何安排，取决于具体推理引擎的调度策略、Token 预算和资源约束。

以 vLLM 的典型调度配置为例，调度器可以优先安排 Decode，再利用剩余 Token 预算安排 Prefill Chunk。具体行为需要结合版本与配置确认。

#### 3. 它解决了什么问题？

- 降低长 Prefill 对 Decode 请求的干扰。
- 改善 Decode TPOT 和服务延迟的稳定性。
- 提高调度灵活性，让长 Prompt 能分多轮处理。
- 在合适的负载和配置下，改善整体服务效率。

#### 4. 它带来了什么代价？

- 长 Prompt 可能需要更多轮才能完成，TTFT 可能增加。
- Chunk 太小可能增加调度、Kernel Launch 等开销。
- Chunk 大小、Token 预算和请求优先级需要权衡。
- 总吞吐不保证一定提升，实际结果依赖工作负载与实现。

| 指标 / 维度 | 可能的影响 |
|---|---|
| Decode TPOT | 通常更容易控制长 Prefill 的干扰 |
| TTFT | 取决于负载和调度策略，可能增加或改善 |
| 调度灵活性 | 提高 |
| 执行开销 | Chunk 太小时可能增加 |
| 总吞吐 | 取决于配置、负载和实现，不能保证总是提升 |

#### 5. 与 Diffusion 步级调度的联系

两者都利用执行边界提高调度灵活性：

- Chunked Prefill：把输入序列拆成多个 Chunk，分轮执行。
- Diffusion 步级抢占：在去噪步骤边界调整活动请求集合，并在需要时暂停和恢复请求。

二者的调度思想有相似之处，但机制并不相同：前者是序列分块执行，后者是请求在去噪步骤边界的抢占与恢复。

#### 面试主答案

> Chunked Prefill 将长 Prompt 拆成多个较小的 Token Chunk，由调度器分轮处理，使 Prefill 能与 Decode 请求更灵活地共享计算资源，减少长 Prefill 对 Decode TPOT 的干扰。代价是长 Prompt 可能需要更多轮才能完成，Chunk 太小还会增加调度和执行开销。因此需要结合 Chunk 大小、Token 预算和负载，在 Decode 延迟、TTFT 与整体吞吐之间进行权衡。

---

## 三、快速复习卡片

| 主题 | 一句话记忆 |
|---|---|
| Prefill / Decode | Prefill 处理输入并建立 KV Cache；Decode 自回归生成后续 Token。 |
| 性能瓶颈 | Prefill 通常偏计算密集；小 Batch Decode 通常偏显存带宽受限。 |
| 算术强度 | FLOPs 与数据搬运量之比，用于分析计算瓶颈与带宽瓶颈。 |
| 计算量估算 | 4096:256 = 16:1 是简化估算，不是精确总 FLOPs 比例。 |
| Decode 的串行性 | 单条请求受自回归依赖约束；投机解码可验证多个候选 Token。 |
| Chunked Prefill | 长 Prompt 分块执行，改善调度灵活性，但需要权衡 TTFT、TPOT 与执行开销。 |

---

## 四、后续扩充规范

后续新增问题时，沿用以下规则：

1. **编号全局递增**：下一道新题从 Q6 开始，不重置编号。
2. **索引与答案分离**：在「问题索引」添加题目和链接，再在「问题与答案」中添加对应答案。
3. **链接使用稳定英文锚点**：例如 `#q6-kv-cache`，避免标题修改后链接难以维护。
4. **每道题统一结构**：
   - 问题
   - 核心概念
   - 原理与实现
   - Trade-off / 易错点
   - 面试主答案
5. **公式统一格式**：
   - 行内公式：`$P=4096$`
   - 独立公式：使用 ` ```math ` 代码围栏。
   - 不要将数学变量放进反引号，否则会被识别为代码。
6. **答案以理解为主**：估算题明确假设；涉及具体框架实现时，注明版本或配置前提。

### 新增题目模板

```markdown
### Q6. Example Topic

**问题：在这里填写问题。**

#### 核心概念

填写概念解释。

#### 原理与实现

填写原理、公式、代码或流程。

#### Trade-off / 易错点

填写适用条件、局限和容易混淆的概念。

#### 面试主答案

> 填写一段适合在面试中直接表达的答案。
```

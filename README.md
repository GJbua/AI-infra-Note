# AI Infra 面试知识库

> 面向 AI Infra / LLM Inference Engineer 的系统学习笔记，聚焦大模型基础架构、推理优化、分布式并行与模型量化，建立从模型计算原理到推理系统工程实践的知识体系。

## 一、项目介绍

本仓库主要用于 AI Infra 相关岗位的基础知识学习和技术面试准备。

不同于单纯罗列技术名词，本仓库采用 **知识框架 → 高频面试问题 → 原理分析 → 面试 QA → Markdown 笔记** 的学习方式。

重点关注：

- **大模型基础架构**：理解 Transformer、Attention、MoE 等核心模型结构。
- **推理优化与系统架构**：理解 LLM 推理流程、KV Cache、批处理及推理优化技术。
- **分布式并行策略**：理解 TP、PP、DP、EP 等并行策略及通信机制。
- **模型压缩与量化**：理解 INT8、INT4、AWQ、GPTQ 等量化方法及其工程实现。

学习目标是能够解释技术原理、分析性能瓶颈、理解系统设计，并结合实际项目回答工程问题。

---

## 二、核心知识模块

### 01. 大模型基础架构（LLM Architecture）

**核心目标：理解大模型如何计算，以及模型结构如何影响推理性能。**

主要内容：

- Transformer 整体架构与 Transformer Block
- Self-Attention 计算流程及 Tensor Shape
- MHA、MQA、GQA 的区别
- MLA 低秩压缩机制
- RoPE 位置编码
- LayerNorm、RMSNorm、FFN、SwiGLU
- MoE 架构与 Router
- 模型参数量、FLOPs、显存估算

### 02. 推理优化与系统架构（LLM Inference Optimization）

**核心目标：理解 LLM 推理过程，掌握提升吞吐、降低延迟和显存开销的主要技术。**

主要内容：

- Prefill 与 Decode 阶段
- TTFT、TPOT、ITL、Throughput
- KV Cache 与 PagedAttention
- FlashAttention 原理
- Roofline Model
- Compute-bound 与 Memory-bound
- Continuous Batching
- Chunked Prefill
- Prefill-Decode Disaggregation
- MTP（Multi-Token Prediction）
- GPU 利用率与推理性能分析
- P95/P99 延迟及实验设计

### 03. 分布式并行策略（Distributed Parallelism）

**核心目标：理解大模型如何跨 GPU 部署，以及计算和通信如何协同。**

主要内容：

- Tensor Parallelism（TP）
- Pipeline Parallelism（PP）
- Data Parallelism（DP）
- Expert Parallelism（EP）
- Sequence Parallelism（SP）
- All-Reduce、All-Gather、Reduce-Scatter、All-to-All
- 计算通信重叠（Compute-Communication Overlap）
- 并行策略组合与部署场景选择

### 04. 模型压缩与量化（Model Quantization）

**核心目标：理解模型量化如何降低显存和计算开销，以及如何控制量化损失。**

主要内容：

- FP16、BF16、FP8、INT8、INT4
- W8A8、W4A16
- 在线量化与离线量化
- Per-Tensor、Per-Channel、Per-Group
- Quantization Scale 与 Zero Point
- 量化误差与精度评估
- 层敏感性分析
- AWQ 与 GPTQ
- 量化部署及推理性能优化

---

## 三、知识模块之间的联系

四个模块并不是完全独立的，而是围绕大模型推理系统逐步展开。

| 基础知识 | 关联技术 | 解决的问题 |
|---|---|---|
| Transformer / Attention | FlashAttention | 降低 Attention 访存开销 |
| MHA / GQA / MLA | KV Cache / PagedAttention | 降低 KV Cache 显存开销 |
| MoE | Expert Parallelism | 扩展模型容量与分布式部署 |
| FLOPs / 显存估算 | Roofline Model | 判断计算与访存瓶颈 |
| 模型精度 | INT8 / INT4 / AWQ | 降低权重显存与计算成本 |
| 推理阶段特征 | Continuous Batching | 提高 GPU 资源利用率 |
| Prefill / Decode | Chunked Prefill / PD 分离 | 优化不同阶段的资源利用 |

其中，性能分析与实验设计融入推理优化模块，不单独建立一个知识模块。

---

## 四、工程实践与项目关联

理论知识最终需要落到真实推理系统中。

本仓库的工程实践主要结合 vLLM-Omni 扩散模型在线推理优化项目。

重点关注：

**1. SLO 感知步级调度**

- Step Execution 与 Continuous Batching
- LSF-Preempt：步边界抢占调度
- RiskGuard：风险分级保护调度
- 请求状态保存、暂停与恢复
- SLO Goodput、Deadline Miss Ratio、P95/P99

**2. DiT / VAE 解耦与流水线执行**

- DiT 与 VAE Disaggregation
- Worker 进程管理与 IPC
- 异步任务通信
- VAE Decode 与 DiT Denoising 流水线
- 资源分配与吞吐量分析

项目代码分析统一基于 vLLM-Omni v0.20.0。

---

## 五、学习与整理规范

### 5.1 学习流程

每个知识模块采用统一的学习方式：

1. **知识框架**：明确模块包含哪些核心知识点。
2. **高频问题**：筛选实际面试中值得掌握的问题。
3. **独立作答**：一次性回答问题，检验当前知识掌握程度。
4. **原理复盘**：针对回答中的错误、遗漏和薄弱点进行系统学习。
5. **笔记整理**：将最终理解整理成 GitHub Markdown。

### 5.2 面试题重要性

采用五级星级标注：

| 星级 | 含义 |
|---|---|
| ★★★★★ | 核心高频问题，必须掌握 |
| ★★★★☆ | 高频问题，需要熟练理解 |
| ★★★☆☆ | 常见追问，掌握基本原理 |
| ★★☆☆☆ | 低频问题，了解即可 |
| ★☆☆☆☆ | 暂不作为重点 |

### 5.3 答案整理

每个核心问题尽量保留两种版本：

- **理解版**：完整解释原理、实现机制、优缺点及适用场景。
- **面试版**：精简表达，适合面试时直接回答。

优先保证原理正确，再追求回答简洁。

### 5.4 数学公式规范

本仓库使用 GitHub Markdown 数学公式语法。

行内公式使用单美元符号：

`$O(N^2D)$`

行间公式统一使用三个反引号加 `math`：

```math
E=x-\hat{x}
```

禁止使用 `$$...$$` 作为行间公式格式。

---

## 六、学习进度

| 模块 | 当前进度 |
|---|---|
| 01. 大模型基础架构 | 第一轮整理完成 |
| 02. 推理优化与系统架构 | 待整理 |
| 03. 分布式并行策略 | 已整理 |
| 04. 模型压缩与量化 | 已整理 |

知识库持续更新，已完成模块仍会根据后续面试反馈进行补充和修订。

---

## 七、学习范围与侧重点

本仓库以 AI Infra / 推理系统开发岗位的面试需求为导向，不追求覆盖所有底层技术。

- **硬件与 Kernel**：不单独建立模块，在 FlashAttention、Roofline 和性能分析中理解必要原理。
- **OS / IPC**：不单独建立模块，结合多进程 Worker、异步通信和实际工程项目掌握。
- **性能分析与实验设计**：归入推理优化模块。
- **简历与项目 QA**：结合实际项目独立复盘，不重复建立基础知识模块。

优先掌握高频知识、核心原理和工程实践，避免在低收益的细节上投入过多时间。

---

**目标：建立完整的 AI Infra 知识体系，能够从模型架构、计算过程、显存管理、分布式通信到推理系统优化，独立分析问题并给出工程解决方案。**

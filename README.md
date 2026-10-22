# AI-infra学习笔记
### 自回归语言模型

自回归语言模型将输出序列的联合概率分解为：

$$
p(y_1,\ldots,y_G \mid x)
=
\prod_{t=1}^{G} p(y_t \mid x,y_{<t})
$$

其中：

- $x$：输入 Prompt。
- $y_t$：第 $t$ 个输出 Token。
- $y_{<t}$：当前 Token 之前已经生成的 Token。

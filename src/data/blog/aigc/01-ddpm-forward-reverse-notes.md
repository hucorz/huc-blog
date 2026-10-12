---
title: 生成扩散模型漫谈 01：DDPM：从逐步加噪到预测噪声
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - DDPM
description: 从逐步加噪、反向重建和降低方差三个环节梳理 DDPM 的核心推导。
---

> [《生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼》](https://spaces.ac.cn/archives/9119) —— 苏剑林

## 记号约定

原文把单步正向过程写成 $x_t=\alpha_t x_{t-1}+\beta_t\varepsilon_t$，并规定 $\alpha_t^2+\beta_t^2=1$。其中 $\varepsilon_t\sim\mathcal N(0,I)$，$\alpha_t$ 是保留原信号的系数，$\beta_t$ 是加入新噪声的系数。

这套记号与 DDPM 论文和常见教程不完全相同：

| 含义         |       本文记号 |                                               常见 DDPM 记号 |
| ------------ | -------------: | -----------------------------------------------------------: |
| 单步信号系数 |     $\alpha_t$ |                               $\sqrt{\alpha_t^{\text{std}}}$ |
| 单步噪声系数 |      $\beta_t$ | $\sqrt{1-\alpha_t^{\text{std}}}=\sqrt{\beta_t^{\text{std}}}$ |
| 累计信号系数 | $\bar\alpha_t$ |                           $\sqrt{\bar\alpha_t^{\text{std}}}$ |
| 累计噪声系数 |  $\bar\beta_t$ |                         $\sqrt{1-\bar\alpha_t^{\text{std}}}$ |

看其他资料时不能只按符号名字对照，要先判断它表示的是系数、方差还是累计乘积。

## 一、正向过程：从 $x_{t-1}$ 到 $x_t$

每一步先缩小上一时刻的信号，再加入一份新的独立高斯噪声：

$$
x_t=\alpha_t x_{t-1}+\beta_t\varepsilon_t,
\qquad
\varepsilon_t\sim\mathcal N(0,I).
\label{eq:forward-step}
$$

若 $x_{t-1}$ 具有单位方差并且与 $\varepsilon_t$ 独立，那么 $\operatorname{Var}(x_t)=\alpha_t^2+\beta_t^2=1$。因此，$\alpha_t^2+\beta_t^2=1$ 的作用是让信号和噪声的比例不断变化，但总体尺度保持稳定。

### 从单步公式得到任意时刻的 $x_t$

以两步为例：

$$
\begin{aligned}
x_2
&=\alpha_2x_1+\beta_2\varepsilon_2\\
&=\alpha_2\alpha_1x_0
+\alpha_2\beta_1\varepsilon_1
+\beta_2\varepsilon_2.
\end{aligned}
$$

展开到任意 $t$，信号部分的系数始终是所有 $\alpha$ 的乘积，因此记 $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$。剩余部分则是多个相互独立的高斯噪声的加权和；这个加权和仍然服从高斯分布，所以可以把它写成“一个总系数乘以一个标准高斯噪声”。由此定义 $\bar\beta_t$ 和 $\bar\varepsilon_t$：

$$
\boxed{
x_t
=\bar\alpha_t x_0
+\bar\beta_t\bar\varepsilon_t,
\qquad
\bar\varepsilon_t\sim\mathcal N(0,I)
}
\label{eq:direct-sampling}
$$

这里的 $\bar\beta_t$ 表示所有历史噪声合并后的总系数，并不是 $\beta_1\beta_2\cdots\beta_t$。它的具体取值还需要从递推关系中算出来。

> 已知每一步都有 $\alpha_t^2+\beta_t^2=1$，为什么累计后仍然能写成 $\bar\alpha_t^2+\bar\beta_t^2=1$？

这个关系可以用递推来验证。当 $t=1$ 时，$\bar\alpha_1=\alpha_1$、$\bar\beta_1=\beta_1$，所以它就是已知条件 $\alpha_1^2+\beta_1^2=1$。再假设 $t-1$ 时关系成立，即 $\bar\alpha_{t-1}^2+\bar\beta_{t-1}^2=1$，并把 $x_{t-1}=\bar\alpha_{t-1}x_0+\bar\beta_{t-1}\bar\varepsilon_{t-1}$ 代入单步正向公式 \eqref{eq:forward-step}：

$$
\begin{aligned}
x_t
&=\alpha_t\bar\alpha_{t-1}x_0
+\alpha_t\bar\beta_{t-1}\bar\varepsilon_{t-1}
+\beta_t\varepsilon_t.
\end{aligned}
$$

历史噪声与当前噪声相互独立，所以它们合并后的方差等于两个系数的平方和：

$$
\boxed{
\bar\beta_t^2
=\alpha_t^2\bar\beta_{t-1}^2+\beta_t^2
}
\label{eq:cumulative-noise}
$$

利用归纳假设、单步条件 $\alpha_t^2+\beta_t^2=1$，以及 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$：

$$
\begin{aligned}
\alpha_t^2\bar\beta_{t-1}^2+\beta_t^2
&=\alpha_t^2(1-\bar\alpha_{t-1}^2)+\beta_t^2\\
&=(\alpha_t^2+\beta_t^2)
-\alpha_t^2\bar\alpha_{t-1}^2\\
&=1-\bar\alpha_t^2.
\end{aligned}
$$

因此 $\bar\beta_t=\sqrt{1-\bar\alpha_t^2}$，也就得到 $\bar\alpha_t^2+\bar\beta_t^2=1$。公式 \eqref{eq:direct-sampling} 表示的是分布意义上的等价，不是说合并前后的每次采样数值都相同。它的实际价值是：训练时不必依次计算 $x_1,x_2,\ldots,x_t$，给定 $x_0$ 和 $t$ 后，一次采样就能直接得到 $x_t$。

Noise Scheduler 会让 $\bar\alpha_T\approx0$，所以终点近似为 $x_T\sim\mathcal N(0,I)$。

## 二、反向过程：从 $x_t$ 重建 $x_{t-1}$

正向过程已经确定，生成时需要学习相反方向的 $x_t\rightarrow x_{t-1}$。最直接的训练方式，是让模型 $\mu_\theta(x_t,t)$ 最小化 $\|x_{t-1}-\mu_\theta(x_t,t)\|^2$。

由单步正向公式 \eqref{eq:forward-step} 可以反解出 $x_{t-1}=\frac{1}{\alpha_t}(x_t-\beta_t\varepsilon_t)$。因此把反向模型写成下面的形式：

$$
\mu_\theta(x_t,t)
=\frac{1}{\alpha_t}
\left[x_t-\beta_t g_\theta(x_t,t)\right].
\label{eq:reverse-parameterization}
$$

代回重建损失后，只剩下一个与模型参数无关的系数：

$$
\left\|x_{t-1}-\mu_\theta(x_t,t)\right\|^2
=\frac{\beta_t^2}{\alpha_t^2}
\left\|\varepsilon_t-g_\theta(x_t,t)\right\|^2.
$$

所以，重建 $x_{t-1}$ 可以等价地参数化为预测第 $t$ 步加入的噪声 $\varepsilon_t$。

### 最初的训练目标为什么需要两个噪声

此时模型要预测的是最后一步噪声 $\varepsilon_t$，因此构造输入时必须保留它与 $x_t$ 的真实关系：

$$
x_t
=\bar\alpha_t x_0
+\alpha_t\bar\beta_{t-1}\bar\varepsilon_{t-1}
+\beta_t\varepsilon_t.
$$

于是最初的损失是：

$$
L_1
=\left\|
\varepsilon_t
-g_\theta\left(
\bar\alpha_t x_0
+\alpha_t\bar\beta_{t-1}\bar\varepsilon_{t-1}
+\beta_t\varepsilon_t,
t
\right)
\right\|^2.
\label{eq:two-noise-loss}
$$

这个目标是正确的，但一次训练需要采样数据 $x_0$、时间步 $t$，以及两份高斯噪声 $\bar\varepsilon_{t-1}$ 和 $\varepsilon_t$。要采样的随机变量越多，就越难用一个 batch 准确估计损失，表现为每次估计的波动（方差）更大。如果能对其中一份噪声直接求期望，就不必再用随机采样近似它，也就能降低损失估计的方差。

下一节并不是要改变正向过程或预测任务，而是要把公式 \eqref{eq:two-noise-loss} 中一部分可以精确计算的随机性直接平均掉，减少用随机采样近似期望的次数。

## 三、降低方差：从两份噪声到 $\varepsilon$

输入 $x_t$ 中的两份噪声可以根据正态分布的叠加性合并，但损失的目标仍然是其中的 $\varepsilon_t$。真正的难点不是“怎样合并输入”，而是“合并之后怎样继续保持输入与目标之间原有的相关性”。

### 先只看 $aU+bV$ 与预测目标 $V$

为了看清结构，临时使用下面四个简写：

| 临时记号 | 原变量                    | 含义                      |
| -------- | ------------------------- | ------------------------- |
| $a$      | $\alpha_t\bar\beta_{t-1}$ | 历史噪声进入 $x_t$ 的系数 |
| $b$      | $\beta_t$                 | 当前噪声进入 $x_t$ 的系数 |
| $U$      | $\bar\varepsilon_{t-1}$   | 历史累计噪声              |
| $V$      | $\varepsilon_t$           | 当前噪声，也是预测目标    |

把这些简写代入公式 \eqref{eq:two-noise-loss}，损失可以写成：

$$
L_1
=\left\|
V-g_\theta(\bar\alpha_t x_0+aU+bV,t)
\right\|^2.
$$

这一步只是换了记号，没有改变损失。现在可以直接看出，模型输入中的噪声是 $aU+bV$，预测目标是 $V$；接下来的变换必须同时处理二者，不能只保证输入的分布不变。

由累计噪声关系 \eqref{eq:cumulative-noise} 可知 $a^2+b^2=\bar\beta_t^2$，所以输入噪声的边缘分布可以写成 $aU+bV\overset d=\bar\beta_t\varepsilon$。

> 既然输入中的两份噪声已经可以合并成一个 $\varepsilon$，为什么不能再独立采样一个 $V$ 作为预测目标？

因为输入噪声与目标并不独立：$\operatorname{Cov}(aU+bV,V)=bI$。如果重新独立采样 $V$，协方差会变成 $0$，训练数据的联合关系就被改变了。

另一方面，知道 $aU+bV$ 也无法唯一恢复 $V$。例如 $a=b=1$ 时，即使知道 $U+V=0$，$(U,V)$ 仍然可以有无数种取值。输入中包含一部分关于 $V$ 的信息，但没有包含完整的 $V$。因此，还需要表示“给定输入后，$V$ 剩余的不确定性”。

### 补回 $V$ 中剩余的不确定性

为表示这部分不确定性，在输入方向 $aU+bV$ 之外，再取一个与它正交的组合，并把它记为 $\omega$：

$$
\varepsilon=\frac{aU+bV}{\bar\beta_t},
\qquad
\omega=\frac{bU-aV}{\bar\beta_t}.
$$

这只是把原来的 $(U,V)$ 旋转成新坐标 $(\varepsilon,\omega)$，没有增加新的随机性。由于旋转矩阵是正交矩阵，$\varepsilon$ 和 $\omega$ 仍然是相互独立的标准高斯变量。

反解目标 $V$：

$$
\boxed{
V
=\underbrace{\frac{b}{\bar\beta_t}\varepsilon}_{\text{输入中能够确定的部分}}
-\underbrace{\frac{a}{\bar\beta_t}\omega}_{\text{给定输入后仍不确定的部分}}
}
\label{eq:target-decomposition}
$$

公式 \eqref{eq:target-decomposition} 说明了 $\omega$ 的作用：$\omega$ 本身与输入独立，但用它构造出来的 $V$ 仍然与输入保持原有的相关性。

如果直接令 $V=\frac{b}{\bar\beta_t}\varepsilon$，虽然保留了相关部分，但 $V$ 的方差只有 $\frac{b^2}{\bar\beta_t^2}I$，不再是单位方差；缺少的方差正是 $-\frac{a}{\bar\beta_t}\omega$ 提供的。

### 对 $\omega$ 求期望

将公式 \eqref{eq:target-decomposition} 代回原损失，模型输入只依赖 $\varepsilon$：

$$
\begin{aligned}
&\mathbb E_\omega
\left[
\left\|
\frac{b}{\bar\beta_t}\varepsilon
-\frac{a}{\bar\beta_t}\omega
-g_\theta(\bar\alpha_t x_0+\bar\beta_t\varepsilon,t)
\right\|^2
\right]\\
&=\left\|
\frac{b}{\bar\beta_t}\varepsilon
-g_\theta(\bar\alpha_t x_0+\bar\beta_t\varepsilon,t)
\right\|^2
+\frac{a^2}{\bar\beta_t^2}\mathbb E\|\omega\|^2.
\end{aligned}
$$

这里先固定 $\varepsilon$，只对 $\omega$ 求期望。交叉项因为 $\mathbb E[\omega]=0$ 而消失，最后一项又与模型参数无关，因此训练时不需要真的采样 $\omega$。

再缩放网络输出，令 $\hat\varepsilon_\theta=\frac{\bar\beta_t}{\beta_t}g_\theta$。与模型有关的部分会变成下面的形式，前面还带有一个只依赖 $t$ 的权重。常见的简化目标会去掉这个权重：

$$
\boxed{
L_{\text{simple}}
=\left\|
\varepsilon
-\hat\varepsilon_\theta(
\bar\alpha_t x_0+\bar\beta_t\varepsilon,
t)
\right\|^2
}
\label{eq:simple-loss}
$$

坐标旋转本身并没有减少采样数量：原来采样 $(U,V)$，旋转后仍然要采样 $(\varepsilon,\omega)$。真正降低方差的是利用平方损失精确计算 $\mathbb E_\omega$，最终只需要采样 $\varepsilon$。去掉随 $t$ 变化的权重还会改变不同时间步在总损失中的相对重要性，所以公式 \eqref{eq:simple-loss} 是常用的简化训练目标，不是与原损失完全相等的代数变形。

## 四、最终的训练与生成

**训练。** 根据直接采样公式 \eqref{eq:direct-sampling}，一次训练只需要：

```text
采样真实数据 x₀
采样时间步 t
采样 ε ~ N(0, I)

xₜ = ᾱₜ x₀ + β̄ₜ ε
ε̂ = ε̂θ(xₜ, t)
loss = ||ε - ε̂||²
```

最终网络预测的是从 $x_0$ 一步构造 $x_t$ 时使用的累计噪声 $\varepsilon$，不是推导开始时的最后一步噪声 $\varepsilon_t$。两种参数化之间的缩放已经吸收到 $\hat\varepsilon_\theta$ 的定义中。

**生成。** 从 $x_T\sim\mathcal N(0,I)$ 开始。把 $g_\theta=\frac{\beta_t}{\bar\beta_t}\hat\varepsilon_\theta$ 代入反向参数化公式 \eqref{eq:reverse-parameterization}，得到：

$$
\boxed{
\mu_\theta(x_t,t)
=\frac{1}{\alpha_t}
\left[
x_t
-\frac{\beta_t^2}{\bar\beta_t}
\hat\varepsilon_\theta(x_t,t)
\right]
}
\label{eq:reverse-mean}
$$

确定性地取均值时令 $x_{t-1}=\mu_\theta(x_t,t)$；随机采样时令 $x_{t-1}=\mu_\theta(x_t,t)+\sigma_tz$，其中 $z\sim\mathcal N(0,I)$。从 $t=T$ 迭代到 $1$，最终得到 $x_0$。

$\sigma_t$ 的选择属于反向分布方差的设计。原文为了说明主线，使用了 $\sigma_t=\beta_t$ 这一简单选择。

## 五、步数与 Noise Scheduler

DDPM 原论文取 $T=1000$。把原论文的设置换成本文的记号，$\alpha_t$ 大致可以写成：

$$
\alpha_t=\sqrt{1-\frac{0.02t}{T}}.
$$

它是一个单调递减函数：越往后，单步保留的信号越少，加入的噪声越多。这个设置与较大的 $T$ 可以放在一起理解。

**为什么需要较大的 $T$。** 这里使用欧氏距离的平方学习 $x_t$ 到 $x_{t-1}$ 的重建。欧氏距离不是衡量图像真实程度的好指标，VAE 使用欧氏距离做重建时往往会得到模糊的结果；只有输入、输出两张图像足够接近时，欧氏距离才比较容易得到清晰的重建。增大 $T$ 可以缩小相邻状态之间的差距，让每一步的重建任务更适合使用欧氏距离。代价是生成时需要反复调用网络，原始 DDPM 的采样因此很慢。

**为什么 $\alpha_t$ 单调递减。** 当 $t$ 较小时，$x_t$ 仍然接近真实图像，需要让 $x_{t-1}$ 与 $x_t$ 尽可能接近，因此使用较大的 $\alpha_t$。当 $t$ 较大时，$x_t$ 已经接近纯噪声，欧氏距离带来的模糊问题不再重要，可以用较小的 $\alpha_t$ 增大单步变化。

> 能不能从头到尾都使用较大的 $\alpha_t$？

可以，但这意味着每一步加入的噪声都很少，因此需要更大的 $T$ 才能满足 $\bar\alpha_T\approx0$。对于上面的设置，可以估算：

$$
\begin{aligned}
\log\bar\alpha_T
&=\sum_{t=1}^T\log\alpha_t\\
&=\frac12\sum_{t=1}^T\log\left(1-\frac{0.02t}{T}\right)\\
&<-0.005(T+1).
\end{aligned}
$$

代入 $T=1000$，可以得到 $\bar\alpha_T$ 大约在 $e^{-5}$ 的量级，已经可以近似看成 $0$。如果始终使用更大的 $\alpha_t$，要达到相同的终点就只能继续增加步数。

不同时间步对应不同噪声强度，本来需要不同的去噪模型。DDPM 共享同一套网络参数，并把 $t$ 作为条件输入，让网络知道当前处于哪一个噪声阶段。

## 容易混淆的地方

- $aU+bV\overset d=\bar\beta_t\varepsilon$ 只说明输入噪声的边缘分布相同，不代表它与预测目标 $V$ 的联合关系也自动保留。
- 输入中“含有 $V$”不等于能够恢复完整的 $V$。给定输入后，$V$ 仍有剩余的不确定性。
- 独立的是 $\varepsilon$ 与 $\omega$，不是模型输入与 $V$。构造出的 $V$ 必须继续与输入相关。
- 引入 $\omega$ 只是为了正确重写联合分布；真正减少采样的是随后精确计算关于 $\omega$ 的期望。
- 推导开始时 $g_\theta$ 预测单步噪声 $\varepsilon_t$，最终的 $\hat\varepsilon_\theta$ 预测累计噪声 $\varepsilon$，两者之间存在随 $t$ 变化的缩放。

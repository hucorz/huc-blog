---
title: 生成扩散模型漫谈 18：Score Matching 与条件目标为何等价
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: true
tags:
  - AIGC
  - Diffusion
  - Score Matching
description: 推导边缘 Score 的条件期望恒等式，并区分总体目标等价与有限样本估计差异。
---

> [《生成扩散模型漫谈（十八）：得分匹配 = 条件得分匹配》](https://spaces.ac.cn/archives/9509) —— 苏剑林

## 边缘目标理论正确，却不能直接计算

给定数据 $x_0\sim p_0$ 和已知扰动核 $p_t(x_t\mid x_0)$，时刻 $t$ 的边缘分布是

$$
p_t(x_t)=\int p_t(x_t\mid x_0)p_0(x_0)dx_0.
\label{eq:sm18-marginal-density}
$$

标准 Score Matching 希望网络拟合边缘 Score：

$$
\mathcal L_{\mathrm{SM}}(\theta)
=\mathbb E_{x_t\sim p_t}
\left[
\|\nabla_{x_t}\log p_t(x_t)-s_\theta(x_t,t)\|^2
\right].
\label{eq:sm18-marginal-loss}
$$

$p_0$ 通常只有数据样本，没有可求导的解析密度，所以公式 \eqref{eq:sm18-marginal-density} 的积分和边缘 Score 都未知。形式上可以写成

$$
\nabla\log p_t(x_t)
=\frac{
\mathbb E_{x_0\sim p_0}[\nabla p_t(x_t\mid x_0)]
}{
\mathbb E_{x_0\sim p_0}[p_t(x_t\mid x_0)]
}.
$$

但用一个 batch 分别估计分子、分母再取比值，一般是有偏估计；分母小时还会不稳定。增大 batch 只能逐渐逼近总体比值，并没有把它变成理想的单样本训练标签。

实际扩散模型使用条件得分匹配：

$$
\mathcal L_{\mathrm{CSM}}(\theta)
=\mathbb E_{x_0\sim p_0,\,x_t\sim p_t(\cdot\mid x_0)}
\left[
\|\nabla_{x_t}\log p_t(x_t\mid x_0)-s_\theta(x_t,t)\|^2
\right].
\label{eq:sm18-conditional-loss}
$$

对高斯扰动，条件 Score 有解析式，因此每个独立样本对 $(x_0,x_t)$ 都能给出无偏的 Monte Carlo loss 与梯度估计。

## 边缘 Score 是条件 Score 的后验均值

从公式 \eqref{eq:sm18-marginal-density} 直接求导，并假设可以交换微分与积分：

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t)
&=\frac{1}{p_t(x_t)}
\int p_0(x_0)\nabla_{x_t}p_t(x_t\mid x_0)dx_0\\
&=\int
\frac{p_0(x_0)p_t(x_t\mid x_0)}{p_t(x_t)}
\nabla_{x_t}\log p_t(x_t\mid x_0)dx_0\\
&=\mathbb E\left[
\nabla_{x_t}\log p_t(x_t\mid x_0)
\mid x_t
\right].
\end{aligned}
\label{eq:sm18-score-identity}
$$

第二行只是乘除 $p_t(x_t\mid x_0)$，第三行使用 Bayes 公式识别出后验 $p_t(x_0\mid x_t)$。这个恒等式不要求扰动核是高斯；需要的是相关密度存在、导数与积分交换合法。

它也解释了为什么只输入 $(x_t,t)$ 的网络能够从依赖 $x_0$ 的监督中学到边缘 Score：对同一个 $x_t$，平方损失会把所有可能 $x_0$ 给出的条件 Score 平均起来。

## Jensen 不等式只给出了一半结论

令

$$
Y=\nabla_{x_t}\log p_t(x_t\mid x_0),
\qquad s=s_\theta(x_t,t).
$$

在固定 $x_t$ 后，公式 \eqref{eq:sm18-score-identity} 给出 $\mathbb E[Y\mid x_t]=\nabla\log p_t(x_t)$。由条件 Jensen 不等式，

$$
\|\mathbb E[Y\mid x_t]-s\|^2
\le
\mathbb E[\|Y-s\|^2\mid x_t].
$$

再对 $x_t$ 求期望：

$$
\mathcal L_{\mathrm{SM}}(\theta)
\le\mathcal L_{\mathrm{CSM}}(\theta).
$$

这说明条件目标是边缘目标的上界，但“最小化一个上界”本身还不能保证两个目标有完全相同的优化过程。更强的结果要把平方项完整展开。

## 两个目标只相差条件方差

使用条件均值的平方误差分解：

$$
\begin{aligned}
\mathbb E[\|Y-s\|^2\mid x_t]
&=\|\mathbb E[Y\mid x_t]-s\|^2\\
&\quad+\mathbb E[
\|Y-\mathbb E[Y\mid x_t]\|^2
\mid x_t].
\end{aligned}
$$

对 $x_t$ 求期望并代回 Score 恒等式：

$$
\boxed{
\mathcal L_{\mathrm{CSM}}(\theta)
=\mathcal L_{\mathrm{SM}}(\theta)+C_t
}
\label{eq:sm18-loss-equivalence}
$$

其中

$$
C_t
=\mathbb E_{x_t}
\left[
\mathbb E_{x_0\mid x_t}
\|\nabla\log p_t(x_t\mid x_0)\|^2
-\|\nabla\log p_t(x_t)\|^2
\right]
$$

是条件 Score 在给定 $x_t$ 后的方差总和，与 $\theta$ 无关。因此两个**总体目标**具有相同梯度、最优参数和优化地形；条件目标数值更大，只是加了一个模型无法改变的常数。

这里的常数通常依赖 $t$。若总训练目标还给不同时刻加权，那么每个 $t$ 上分别相差常数，积分后仍与 $\theta$ 无关；但如果权重本身可学习或依赖模型参数，就需要重新检查这个结论。

## 总体等价不等于估计器等价

> 既然公式 \eqref{eq:sm18-loss-equivalence} 说明两者等价，为什么直接估计边缘 Score 仍需要更大的 batch？

等价的是精确期望下的函数，不是任意有限样本实现。条件目标直接采样 $(x_0,x_t)$ 后计算一个解析标签，它的 batch 均值是总体条件目标的无偏估计。直接边缘目标若先用 batch 估计公式 \eqref{eq:sm18-marginal-density} 中的两个期望再取比值，会因为非线性的除法产生偏差；把这个有偏 Score 再放进平方损失，还会继续改变统计性质。

只有先用恒等式与平方展开把目标化成公式 \eqref{eq:sm18-conditional-loss}，才得到实用的无偏 Monte Carlo 形式。因此“SM = CSM”应理解为：经过解析化简后，CSM 是 SM 的等价可训练形式；不能理解为任何直接计算 SM 的小 batch 算法都与 CSM 一样稳定。

在高斯核 $x_t=\alpha_tx_0+\sigma_t\varepsilon$ 下，条件 Score 为 $-\varepsilon/\sigma_t$，再把网络缩放为噪声预测器，就得到常见的噪声 MSE。这条链把“预测噪声”与“学习未知边缘密度的对数梯度”严格连接起来，而不是只靠直觉类比。

---
title: DDPM 公式笔记
author: Huc
pubDatetime: 2026-06-20
modDatetime: 2026-06-20
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - DDPM
description: DDPM 正向加噪、反向去噪与训练目标的公式笔记。
---

# DDPM

这篇主要记公式，不展开太多概念解释。主线只有三件事：

1. 正向过程怎样逐步把数据加成高斯噪声。
2. 反向过程为什么要学习 $p_\theta(x_{t-1}|x_t)$。
3. 训练目标最后为什么会落到预测噪声 $\epsilon$。

---

## 正向加噪

最直接的想法是每一步都往前一个状态上加一点高斯噪声：

$$
x_t = x_{t-1} + \beta \epsilon
$$

相当于均值为 $x_{t-1}$，方差为 $\beta^2$ 的分布中采样。但这样写不太好，因为均值始终围绕前一个的样本点，噪声强度的控制也不够自然。DDPM 的做法是给每一步定义一个噪声系数序列：

$$
0 < \beta_1 < \beta_2 < \cdots < \beta_T < 1
$$

常见会把 $\beta_t$ 设得很小，例如从 `0.0001` 到 `0.02`。

于是正向过程写成：

$$
q(x_t|x_{t-1})
=
\mathcal N(x_t;\sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)
$$

也就是：

$$
x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t,
\qquad \epsilon_t \sim \mathcal N(0, I)
$$

记

$$
\alpha_t = 1-\beta_t
$$

则上式可以写成更常用的形式：

$$
q(x_t|x_{t-1})
=
\mathcal N(x_t;\sqrt{\alpha_t}\,x_{t-1}, (1-\alpha_t)I)
$$

$$
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t
$$

继续展开：

$$
x_1 = \sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\epsilon_1
$$

$$
x_2 = \sqrt{\alpha_2}x_1 + \sqrt{1-\alpha_2}\epsilon_2
$$

$$
\cdots
$$

记累计乘积

$$
\bar\alpha_t = \prod_{s=1}^{t}\alpha_s
$$

则可以得到闭式：

$$
q(x_t|x_0)
=
\mathcal N(x_t;\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)
$$

也就是：

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon \sim \mathcal N(0, I)
$$

所以正向过程的目标很明确：随着 $t$ 变大，$x_t$ 会逐渐接近均值为 $0$、方差为 $I$ 的标准高斯分布。

---

## 反向过程

如果正向过程是把 $x_0$ 一步步加噪到 $x_T$，那生成时就希望反过来：从高斯噪声开始，一步步还原出数据。

理想情况下，我们想要的是真实反向后验：

$$
q(x_{t-1}|x_t)
=
\frac{q(x_t|x_{t-1})q(x_{t-1})}{q(x_t)}
$$

但这个分布通常不好直接求，所以更常见的做法是考虑：

$$
q(x_{t-1}|x_t, x_0)
=
\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}
$$

由于 Markov 性质，

$$
q(x_t|x_{t-1},x_0)=q(x_t|x_{t-1})
$$

因此：

$$
q(x_{t-1}|x_t, x_0)
=
\frac{q(x_t|x_{t-1})q(x_{t-1}|x_0)}{q(x_t|x_0)}
$$

这一项是可以算的，因为分子分母全都是高斯分布。

于是训练时就让神经网络去拟合这个真实后验：

$$
p_\theta(x_{t-1}|x_t)
\approx
q(x_{t-1}|x_t, x_0)
$$

生成时只要从

$$
x_T \sim \mathcal N(0, I)
$$

开始，不断采样

$$
p_\theta(x_{t-1}|x_t)
$$

就可以一步步走回数据分布。

---

## 训练目标

给定样本 $x^1,x^2,\dots,x^n$，目标仍然是最大化训练数据的似然：

$$
\arg\max_\theta \prod_i p_\theta(x^i)
\quad \Longleftrightarrow \quad
\arg\max_\theta \sum_i \log p_\theta(x^i)
\quad \Longleftrightarrow \quad
\arg\min_\theta \sum_i -\log p_\theta(x^i)
$$

正向过程定义为：

$$
q(x_{1:T}|x_0)=\prod_{t=1}^{T} q(x_t|x_{t-1})
$$

反向生成过程定义为：

$$
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_t)
$$

对单个样本 $x_0$，从负对数似然开始：

$$
-\log p_\theta(x_0)
=
-\log \int p_\theta(x_{0:T})\,dx_{1:T}
$$

乘除同一个 $q(x_{1:T}|x_0)$：

$$
\begin{aligned}
-\log p_\theta(x_0)
&=
-\log \int q(x_{1:T}|x_0)
\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}
\,dx_{1:T} \\
&=
-\log \mathbb E_{q(x_{1:T}|x_0)}
\left[
\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}
\right]
\end{aligned}
$$

由于 $-\log$ 是凸函数，由 Jensen 不等式可得：

$$
\begin{aligned}
-\log p_\theta(x_0)
&\le
\mathbb E_{q(x_{1:T}|x_0)}
\left[
\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}
\right]
\end{aligned}
$$

后面继续展开会得到一个很长的式子，这里直接省略中间整理过程：

$$
\begin{aligned}
-\log p_\theta(x_0)
\le\;&
\mathbb E_q
\left[
\log \frac{q(x_T|x_0)}{p(x_T)}
\right] \\
&+
\mathbb E_q
\left[
\sum_{t=2}^{T}
\log \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)}
\right] \\
&-
\mathbb E_q
\left[
\log p_\theta(x_0|x_1)
\right]
\end{aligned}
$$

这里三项里：

1. 第一项没有可训练参数。
2. 第二项是每一步反向分布的拟合误差。
3. 第三项通常可以单独处理，或者在这类笔记里先略掉。

所以训练时最核心的是最小化第二项：

$$
\mathbb E_q
\left[
\sum_{t=2}^{T}
\log \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)}
\right]
$$

把求和和期望拆开，省略推导过程，此项可写成：

$$
\sum_{t=2}^{T}
\mathbb E_{q(x_t|x_0)}
\left[
D_{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))
\right]
$$

所以 DDPM 的训练，本质上是在每个时间步都让模型学会把真实后验

$$
q(x_{t-1}|x_t,x_0)
$$

拟合成模型的反向分布

$$
p_\theta(x_{t-1}|x_t)
$$

---

## 后验分布的均值和方差

因为

$$
q(x_{t-1}|x_t,x_0)
=
\frac{q(x_t|x_{t-1})q(x_{t-1}|x_0)}{q(x_t|x_0)}
$$

而右边都是高斯分布，所以这个后验本身也一定是高斯分布：

$$
q(x_{t-1}|x_t,x_0)
=
\mathcal N(x_{t-1};\tilde\mu_t(x_t,x_0), \tilde\beta_t I)
$$

中间配方过程比较长，这里直接记结果。

方差是：

$$
\tilde\beta_t
=
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
=
\frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
$$

均值可以写成：

$$
\tilde\mu_t(x_t,x_0)
=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t
$$

又因为

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
$$

所以

$$
x_0
=
\frac{1}{\sqrt{\bar\alpha_t}}
\left(
x_t-\sqrt{1-\bar\alpha_t}\epsilon
\right)
$$

代回去之后，可以把均值改写成更常用的形式：

$$
\tilde\mu_t(x_t,\epsilon)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon
\right)
$$

这一步很关键，因为它说明了：

> 只要网络能预测出噪声 $\epsilon$，就能反推出从 $x_t$ 到 $x_{t-1}$ 所需的均值。

---

## 最后落到预测噪声

上面已经看到，反向分布的方差 $\tilde\beta_t$ 可以直接由 schedule 算出来；而均值可以通过预测噪声来得到。

所以实际做法通常是让网络输出：

$$
\epsilon_\theta(x_t,t)
$$

然后用它代替真实噪声 $\epsilon$，得到反向过程的均值：

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)
\right)
$$

于是模型分布写成：

$$
p_\theta(x_{t-1}|x_t)
=
\mathcal N(x_{t-1};\mu_\theta(x_t,t), \tilde\beta_t I)
$$

再进一步，DDPM 常用的简化训练目标就是直接做噪声回归：

$$
\mathcal L_{\text{simple}}
=
\mathbb E_{x_0,\epsilon,t}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right]
$$

其中

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
$$

这就是最后最常见的训练形式。

换句话说，DDPM 看起来是在学习“反向去噪分布”，但落到实现上，通常就是让网络学会：

> 给定某一步的 noisy sample $x_t$ 和时间步 $t$，预测当时被加进去的噪声 $\epsilon$。

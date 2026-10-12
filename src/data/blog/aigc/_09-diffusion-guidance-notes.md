---
title: 生成扩散模型漫谈 09：扩散模型的条件引导
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: true
tags:
  - AIGC
  - Diffusion
  - Guidance
description: 从条件 score 推导 Classifier Guidance 与 Classifier-Free Guidance。
---

> [《生成扩散模型漫谈（九）：条件控制生成结果》](https://spaces.ac.cn/archives/9257) —— 苏剑林

## 条件生成改变的是反向分布

无条件生成学习 $p(x_{t-1}\mid x_t)$，加入条件 $y$ 后，目标变成
$p(x_{t-1}\mid x_t,y)$。两类常见方法的差异在于条件信息何时进入模型：

- Classifier Guidance 保留已有的无条件扩散模型，在采样时用额外判别模型修改反向一步；
- Classifier-Free Guidance 从训练开始就让去噪网络接收 $y$，采样时组合条件与无条件预测。

前者可以复用无条件模型，但每一步都要额外求条件模型对输入的梯度；后者推理结构更直接，却需要带条件数据重新训练扩散模型。

## 贝叶斯分解怎样引出分类器梯度

先在每一步应用条件贝叶斯：

$$
p(x_{t-1}\mid x_t,y)
=p(x_{t-1}\mid x_t)
\frac{p(y\mid x_{t-1},x_t)}{p(y\mid x_t)}.
\label{eq:diffusion-guidance-bayes-factor}
$$

正向马尔可夫链满足 $y\to x_{t-1}\to x_t$：一旦给定较干净的
$x_{t-1}$，再观察由它加噪得到的 $x_t$ 不应提供额外的 $y$ 信息。因此

$$
p(y\mid x_{t-1},x_t)=p(y\mid x_{t-1}).
$$

把公式 \eqref{eq:diffusion-guidance-bayes-factor} 写成指数形式：

$$
p(x_{t-1}\mid x_t,y)
\propto p(x_{t-1}\mid x_t)
\exp\left[
\log p(y\mid x_{t-1})-\log p(y\mid x_t)
\right].
$$

当反向一步足够小时，$x_{t-1}$ 接近 $x_t$，所以在 $x_t$ 处做一阶展开：

$$
\log p(y\mid x_{t-1})-\log p(y\mid x_t)
\approx
(x_{t-1}-x_t)^\top
\nabla_{x_t}\log p(y\mid x_t).
\label{eq:diffusion-guidance-local-expansion}
$$

这里关于时间变化的项与待采样变量 $x_{t-1}$ 无关，会被归一化常数吸收。若无条件反向核是

$$
p(x_{t-1}\mid x_t)
=\mathcal N(x_{t-1};\mu_t(x_t),\Sigma_t),
$$

把公式 \eqref{eq:diffusion-guidance-local-expansion} 代入并对高斯指数配方，就得到

$$
p(x_{t-1}\mid x_t,y)
\approx
\mathcal N\left(
x_{t-1};
\mu_t(x_t)+\Sigma_t\nabla_{x_t}\log p(y\mid x_t),
\Sigma_t
\right).
\label{eq:diffusion-guidance-classifier-kernel}
$$

各向同性协方差 $\Sigma_t=\sigma_t^2I$ 时，Classifier Guidance 的采样式为

$$
x_{t-1}
=\mu_t(x_t)
+\sigma_t^2\nabla_{x_t}\log p(y\mid x_t)
+\sigma_t z,
\qquad z\sim\mathcal N(0,I).
\label{eq:diffusion-guidance-classifier-sampling}
$$

梯度乘的是方差而不是标准差，因为它来自高斯自然参数的平移。若
$\Sigma_t$ 不是单位阵的倍数，正确修正是
$\Sigma_t\nabla\log p(y\mid x_t)$，不同方向会按反向核的不确定性缩放。

分类器必须理解带噪输入 $x_t$。直接把只见过干净样本的分类器用于高噪声状态，会产生分布外梯度；一种近似补救是先用
$\hat x_0(x_t)$ 去噪再分类，但此时反向传播还会穿过 $\hat x_0$，得到的已不是严格的
$\nabla_{x_t}\log p(y\mid x_t)$。

## Guidance Scale 控制条件能量

实际采样会加入强度 $\gamma$：

$$
x_{t-1}
=\mu_t(x_t)
+\gamma\Sigma_t\nabla_{x_t}\log p(y\mid x_t)
+\Sigma_t^{1/2}z.
\label{eq:diffusion-guidance-scaled-classifier}
$$

$\gamma>1$ 会更强地把样本推向条件模型的高分区域，通常提高条件一致性，同时压缩多样性。不能简单把它严格解释为从归一化分布

$$
\tilde p(y\mid x_t)
=\frac{p(y\mid x_t)^\gamma}{Z(x_t)}
$$

采样，因为

$$
\nabla_{x_t}\log\tilde p(y\mid x_t)
=\gamma\nabla_{x_t}\log p(y\mid x_t)
-\nabla_{x_t}\log Z(x_t),
$$

而 $\gamma\ne1$ 时 $Z(x_t)=\sum_y p(y\mid x_t)^\gamma$ 一般依赖
$x_t$。忽略第二项后，缩放梯度更准确的理解是人为调节条件能量，而不是某个已归一化分类分布的精确 score。

这个角度也允许把类别概率推广为任意可微相似度 $s(x_t,y)$。直接定义能量重加权

$$
p_\gamma(x_{t-1}\mid x_t,y)
\propto
p(x_{t-1}\mid x_t)
\exp\left[\gamma s(x_{t-1},y)\right],
$$

再对 $s(x_{t-1},y)$ 局部展开，可得

$$
p_\gamma(x_{t-1}\mid x_t,y)
\approx
\mathcal N\left(
x_{t-1};
\mu_t(x_t)+\gamma\Sigma_t\nabla_{x_t}s(x_t,y),
\Sigma_t
\right).
\label{eq:diffusion-guidance-similarity-kernel}
$$

$y$ 因而可以是类别、文本或参考图像；$s$ 可以来自共享 embedding 的余弦相似度，也可以是适合具体任务的可微度量。关键条件仍然是度量网络要能处理当前噪声水平。

## 条件 score 统一随机与确定性采样

离散配方公式 \eqref{eq:diffusion-guidance-classifier-kernel} 的修正含
$\Sigma_t$，于是当确定性 DDIM 取 $\Sigma_t=0$ 时，看起来 Guidance 会消失。原因是这个公式从一个带方差的局部高斯核推导，不是确定性概率流的统一表达。

在 score 框架中，条件贝叶斯直接给出

$$
\nabla_x\log p_t(x\mid y)
=\nabla_x\log p_t(x)
+\nabla_x\log p_t(y\mid x).
\label{eq:diffusion-guidance-conditional-score}
$$

无论反向过程选择随机 SDE 还是 probability flow ODE，只需把无条件 score
$s_\theta(x,t)$ 换成公式 \eqref{eq:diffusion-guidance-conditional-score}。若采用

$$
s_\theta(x_t,t)
=-\frac{\varepsilon_\theta(x_t,t)}{\bar\beta_t},
$$

那么等价的噪声预测替换是

$$
\varepsilon_\theta^{\mathrm{guided}}(x_t,t,y)
=\varepsilon_\theta(x_t,t)
-\gamma\bar\beta_t
\nabla_{x_t}\log p_t(y\mid x_t).
\label{eq:diffusion-guidance-noise-replacement}
$$

公式 \eqref{eq:diffusion-guidance-noise-replacement} 不依赖采样方差是否为零，所以也适用于确定性 DDIM/ODE。离散均值平移和条件 score 替换不是两套无关技巧：在小步长、高斯反向核下，前者正是后者的一阶离散结果。

## Classifier-Free Guidance 如何得到两次预测

Classifier-Free 模型直接训练条件噪声预测器：

$$
L_{\mathrm{cond}}
=\mathbb E_{x_0,y,t,\varepsilon}
\left[
\left\|
\varepsilon-
\varepsilon_\theta(
\bar\alpha_t x_0+\bar\beta_t\varepsilon,
y,t
)
\right\|^2
\right].
\label{eq:diffusion-guidance-conditional-training}
$$

训练时以一定概率把 $y$ 替换为空条件 $\varnothing$。同一网络于是同时学习

$$
\varepsilon_\theta(x_t,y,t)
\quad\text{和}\quad
\varepsilon_\theta(x_t,\varnothing,t).
$$

条件 score 与无条件 score 的差满足

$$
\begin{aligned}
\nabla_x\log p_t(x\mid y)-\nabla_x\log p_t(x)
&=\nabla_x\log p_t(y\mid x)\\
&=-\frac{
\varepsilon_\theta(x_t,y,t)
-\varepsilon_\theta(x_t,\varnothing,t)
}{\bar\beta_t}.
\end{aligned}
\label{eq:diffusion-guidance-score-difference}
$$

因此不需要显式分类器，也可以用两次网络预测构造引导：

$$
\boxed{
\tilde\varepsilon_\theta(x_t,y,t)
=(1+w)\varepsilon_\theta(x_t,y,t)
-w\varepsilon_\theta(x_t,\varnothing,t)
}
\label{eq:diffusion-guidance-cfg}
$$

其中 $w=0$ 是普通条件预测，$w>0$ 把预测沿“无条件 $\to$ 条件”的方向外推。若把
$\gamma=1+w$，也可写成
$\varepsilon_{\mathrm{uncond}}+\gamma
(\varepsilon_{\mathrm{cond}}-\varepsilon_{\mathrm{uncond}})$。

这种线性外推与 Classifier Guidance 一样会以多样性换条件一致性。它并不保证对应某个归一化概率分布的精确 score，尤其当 scale 很大、网络预测又有误差时，外推结果可能离开训练时见过的 score 范围。空条件训练比例与 guidance scale 因此共同决定无条件基线是否可靠以及外推能走多远。

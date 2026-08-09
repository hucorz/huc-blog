---
title: SiD：用得分恒等式蒸馏单步生成器
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - Distillation
  - Score Matching
description: 从学生分布的得分匹配推导 SiD 损失，并解释恒等变换与权重选择。
---

> [《生成扩散模型漫谈（二十五）：基于恒等式的蒸馏（上）》](https://spaces.ac.cn/archives/10085) —— 苏剑林  
> [《生成扩散模型漫谈（二十六）：基于恒等式的蒸馏（下）》](https://spaces.ac.cn/archives/10567) —— 苏剑林

## 不采教师样本，改为匹配得分

设教师扩散模型的噪声预测器为 $\epsilon_{\varphi^*}$，单步学生生成器为 $x_0^{(g)}=g_\theta(z)$，$z\sim\mathcal N(0,I)$。对学生样本加噪：

$$
x_t^{(g)}=\bar\alpha_t g_\theta(z)+\bar\beta_t\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I).
\label{eq:sid-student-noise}
$$

教师预测器在真实数据分布上训练，不能直接给出学生加噪分布 $p_\theta(x_t^{(g)})$ 的得分。因此再训练一个学生 Denoiser $\epsilon_\psi$：

$$
\mathcal L_D(\psi)
=\mathbb E_{z,\varepsilon,t}
\left\|\epsilon_\psi(x_t^{(g)},t)-\varepsilon\right\|^2.
\label{eq:sid-denoiser-loss}
$$

若对当前 $\theta$ 把它优化到最优，则

$$
\epsilon_{\psi^*}(x,t)
=-\bar\beta_t\nabla_x\log p_\theta(x).
$$

同理教师满足 $\epsilon_{\varphi^*}(x,t)=-\bar\beta_t\nabla_x\log p_{\text{data},t}(x)$。让两者在学生样本上接近，就在匹配两种加噪分布的得分：

$$
\mathcal L_1(\theta)
=\mathbb E_{z,\varepsilon,t}
\left\|
\epsilon_{\varphi^*}(x_t^{(g)},t)
-\epsilon_{\psi^*}(x_t^{(g)},t)
\right\|^2.
\label{eq:sid-naive-generator-loss}
$$

该方案不需要教师逐步生成训练对，也不需要教师的原始训练集；但与 GAN 一样，实践只能交替更新 $\psi$ 和 $\theta$，$\psi$ 未必达到最优，并且通常会错误地把随 $\theta$ 变化的 $\psi^*(\theta)$ 当作常数。

## 得分恒等式从哪里来

对任意只通过 $x_t$ 依赖样本的向量函数 $f(x_t,t)$，条件得分匹配给出

$$
\mathbb E\left[
\left\langle f(x_t,t),\epsilon_{\psi^*}(x_t,t)\right\rangle
\right]
=\mathbb E\left[
\left\langle f(x_t,t),\varepsilon\right\rangle
\right].
\label{eq:sid-score-identity}
$$

原因是最优噪声预测器等于条件期望 $\mathbb E[\varepsilon\mid x_t]$。按全期望公式，

$$
\mathbb E\langle f(x_t),\mathbb E[\varepsilon\mid x_t]\rangle
=\mathbb E\langle f(x_t),\varepsilon\rangle.
$$

恒等式只在整体期望下成立，并不表示对每个样本都有 $\epsilon_{\psi^*}(x_t,t)=\varepsilon$。此外，$f$ 必须是 $x_t$ 的函数；若先求导后某一因子不再满足这个条件，便不能机械替换。

## 恒等变换削弱对在线 Denoiser 的依赖

展开公式 \eqref{eq:sid-naive-generator-loss}：

$$
\|\epsilon_{\varphi^*}-\epsilon_{\psi^*}\|^2
=\langle
\epsilon_{\varphi^*}-\epsilon_{\psi^*},
\epsilon_{\varphi^*}-\epsilon_{\psi^*}
\rangle.
$$

选择其中一个满足条件的 $\epsilon_{\psi^*}$，用公式 \eqref{eq:sid-score-identity} 替换成已知采样噪声 $\varepsilon$，可构造改进目标

$$
\mathcal L_2(\theta)
=\mathbb E
\left\langle
\epsilon_{\varphi^*}(x_t^{(g)},t)-\epsilon_{\psi}(x_t^{(g)},t),
\epsilon_{\varphi^*}(x_t^{(g)},t)-\varepsilon
\right\rangle.
\label{eq:sid-transformed-loss}
$$

在 $\psi=\psi^*$ 时，它与原目标共享正确期望；实际交替训练中，它只出现一次 $\epsilon_\psi$，因而对在线 Denoiser 的近似误差更不敏感。这是稳定性的来源，不是说消除了双层优化。

SiD 实际使用

$$
\mathcal L_{\text{SiD}}
=\mathcal L_2-\lambda\mathcal L_1.
\label{eq:sid-combined-loss}
$$

负的 $\mathcal L_1$ 不是通常意义上的“再做一次得分匹配”，而是调整生成器梯度中不同路径的系数。

## 先变换再求导与先求导再变换

下篇的关键是：公式 \eqref{eq:sid-score-identity} 在函数值层面成立，但求 $\theta$ 梯度时，$\epsilon_{\psi^*}$ 同时通过输入 $x_t^{(g)}$ 和最优参数 $\psi^*(\theta)$ 依赖 $\theta$。将全导数拆开：

$$
\nabla_\theta\epsilon_{\psi^*(\theta)}(x_t^{(g)},t)
=\underbrace{J_x\epsilon_{\psi^*}\,\nabla_\theta x_t^{(g)}}_{\text{输入路径}}
+\underbrace{J_\psi\epsilon_{\psi^*}\,\nabla_\theta\psi^*}_{\text{最优响应路径}}.
\label{eq:sid-total-gradient}
$$

交替训练通常保留第一项、忽略第二项。先做恒等变换再求导，与先对平方范数求导后再对合法因子应用恒等式，会产生不同的系数。FGM 从分布生成流的目标梯度出发，得到的组合对应 $\lambda=1/2$；它抵消的是由于 stop-gradient 近似带来的特定重复项。

另一种把目标看成“教师与学生得分差对生成样本的推动方向”的推导，可得到 $\lambda=1$ 的解释。两者并不说明所有 $\lambda$ 等价，而是说明不同代理目标在理想条件下对应不同的梯度分解。实际 $\epsilon_\psi$ 未达最优时，仍需把 $\lambda$ 当作稳定性和偏差之间的超参数。

## 训练时真正维护的三套模型

教师 $\epsilon_{\varphi^*}$ 固定；学生 Denoiser $\epsilon_\psi$ 用公式 \eqref{eq:sid-denoiser-loss} 拟合当前学生分布；生成器 $g_\theta$ 用公式 \eqref{eq:sid-combined-loss} 更新。每次生成器更新时应冻结两套 Denoiser 参数，但保留从它们的输出经 $x_t^{(g)}$ 回到生成器的输入梯度。

三套同量级网络带来较高显存占用。用 LoRA 表示额外 Denoiser 可以节省参数，但不能改变方法对在线估计学生得分的依赖。SiD 的收益是省掉教师多步采样，代价则是交替优化和更复杂的梯度路径。

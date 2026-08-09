---
title: Skip Tuning：为少步采样补回非线性
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - Sampling
description: 从 U-Net 跳连的恒等偏置解释 Skip Tuning 如何改善少步扩散采样。
---

> [《生成扩散模型漫谈（二十四）：少走捷径，更快到达》](https://spaces.ac.cn/archives/10077) —— 苏剑林

## 少步采样损失了什么

扩散采样把 $x_T$ 到 $x_0$ 表示为多步非线性映射的复合。减少步数后，每一步必须覆盖更长轨迹，数值 Solver 或蒸馏模型能表达的组合非线性下降。ReFlow 一类方法试图把轨迹拉直，让所需映射本身更简单；Skip Tuning 从模型结构的另一端入手，调整 U-Net 的恒等偏置。

U-Net 第 $i$ 个解码块可以抽象为

$$
h_i^{\text{out}}
=F_i(h_i^{\text{in}},s_i)+\rho_i s_i,
\label{eq:skip-block}
$$

其中 $s_i$ 是编码器同尺度特征，$F_i$ 是主分支，$\rho_i$ 是 Skip Connection 的权重。训练时通常有 $\rho_i=1$。强跳连让信息和梯度容易传播，也鼓励网络学习对输入的小修正；极端情况下主分支很弱，整个网络接近恒等或低复杂度映射。

## 只在推理时缩小跳连

Skip Tuning 不重新训练模型，只在采样时令深层跳连权重小于训练值。若共有 $k+1$ 组跳连，可在靠近输入端的 $\rho_{\text{top}}$ 与瓶颈端的 $\rho_{\text{bottom}}$ 之间按深度插值：

$$
\rho_i
=\rho_{\text{top}}
+\frac{i}{k}(\rho_{\text{bottom}}-\rho_{\text{top}}),
\qquad i=0,\ldots,k.
\label{eq:skip-schedule}
$$

常固定 $\rho_{\text{top}}=1$，只调 $\rho_{\text{bottom}}$。深层特征分辨率低、语义更全局，减弱这些跳连会迫使解码主分支参与更多非线性变换；浅层跳连保持不变，则尽量保留边缘和纹理传递。

这不是严格地“恢复 Solver 丢失的那一项截断误差”。公式 \eqref{eq:skip-block} 只说明缩小 $\rho_i$ 改变了主分支与捷径的相对占比；少步采样误差与该变化是否匹配，需要实验验证。它更适合作为已有少步 Solver 的插件，而非独立的采样理论。

## 为什么不能无限减小

模型是在 $\rho_i=1$ 下训练的，推理时修改权重会产生分布偏移。$\rho_i$ 太小可能破坏已学到的特征尺度、条件控制和细节重建。最优值会随模型架构、参数化、CFG 强度、采样步数和 Solver 改变，不能跨配置照搬。

因此调参时应固定总 NFE、采样器和 guidance，仅扫描 $\rho_{\text{bottom}}$；还要同时检查 FID 一类分布指标与条件一致性、细节和多样性。Skip Tuning 的价值是几乎零训练成本，但它用一个推理超参数换取改进，并不保证对所有模型单调有效。

---
title: 生成扩散模型漫谈 31：JiT：低秩模型为什么应预测数据
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: true
tags:
  - AIGC
  - Diffusion
  - Manifold
  - Image Generation
description: 从噪声的全空间支撑与图像的低维流形解释数据预测在像素扩散中的优势。
---

> [《生成扩散模型漫谈（三十一）：预测数据而非噪声》](https://spaces.ac.cn/archives/11428) —— 苏剑林

## 大 Patch 暴露了低秩瓶颈

像素空间高分辨率扩散若想保持近似固定计算量，可以增大 ViT 的 Patch Size。以 $32\times32$ RGB Patch 为例，原始输入维度为

$$
d_{\text{patch}}=32\times32\times3=3072.
$$

若先线性投影到 768 维，再经过 hidden size 为 768 的 Transformer，单个 Patch 必经秩至多 768 的瓶颈。它不可能在整个 $\mathbb R^{3072}$ 上实现恒等映射。

对语义图像，这种压缩未必致命，因为自然图像 Patch 具有强结构；对独立高斯噪声，它会直接丢掉不可预测的方向。问题因此不只在“模型容量小”，还在模型被要求预测什么分布上的向量。

## 噪声目标与数据目标的有效维度不同

ReFlow 路径写成

$$
x_t=(1-t)x_0+t x_1,
\label{eq:jit-flow-path}
$$

其中 $x_0$ 是全维高斯噪声，$x_1$ 是真实数据。速度预测的标准目标为

$$
\mathcal L_v
=\mathbb E\left\|
v_\theta(x_t,t)-(x_1-x_0)
\right\|^2.
\label{eq:jit-v-loss}
$$

$x_1-x_0$ 仍含有全维噪声；DDPM 的 $\varepsilon$-prediction 更是直接回归全维高斯。若模型内部存在低秩瓶颈，它连把输入噪声相关分量原样传到输出都做不到。

自然图像则通常集中在环境空间中的低维流形附近。若数据流形的局部有效维度远小于 Patch 原始维度，低秩表征仍可能保留预测 $x_1$ 所需的主要坐标。因此 JiT 让神经网络直接输出数据估计

$$
\hat x_{1,\theta}=D_\theta(x_t,t),
\label{eq:jit-x-pred}
$$

再由代数关系

$$
x_1-x_0=\frac{x_1-x_t}{1-t}
$$

把它转换成速度：

$$
v_\theta(x_t,t)
=\frac{D_\theta(x_t,t)-x_t}{1-t}.
\label{eq:jit-x-to-v}
$$

公式 \eqref{eq:jit-x-to-v} 说明改变的是网络原生预测对象，不是改变采样 ODE 的目标分布。

## Prediction 与 Loss 是两层选择

模型输出可以参数化为 $x$、$\varepsilon$ 或 $v$；损失也可以在转换后对 $x$、$\varepsilon$ 或 $v$ 计算。两者不是同一选择。例如用公式 \eqref{eq:jit-x-pred} 做 $x$-prediction，却把它经公式 \eqref{eq:jit-x-to-v} 转成速度后计算 $v$-loss。

代数上这些参数化可互换，但转换包含随 $t$ 变化的系数，所以损失会对时间步重新加权；有限容量网络的优化也不具有参数化不变性。实验中，有低秩瓶颈时，决定能否成功训练的主要因素是神经网络原生预测数据，具体回归哪种等价 loss 的影响相对次要；没有瓶颈时，九种组合的差距明显缩小。

## 低秩不一定只是妥协

数据预测只需覆盖数据流形附近的方向，适度低秩会像结构先验或正则化：压掉与图像流形无关的自由度，反而可能改善 FID。这里不能推出“维度越低越好”；若瓶颈低于数据流形的有效维度，语义与细节仍会不可逆丢失。

这也解释了为什么同等计算量的大 Patch 模型可以在不同分辨率上取得接近的降采样指标：模型优先学习低维语义结构，高分辨率新增的像素自由度不必全部通过主干网络。至于真正的高频细节质量，仍不能只靠 FID/IS 判断。

## 与 SNR 问题的边界

高分辨率像素扩散还有另一个困难：下采样会降低独立噪声方差，使同一 Noise Schedule 的有效 SNR 变高。Schedule 对齐解决的是训练样本难度；数据预测解决的是网络输出经过低秩瓶颈时的可表示性。两者互补但不能互相替代。

JiT 的结论尤其适用于大 Patch、窄 hidden size 或其他明显压缩结构。若网络没有相关瓶颈、数据也不接近低维流形，$x$-prediction 的优势可能减弱；“数据比噪声容易预测”是依赖分布与容量的建模判断，不是无条件定理。

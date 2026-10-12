---
title: 生成扩散模型漫谈 22-23：高分辨率扩散：信噪比对齐与免训练放大
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: true
tags:
  - AIGC
  - Diffusion
  - Signal-to-Noise Ratio
  - Image Generation
description: 从下采样后的噪声方差推导高分辨率扩散的训练与免训练放大方法。
---

> [《生成扩散模型漫谈（二十二）：信噪比与大图生成（上）》](https://spaces.ac.cn/archives/10047) —— 苏剑林  
> [《生成扩散模型漫谈（二十三）：信噪比与大图生成（下）》](https://spaces.ac.cn/archives/10055) —— 苏剑林

## 同一 Noise Schedule 为什么随分辨率变简单

设加噪图像为

$$
x_t=\bar\alpha_t x_0+\bar\beta_t\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I).
\label{eq:hires-forward}
$$

把 $s\times s$ 个像素做平均池化。若噪声像素独立且方差为 $1$，平均后的噪声方差变成 $1/s^2$，标准差变成 $1/s$；局部平滑的图像信号却近似保持。因此下采样后的式子是

$$
\mathcal D_s[x_t]
\approx \bar\alpha_t x_0^{\text{low}}
+\frac{\bar\beta_t}{s}\varepsilon^{\text{low}},
\qquad \varepsilon^{\text{low}}\sim\mathcal N(0,I).
\label{eq:hires-downsample}
$$

其信噪比为

$$
\operatorname{SNR}_{\text{down}}(t)
=\frac{s^2\bar\alpha_t^2}{\bar\beta_t^2},
$$

是低分辨率模型在同一 $t$ 下 SNR 的 $s^2$ 倍。这里依赖“噪声独立、池化取平均、信号在池化窗口内较平滑”；若噪声有空间相关性，方差不会严格按 $1/s^2$ 缩小。

高分辨率训练若照搬低分辨率 Noise Schedule，Denoiser 实际看到的样本偏简单，难以充分学习困难噪声区间。解决方向是增大噪声，令不同分辨率在可比较尺度上具有相近 SNR。

## 训练端：Schedule、架构和多尺度损失

Schedule 对齐只处理学习难度，像素空间大图还带来计算量。Simple Diffusion 的结构策略是尽早把高分辨率特征降采样，把新增层、通道和 Dropout 集中到最低分辨率处，再在输出前恢复大小；模型容量增加，但重计算不长期停留在大特征图上。

普通噪声预测损失为

$$
\mathcal L_{1\times1}
=\frac{1}{wh}\left\|
\varepsilon-\epsilon_\theta(x_t,t)
\right\|^2.
$$

再对目标与预测同时平均池化，得到尺度 $s$ 的辅助目标：

$$
\mathcal L_{s\times s}
=\frac{1}{(w/s)(h/s)}
\left\|
\mathcal D_s[\varepsilon]
-\mathcal D_s[\epsilon_\theta(x_t,t)]
\right\|^2.
\label{eq:hires-multiscale-loss}
$$

多个尺度的损失共同约束低频结构与高频细节。它不是简单重复像素 MSE：池化后再求平方会显式惩罚区域平均误差，保证高分辨率模型在低分辨率观察下也能正确去噪。

## 推理端：低分辨率模型的主项

现在换一个问题：模型已经只在低分辨率上训练，能否免训练生成大图？直接把 CNN 用到大尺寸输入，往往还能产生清晰纹理，却因为 SNR 和全局结构错配而失去语义；单纯把小图上采样则保留语义但缺少细节。这正好给出“低分辨率主项 + 高分辨率残差项”的分工。

对公式 \eqref{eq:hires-downsample} 定义尺度修正

$$
\rho_t=\sqrt{\bar\alpha_t^2+\frac{\bar\beta_t^2}{s^2}}.
$$

除以 $\rho_t$ 后，信号与噪声系数的平方和重新为 $1$。再找训练时间 $\tau$ 满足

$$
\frac{\bar\alpha_\tau^2}{\bar\beta_\tau^2}
=\frac{s^2\bar\alpha_t^2}{\bar\beta_t^2}.
\label{eq:hires-time-remap}
$$

于是适配低分辨率模型分布的主项是

$$
\epsilon_\theta^{\text{main}}(x_t,t)
=\mathcal U_s\!\left[
\epsilon_\theta\!\left(
\frac{\mathcal D_s[x_t]}{\rho_t},\tau
\right)
\right].
\label{eq:hires-main-denoiser}
$$

$\mathcal U_s$ 把低分辨率噪声预测升回目标尺寸。时间重映射 \eqref{eq:hires-time-remap} 对齐 SNR，$\rho_t$ 则对齐模型训练时的总体输入尺度，两者作用不同。

## 用高分辨率调用补回纹理

同一个卷积 Denoiser 直接在大图 $x_t$ 上运行得到 $\epsilon_\theta(x_t,t)$。它的全局预测不可靠，但局部纹理仍有价值。可先取其低频部分，再构造只保留高频的残差：

$$
\epsilon_\theta^{\text{detail}}(x_t,t)
=\epsilon_\theta(x_t,t)
-\mathcal U_s\mathcal D_s[\epsilon_\theta(x_t,t)].
$$

Upsample Guidance 把公式 \eqref{eq:hires-main-denoiser} 的全局主项与该细节项组合，再代入原 DDPM/DDIM 更新。主项保证低频语义和正确 SNR，细节项利用 CNN 的局部平移不变性补高频。这里是近似的频率分工，不代表两个分量统计独立，也不能保证任意倍率放大；倍率越大，低分辨率模型未见过的中长程结构越多，近似越弱。

## 两篇工作的共同闭环

训练大图与免训练放大看似一个改模型、一个改采样，核心变量却相同：平均池化使独立噪声标准差缩小 $s$ 倍，从而让可见 SNR 增加 $s^2$ 倍。训练时据此重设 Schedule 并增加多尺度约束；推理时据此重映射时间和归一化输入。前者让新模型适配分辨率，后者让旧模型回到它熟悉的分布。

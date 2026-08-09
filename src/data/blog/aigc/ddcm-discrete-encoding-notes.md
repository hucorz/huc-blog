---
title: DDCM：把 DDPM 变成离散编码器
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - Discrete Representation
  - DDPM
description: 从有限噪声码本和条件引导推导 DDCM 的离散编码与重建过程。
---

> [《生成扩散模型漫谈（二十九）：用DDPM来离散编码》](https://spaces.ac.cn/archives/10711) —— 苏剑林

## 连续噪声序列就是 DDPM 的隐变量

写出 DDPM 的随机反向一步：

$$
x_{t-1}=\mu_t(x_t)+\sigma_t\varepsilon_t,
\qquad \varepsilon_t\sim\mathcal N(0,I).
\label{eq:ddcm-ddpm-step}
$$

连同初始 $x_T$，一条生成轨迹共使用 $T+1$ 个高维高斯变量。DDCM 为每一步预采样独立码本

$$
\mathcal C_t=\{c_{t,1},\ldots,c_{t,K}\},
\qquad c_{t,k}\overset{\text{i.i.d.}}\sim\mathcal N(0,I),
$$

并把公式 \eqref{eq:ddcm-ddpm-step} 中的噪声限制为 $\varepsilon_t\in\mathcal C_t$。生成过程便成为离散序列到图像的映射：

$$
(i_T,\ldots,i_1)\in\{1,\ldots,K\}^{T}
\longmapsto x_0.
\label{eq:ddcm-code-map}
$$

每个时间步应使用独立码本，因为不同 $t$ 的噪声扮演不同局部修正角色；共享码本会显著降低可选方向的组合多样性，通常需要更大的 $K$ 才能补偿。

有限码本的经验前提是：从高斯分布抽取的一组固定方向已足以近似每一步所需随机性。它不表示离散分布严格等于高斯；$K$ 增大时，经验分布才逐渐逼近连续分布。

## 编码不能从 $x_0$ 直接倒推

给定目标图像 $x_0^*$，希望找到索引序列使解码结果接近它。直接由

$$
x_0^*=\mu_1(x_1)+\sigma_1\varepsilon_1
$$

倒推会同时遇到未知 $x_1$ 和未知码字，形成组合搜索。DDCM 改为从固定 $x_T$ 正向执行反向采样，并在每一步选择最能把当前预测推向 $x_0^*$ 的码字。

令 $\bar\mu_t(x_t)$ 表示模型由 $x_t$ 预测的干净数据。DDPM 的反向均值可写成

$$
\mu_t(x_t)
=a_t x_t+b_t\bar\mu_t(x_t),
$$

其中在常用记号下

$$
a_t=\frac{\alpha_t\bar\beta_{t-1}^2}{\bar\beta_t^2},
\qquad
b_t=\frac{\bar\alpha_{t-1}\beta_t^2}{\bar\beta_t^2}.
$$

目标残差为 $r_t=x_0^*-\bar\mu_t(x_t)$。原方法选择

$$
\varepsilon_t
=\mathop{\operatorname{argmax}}_{c\in\mathcal C_t}
\langle c,r_t\rangle,
\label{eq:ddcm-argmax-code}
$$

再用公式 \eqref{eq:ddcm-ddpm-step} 更新。内积最大意味着码字方向最能补偿当前干净图预测的残差。每一步记录 $c$ 的索引，最终得到天然有序的一维离散编码。

## 从条件后验理解选择规则

已知真实 $x_0^*$ 时，DDPM 的解析后验为

$$
q(x_{t-1}\mid x_t,x_0^*)
=\mathcal N\left(
a_t x_t+b_t x_0^*,
\sigma_t^2I
\right).
$$

把 $x_0^*=\bar\mu_t(x_t)+r_t$ 代入，其均值等于无条件均值再加 $b_t r_t$。所以条件采样希望噪声修正 $\sigma_t c$ 接近 $b_t r_t$：

$$
c_t^*
=\mathop{\operatorname{argmin}}_{c\in\mathcal C_t}
\left\|b_t r_t-\sigma_t c\right\|^2.
\label{eq:ddcm-nearest-guidance}
$$

若码字范数近似相等，展开平方后与公式 \eqref{eq:ddcm-argmax-code} 等价，因为只剩最大化 $\langle c,r_t\rangle$。这说明 argmax 规则是有限候选集上的条件引导近似，而不是任意的启发式最近邻。

## 为什么还需要随机化版本

纯 argmax 在 $K\to\infty$ 时会选择越来越极端地对齐残差的方向，不会退化回原 DDPM 随机采样。更一致的有限近似是在码本上按

$$
p_t(c\mid r_t)
\propto
\exp\left(
-\frac12\left\|c-\frac{b_t}{\sigma_t}r_t\right\|^2
\right),
\qquad c\in\mathcal C_t
\label{eq:ddcm-code-distribution}
$$

采样。残差为零时，它近似原标准高斯的有限经验采样；随着 $K$ 增大，也更有希望恢复连续条件高斯。argmax 是公式 \eqref{eq:ddcm-code-distribution} 的 MAP 选择，重建更确定，但牺牲了连续极限与多样性。

## 编码长度、码本大小与压缩率

若有效记录 $T'$ 个索引，每个索引需要 $\log_2K$ bit，总码长约为 $T'\log_2K$。减少采样步数同时缩短编码，却会提高压缩率并增加重建误差；因此普通扩散加速技巧不能无代价地移植到 DDCM。

相比保留二维网格的 VQ 编码，DDCM 的索引按扩散时间天然排列成一维序列，便于接入自回归模型。代价是编码也必须运行完整的多步扩散过程，速度与原 DDPM 同阶，并且码本质量、时间步数和随机/确定选择共同决定率失真权衡。

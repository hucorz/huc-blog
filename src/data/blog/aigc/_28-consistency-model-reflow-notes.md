---
title: 生成扩散模型漫谈 28：一致性模型：从 ReFlow 的逐点回归出发
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: true
tags:
  - AIGC
  - Diffusion
  - Consistency Model
  - Flow Matching
description: 将 ReFlow 的数据端预测改写为相邻时间一致性，梳理训练、采样与蒸馏。
---

> [《生成扩散模型漫谈（二十八）：分步理解一致性模型》](https://spaces.ac.cn/archives/10633) —— 苏剑林

## ReFlow 先改写成预测数据端

令 $x_0$ 是数据、$x_1$ 是噪声，使用从数据端到噪声端的直线路径

$$
x_t=(1-t)x_0+t x_1.
\label{eq:cm-linear-path}
$$

若速度模型 $v_\theta(x_t,t)$ 预测 $x_1-x_0$，则可以由当前点反推数据端：

$$
f_\theta(x_t,t)=x_t-t\,v_\theta(x_t,t).
\label{eq:cm-endpoint-map}
$$

这里数据在 $t=0$、噪声在 $t=1$，$f_\theta$ 预测的正是数据 $x_0$。ReFlow 的速度 MSE 等价于带权端点回归：

$$
\mathcal L_{\text{endpoint}}
=\mathbb E\left[
w(t)\|f_\theta(x_t,t)-x_0\|^2
\right],
$$

其中 $w(t)$ 吸收由公式 \eqref{eq:cm-endpoint-map} 带来的 $1/t^2$ 权重。直接回归同一个 $x_0$ 已暗含：同一条轨迹上所有时间点应输出相同端点。

## 从直接标签换成相邻时间标签

将 $[0,1]$ 离散成 $t_k=k/n$。最靠近边界的模型由真实 $x_0$ 锚定；随后不再让每个 $t_k$ 都直接回归 $x_0$，而是要求相邻点预测一致：

$$
\mathcal L_{\text{CT}}
=\mathbb E_k\left[
w(t_k)
\left\|
f_\theta(x_{t_k},t_k)
-f_{\bar\theta}(x_{t_{k-1}},t_{k-1})
\right\|^2
\right].
\label{eq:cm-consistency-loss}
$$

$\bar\theta$ 通常是 $\theta$ 的 EMA，并在目标分支停止梯度。相邻点 $x_{t_{k-1}}$ 可由已知训练对直接构造；在蒸馏场景中，也可用教师 ODE 从 $x_{t_k}$ 数值推进得到。

如果每个相邻差都足够小，望远镜求和给出

$$
f_\theta(x_{t_n},t_n)-x_0
=\sum_{k=1}^{n}
\left[f_\theta(x_{t_k},t_k)-f_\theta(x_{t_{k-1}},t_{k-1})\right].
\label{eq:cm-telescope}
$$

因此局部一致性控制整个轨迹端点误差。但范数只满足三角不等式，局部误差可能同向累积；一致性不是“每项小就自动零误差”的严格承诺，边界条件和离散密度很重要。

## 为什么它能一步生成

一致性训练直接让任意时间点映射到同一端点，所以噪声端可以一次调用：

$$
\hat x_0=f_\theta(x_1,1),
\qquad x_1\sim\mathcal N(0,I).
\label{eq:cm-one-step}
$$

这与对瞬时速度做一步欧拉法不同。欧拉法假设整个区间速度等于起点速度；公式 \eqref{eq:cm-one-step} 则直接学习区间端点映射，训练目标已经针对大跨度输出。

## 多步采样为什么要重新加噪

若一步预测仍有缺陷，可以把当前 $\hat x_0$ 重新加到较小时间 $t_k$：

$$
x_{t_k}=(1-t_k)\hat x_0+t_k z,
\qquad z\sim\mathcal N(0,I),
$$

再计算 $\hat x_0\leftarrow f_\theta(x_{t_k},t_k)$。重新加噪把不完美样本送回模型训练过的带噪分布，下一次一致性映射再做修正。它不是沿同一确定性 ODE 轨迹继续积分，因为每轮新采样的 $z$ 会改变联合轨迹；边缘噪声级别正确即可。

## 从零训练与蒸馏的差别

一致性训练使用真实数据与预设路径自行构造相邻点，不要求教师；一致性蒸馏则让已有扩散 ODE 给出相邻点，继承教师轨迹。两者共享公式 \eqref{eq:cm-consistency-loss} 的形式，却拥有不同监督来源。

EMA/stop-gradient 能稳定自举，但也使目标耦合优化动力学：即使存在零损失退化解，边界参数化和教师分支的缓慢更新仍需共同阻止坍缩。理解一致性模型时，不能把 EMA 当作纯代数记号。

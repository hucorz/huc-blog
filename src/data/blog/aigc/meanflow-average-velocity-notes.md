---
title: MeanFlow：直接学习区间平均速度
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - Flow Matching
  - Sampling
description: 从瞬时速度与区间平均速度的恒等关系推导 MeanFlow 的单步生成目标。
---

> [《生成扩散模型漫谈（三十）：从瞬时速度到平均速度》](https://spaces.ac.cn/archives/10958) —— 苏剑林

## 一步生成真正需要平均速度

对 ODE $dx_t/dt=v(x_t,t)$，从较晚时间 $t$ 走到较早时间 $r<t$ 的精确更新为

$$
x_r=x_t-(t-r)u(x_t,r,t),
\label{eq:meanflow-exact-update}
$$

其中沿真实轨迹的平均速度定义为

$$
u(x_t,r,t)
=\frac{1}{t-r}\int_r^t v(x_\tau,\tau)\,d\tau.
\label{eq:meanflow-average-definition}
$$

普通 Flow Matching 学的是 $v$，一步欧拉法却拿瞬时量代替整段平均量；MeanFlow 直接建模 $u_\theta(x_t,r,t)$，所以 $r=0,t=1$ 时一次调用即可生成。边界上有

$$
u(x_t,t,t)=v(x_t,t),
\label{eq:meanflow-boundary}
$$

这是平均速度退化到瞬时速度的极限，不需要另外定义两套网络。

## 对积分恒等式求导

由公式 \eqref{eq:meanflow-average-definition}，

$$
(t-r)u(x_t,r,t)=\int_r^t v(x_\tau,\tau)\,d\tau.
$$

对上端点 $t$ 求全导数。左边既显式依赖 $t$，也通过 $x_t$ 依赖 $t$：

$$
\frac{d}{dt}[(t-r)u]
=u+(t-r)\left[
J_xu\,\frac{dx_t}{dt}+\partial_tu
\right].
$$

右边由微积分基本定理得到 $v(x_t,t)$。再用 $dx_t/dt=v(x_t,t)=u(x_t,t,t)$，得到核心恒等式

$$
v(x_t,t)
=u(x_t,r,t)
+(t-r)\left[
J_xu(x_t,r,t)\,u(x_t,t,t)
+\partial_tu(x_t,r,t)
\right].
\label{eq:meanflow-core-identity}
$$

括号是 $u$ 沿 ODE 轨迹和时间上端点的方向导数，可用 JVP 计算，不必显式构造高维 Jacobian。

## 用 ReFlow 提供瞬时速度监督

在线性插值

$$
x_t=(1-t)x_0+t x_1
$$

下，条件瞬时速度目标是 $x_1-x_0$。把它代入公式 \eqref{eq:meanflow-core-identity}，可直接训练平均速度：

$$
\mathcal L_{\text{MF}}
=\mathbb E_{r,t,x_0,x_1}
\left\|
u_\theta(x_t,r,t)
+(t-r)\left[
J_xu_\theta\,u_\theta(x_t,t,t)
+\partial_tu_\theta
\right]
-(x_1-x_0)
\right\|^2.
\label{eq:meanflow-training-loss}
$$

它是单个显式最小化目标，不需要教师、对抗训练或 EMA。在函数容量充足并达到全局最优时，ReFlow 的边缘速度保证与公式 \eqref{eq:meanflow-core-identity} 一起约束出正确的平均速度。

实现时常把恒等式右侧构造为 stop-gradient target，以避免二阶导数并节省显存。这会让实际优化动力学与纯公式 \eqref{eq:meanflow-training-loss} 有差别；理论上“无 stop-gradient”的目标与工程实现不能混为一谈。

## 一步与多步使用同一模型

一步生成直接取

$$
x_0=x_1-u_\theta(x_1,0,1),
\qquad x_1\sim\mathcal N(0,I).
$$

若一步质量不足，可选网格 $1=t_N>\cdots>t_0=0$，逐段使用

$$
x_{t_n}=x_{t_{n+1}}
-(t_{n+1}-t_n)
u_\theta(x_{t_{n+1}},t_n,t_{n+1}).
$$

每段预测的本来就是该区间平均速度，所以这不是缩小步长来修补普通欧拉截断误差，而是换用更短、通常更容易拟合的区间条件。

## 与 AMED、Shortcut 和一致性模型的关系

AMED 试图找一个中间点瞬时速度来近似平均速度，仍属于定制 Solver，并需要少量蒸馏。Shortcut 将步长作为条件，用“两小步等于一大步”直接正则平均速度的组合性质。MeanFlow 则从定义求导得到公式 \eqref{eq:meanflow-core-identity}，把平均速度连接回可监督的瞬时 Flow。

一致性模型学习的是同一路径上不变的端点映射；MeanFlow 学的是任意区间位移除以区间长度。两者都能一步生成，但预测对象、边界条件和训练恒等式不同。

---
title: PFGM：沿 Poisson 场线生成数据
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - PFGM
  - Differential Equation
description: 从多源引力场、升维消除模式坍缩和场线 ODE 梳理 PFGM 的构造。
---

> [《生成扩散模型漫谈（十三）：从万有引力到扩散模型》](https://spaces.ac.cn/archives/9305) —— 苏剑林

## 把数据点看成引力源

在 $d$ 维欧氏空间中，位于 $y$ 的单位点源产生的场可以写成

$$
F(x)=-\frac{1}{S_d(1)}
\frac{x-y}{\|x-y\|^d},
$$

其中 $S_d(1)$ 是 $d$ 维单位超球面的表面积。它是 Poisson 方程格林函数的梯度；多个点源的场可以线性叠加。

若训练数据中的每个样本都是一个源，那么从远处出发沿场线运动，大多数轨迹会终止在某个样本附近。这就把生成变成一个确定性运输问题：远处的简单分布提供初值，数据产生的向量场负责把初值拉回数据分布。

远处的多源场近似等价于全部质量集中在质心的单源场，因此趋于各向同性。这个性质让先验容易设计：在足够远的球面上均匀采样方向即可。但同一个各向同性性质也带来更严重的问题。

## 原空间中的场会遗漏模式

均匀球壳在壳内产生的合场为零。对连续数据分布，即使整体并非球对称，也可能存在互相抵消的各向同性分量；这些源对外部场线没有贡献，相应的数据模式就无法由场线到达。这里的模式坍缩不是有限网络训练失败，而是原空间场构造本身允许源被抵消。

PFGM 的处理是增加一个维度。原数据 $x_0\in\mathbb R^d$ 被嵌入 $\mathbb R^{d+1}$ 的 $t=0$ 超平面：

$$
(x,t)\sim p_0(x)\delta(t).
$$

一个在 $d$ 维中各向同性的集合，嵌入 $d+1$ 维后只占一个超平面，不再是高维各向同性球壳，因此不能通过同样机制把场完全抵消。额外坐标还自然给出了生成终点：轨迹到达 $t=0$ 就停止，不必另行判断“是否撞到数据点”。

## 升维后的场线 ODE

对平面上的全部数据源积分，$d+1$ 维场为

$$
\begin{aligned}
F(x,t)
&=-\frac{1}{S_{d+1}(1)}
\int
\frac{(x-x_0,t)}
{(\|x-x_0\|^2+t^2)^{(d+1)/2}}
p_0(x_0)dx_0\\
&\triangleq (F_x(x,t),F_t(x,t)).
\end{aligned}
\label{eq:pfgm13-augmented-field}
$$

如果用辅助时间 $\tau$ 参数化场线，则 $(dx,dt)=F(x,t)d\tau$。只要 $F_t\neq0$，可以消去 $\tau$，直接用额外坐标 $t$ 作为积分时间：

$$
\boxed{
\frac{dx}{dt}=\frac{F_x(x,t)}{F_t(x,t)}
}
\label{eq:pfgm13-field-line-ode}
$$

公式 \eqref{eq:pfgm13-field-line-ode} 只依赖场的方向，所以给 $F$ 乘任意正标量不会改变轨迹。这也是训练时可以学习归一化方向而不必精确恢复场强的原因。生成从 $t=T$ 的平面出发，沿 ODE 反向积分到 $t=0$。

## $t=T$ 平面上的先验不是高斯

在充分远的 $d+1$ 维球面上，场线穿过球面的方向近似均匀。但生成希望把 $t$ 固定为 $T$，所以需要把球面上的概率通量投影到 $t=T$ 平面。投影的面积伸缩会再贡献一个距离因子，最终得到

$$
p_{\text{prior}}(x)
\propto
\frac{1}{(\|x\|^2+T^2)^{(d+1)/2}}.
\label{eq:pfgm13-plane-prior}
$$

它只依赖半径 $r=\|x\|$。把笛卡尔体积元换为超球坐标，径向密度是

$$
p_{\text{prior}}(r)
\propto
r^{d-1}(r^2+T^2)^{-(d+1)/2}.
$$

因此采样可以分成两步：先从这个一维径向分布采样 $r$，再在单位球面均匀采样方向 $u$，最后令 $x=ru$。$T$ 要足够大，使有限数据源的场接近质心点源的场；它不是理论上任意小都成立的普通 Noise Schedule 端点。

## 怎样用有限数据学习连续场

去掉与轨迹无关的常数，公式 \eqref{eq:pfgm13-augmented-field} 是单样本场贡献的期望：

$$
F(x,t)
=\mathbb E_{x_0\sim p_0}
\left[
-\frac{(x-x_0,t)}
{(\|x-x_0\|^2+t^2)^{(d+1)/2}}
\right].
$$

平方损失的条件均值性质允许网络 $s_\theta(x,t)$ 直接回归单样本贡献：

$$
\mathbb E
\left[
\left\|
s_\theta(x,t)
+\frac{(x-x_0,t)}
{(\|x-x_0\|^2+t^2)^{(d+1)/2}}
\right\|^2
\right].
\label{eq:pfgm13-unbiased-field-loss}
$$

这里还缺少训练点 $(x,t)$ 的分布。PFGM 没有先规定一条像 DDPM 那样的正向马尔可夫链，而是围绕数据点采样多尺度扰动：

$$
x=x_0+\|\varepsilon_x\|(1+\tau)^m u,
\qquad
t=|\varepsilon_t|(1+\tau)^m,
$$

其中 $(\varepsilon_x,\varepsilon_t)$ 来自各向同性高斯，$u$ 是 $d$ 维单位球面上的均匀方向，$m$ 在一个有限区间内均匀采样。共同的尺度 $(1+\tau)^m$ 让训练覆盖从数据平面附近到远场的多个数量级。它在功能上承担了前向采样的角色，但不是由生成 ODE 正向积分得到的真实轨迹分布。

原论文更接近先用一个 batch 估计期望场，再对估计结果归一化，最后学习场方向。要注意

$$
\operatorname{Normalize}(\mathbb E[Y])
\neq
\mathbb E[\operatorname{Normalize}(Y)],
$$

而有限 batch 下对样本均值再做非线性归一化，也不是总体归一化方向的无偏估计。归一化可能让不同尺度下的训练更稳定，但估计质量会更依赖 batch size。公式 \eqref{eq:pfgm13-unbiased-field-loss} 与论文实践目标不能当作严格相同的有限样本统计量。

训练后取网络输出的前 $d$ 个分量与最后一个分量之比，替代公式 \eqref{eq:pfgm13-field-line-ode} 中的 $F_x/F_t$，从公式 \eqref{eq:pfgm13-plane-prior} 的先验反向积分到 $t=0$。这条路线与高斯 Score ODE 的共同点是都学习时间相关向量场；区别是 PFGM 从 Poisson 场的几何边界条件出发，不依赖高斯扰动作为理论起点。

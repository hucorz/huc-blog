---
title: 从连续性方程直接构造扩散 ODE
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - Differential Equation
description: 从概率守恒推到热方程，并得到可训练、可逆向采样的确定性扩散流。
---

> [《生成扩散模型漫谈（十二）：“硬刚”扩散ODE》](https://spaces.ac.cn/archives/9280) —— 苏剑林

## ODE 怎样搬运一整个分布

考虑确定性动力系统

$$
\frac{dx_t}{dt}=f_t(x_t),
\qquad t\in[0,T].
\label{eq:ode12-dynamics}
$$

在解存在且唯一、流映射足够光滑的条件下，给定 $x_0$ 可以确定 $x_T$，也可以从 $x_T$ 反向积分回 $x_0$。生成问题因此变成：怎样选择速度场 $f_t$，使数据分布 $p_0$ 被推到容易采样的 $p_T$？

先把公式 \eqref{eq:ode12-dynamics} 离散一小步：

$$
x_{t+\Delta t}=x_t+f_t(x_t)\Delta t+o(\Delta t).
$$

确定性变量变换必须守恒概率质量：

$$
p_t(x_t)dx_t
=p_{t+\Delta t}(x_{t+\Delta t})
\left|\frac{\partial x_{t+\Delta t}}{\partial x_t}\right|dx_t.
$$

对应的雅可比矩阵为 $I+J_f\Delta t+o(\Delta t)$。利用
$\det(I+A\Delta t)=1+\operatorname{Tr}(A)\Delta t+o(\Delta t)$，可得

$$
\log p_{t+\Delta t}(x_{t+\Delta t})-
\log p_t(x_t)
=-\nabla\cdot f_t(x_t)\Delta t+o(\Delta t).
\label{eq:ode12-jacobian-density-change}
$$

这个式子从“体积怎样伸缩”描述密度变化。另一边，对 $\log p_t(x)$ 同时沿空间和时间做一阶泰勒展开：

$$
\begin{aligned}
&\log p_{t+\Delta t}(x_{t+\Delta t})-
\log p_t(x_t)\\
&=f_t(x_t)\cdot\nabla\log p_t(x_t)\Delta t
+\partial_t\log p_t(x_t)\Delta t
+o(\Delta t).
\end{aligned}
$$

把它与公式 \eqref{eq:ode12-jacobian-density-change} 对齐并乘以 $p_t$，得到连续性方程：

$$
\boxed{
\partial_t p_t(x)
=-\nabla\cdot\left(p_t(x)f_t(x)\right)
}
\label{eq:ode12-continuity-equation}
$$

它表达的是局部概率质量守恒：某区域密度的增加，只能来自概率流 $p_tf_t$ 的净流入。它也是 Fokker–Planck 方程在随机扩散项为零时的特例，但这里不需要先引入 SDE。

## 用 Score 指定速度方向

连续性方程只约束 $p_t$ 与 $f_t$ 的配合，并没有唯一指定速度场。为了得到可解的一族模型，令

$$
f_t(x)=-D_t(x)\nabla_x\log p_t(x).
\label{eq:ode12-score-velocity}
$$

若 $D_t(x)$ 是非负标量，从数据到噪声的正向时间里，速度指向密度下降方向；反向积分时方向翻转，样本会走向高密度区域。把公式 \eqref{eq:ode12-score-velocity} 代入连续性方程 \eqref{eq:ode12-continuity-equation}：

$$
\partial_t p_t(x)
=\nabla\cdot\left(D_t(x)\nabla p_t(x)\right).
\label{eq:ode12-diffusion-equation}
$$

当 $D_t(x)=D_t$ 只依赖时间且为标量时，进一步化成热传导方程

$$
\partial_t p_t(x)=D_t\nabla^2p_t(x).
\label{eq:ode12-heat-equation}
$$

这里要区分两种“扩散”：概率密度 $p_t$ 满足热方程，沿 ODE 运动的单个粒子仍是确定性的。分布会越来越平滑，不代表同一个初值会随机分叉。

## 热方程为什么对应高斯加噪

对空间变量做傅里叶变换，记 $\widehat p_t(\omega)$ 为 $p_t$ 的特征函数。因为 $\nabla^2$ 在频域对应乘以 $-\|\omega\|^2$，公式 \eqref{eq:ode12-heat-equation} 变成关于 $t$ 的 ODE：

$$
\partial_t\widehat p_t(\omega)
=-D_t\|\omega\|^2\widehat p_t(\omega).
$$

令

$$
\sigma_t^2=2\int_0^tD_sds,
\qquad \sigma_0=0,
$$

则

$$
\widehat p_t(\omega)
=\widehat p_0(\omega)
\exp\left(-\frac12\sigma_t^2\|\omega\|^2\right).
$$

频域乘积对应空间卷积，而第二个因子正是协方差为 $\sigma_t^2I$ 的高斯分布的特征函数，所以

$$
\boxed{
p_t(x_t)
=\int\mathcal N(x_t;x_0,\sigma_t^2I)p_0(x_0)dx_0
}
\label{eq:ode12-heat-kernel-solution}
$$

等价的采样表达是 $x_t=x_0+\sigma_t\varepsilon$，其中 $\varepsilon\sim\mathcal N(0,I)$。这是边缘分布的等价表达；正向训练时直接这样采样，不等于 ODE 粒子轨迹真的在每个时刻加入独立噪声。

由 $\sigma_t^2=2\int_0^tD_sds$ 可得 $D_t=\dot\sigma_t\sigma_t$，因此公式 \eqref{eq:ode12-score-velocity} 变成

$$
\frac{dx_t}{dt}
=-\dot\sigma_t\sigma_t\nabla_{x_t}\log p_t(x_t).
\label{eq:ode12-final-sampling-ode}
$$

## 端点与 Score 的两个缺口

要从 $x_T$ 逆向生成，首先要求终点容易采样。由公式 \eqref{eq:ode12-heat-kernel-solution}，

$$
x_T=x_0+\sigma_T\varepsilon.
$$

当 $\sigma_T$ 远大于数据的典型尺度时，$x_0$ 的贡献相对很小，故 $p_T$ 近似 $\mathcal N(0,\sigma_T^2I)$。因此 Noise Schedule 至少要满足 $\sigma_0=0$、单调光滑并且 $\sigma_T$ 足够大。若数据没有预先中心化，终点均值也会残留相应偏移；“近似纯高斯”依赖尺度假设，并非任意有限 $\sigma_T$ 下严格成立。

第二个缺口是公式 \eqref{eq:ode12-final-sampling-ode} 需要未知的边缘 Score $\nabla_{x_t}\log p_t(x_t)$。高斯条件核的 Score 却有解析式：

$$
\nabla_{x_t}\log p_t(x_t\mid x_0)
=-\frac{x_t-x_0}{\sigma_t^2}
=-\frac{\varepsilon}{\sigma_t}.
$$

因此用网络 $s_\theta(x_t,t)$ 做条件得分匹配：

$$
\mathbb E_{x_0,\varepsilon,t}
\left[
\left\|
s_\theta(x_0+\sigma_t\varepsilon,t)
+\frac{\varepsilon}{\sigma_t}
\right\|^2
\right].
\label{eq:ode12-conditional-score-loss}
$$

平方损失的最优解是条件目标在给定 $x_t$ 后的均值，而这个条件均值恰好等于边缘 Score。训练完成后，用 $s_\theta$ 替换公式 \eqref{eq:ode12-final-sampling-ode} 中的真实 Score，从近似高斯的 $x_T$ 出发反向求解 ODE，便得到数据样本。

这条推导链的关键不是“热方程又一次给出了高斯噪声”，而是说明了设计顺序：先由概率守恒约束 ODE，再选择一类速度场把连续性方程化成可解 PDE，最后才从 PDE 的解读出前向采样核、终点分布和训练目标。

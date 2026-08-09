---
title: 从格林函数到 Rectified Flow
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - Flow Matching
  - Differential Equation
description: 沿格林函数、条件轨迹和速度回归三条路径梳理扩散 ODE 的一般构造。
---

> [《生成扩散模型漫谈（十四）：构建ODE的一般步骤（上）》](https://spaces.ac.cn/archives/9370) —— 苏剑林
>
> [《生成扩散模型漫谈（十五）：构建ODE的一般步骤（中）》](https://spaces.ac.cn/archives/9379) —— 苏剑林
>
> [《生成扩散模型漫谈（十七）：构建ODE的一般步骤（下）》](https://spaces.ac.cn/archives/9497) —— 苏剑林

## 从连续性方程看构造自由度

ODE $dx_t/dt=f_t(x_t)$ 搬运的边缘密度满足

$$
\partial_t p_t(x)=-\nabla_x\cdot\left(p_t(x)f_t(x)\right).
\label{eq:odefactory-continuity}
$$

给定完整的分布路径 $p_t$ 后，公式 \eqref{eq:odefactory-continuity} 对 $d$ 维速度 $f_t$ 只有一个标量约束，因此解通常不唯一。理论上的自由度很大，实践上的问题却是：怎样把解写成可采样、可用有限数据回归、并且终点确实是简单分布的形式。

把密度与概率流合并成 $d+1$ 维向量

$$
u(t,x)=\left(p_t(x),p_t(x)f_t(x)\right),
$$

连续性方程就变成 $\nabla_{(t,x)}\cdot u=0$。这样可以先求一个从边界点源 $(0,x_0)$ 出发的无散场 $G(t,0;x,x_0)$，满足

$$
G_1(0,0;x,x_0)=\delta(x-x_0),
\qquad
\int G_1(t,0;x,x_0)dx=1.
$$

再对真实数据源叠加：

$$
u(t,x)
=\int G(t,0;x,x_0)p_0(x_0)dx_0.
\label{eq:odefactory-green-superposition}
$$

$G_1(t,0;x,x_0)$ 就是条件密度 $p_t(x\mid x_0)$，其余分量是 $p_t(x\mid x_0)f_t(x\mid x_0)$。公式 \eqref{eq:odefactory-green-superposition} 同时解释了“点源力场叠加”和“条件扩散核混合”为何是同一件事。

## 各向同性格林函数与 CDF 构造

若要求点源解在 $(t,x-x_0)$ 上各向同性，无散条件把解限制为 Poisson 型：

$$
G(t,0;x,x_0)
\propto
\frac{(t,x-x_0)}
{(t^2+\|x-x_0\|^2)^{(d+1)/2}}.
$$

因此

$$
p_t(x\mid x_0)
\propto
\frac{t}{(t^2+\|x-x_0\|^2)^{(d+1)/2}},
$$

正是 PFGM 对应的重尾条件核。令 $z=(x-x_0)/t$ 后，$z$ 的密度不再依赖 $t$ 与 $x_0$，于是前向采样可以写成 $x_t=x_0+tz$；$T$ 足够大时，$x_0$ 的影响消失，得到同型的简单先验。

各向同性方案并不限于这个核。若径向密度为 $\phi_t(r)$，其中 $r=\|x-x_0\|$，笛卡尔密度换到球坐标后要乘 $r^{d-1}$。与其先猜密度再做积分，不如直接构造径向 CDF $\psi_t(r)$，要求

- $\psi_t(0)=0$、$\psi_t(\infty)=1$；
- 对 $r$ 单调递增；
- 任意固定 $r>0$ 都有 $\lim_{t\to0^+}\psi_t(r)=1$，使条件分布收缩到 $x_0$。

于是可以通过求导得到密度与径向概率流：

$$
\phi_t(r)
=\frac{1}{r^{d-1}}\partial_r\psi_t(r),
\qquad
\varphi_t(r)
=-\frac{1}{r^d}\partial_t\big(\psi_t(r)+\lambda_t\big).
$$

$\lambda_t$ 是只依赖时间的自由项，通常可以取零。像 $\tanh(r/t)$、$1-e^{-r/t}$ 这样的光滑阶跃近似都能给出一族解。高斯扩散、Poisson 扩散在这个视角下都只是不同的径向 CDF；它们在连续数学上可经坐标变换关联，但有限网络、采样器和训练权重仍会造成实践差异。

这个格林函数方案能从守恒方程系统地产生模型，但终点分布往往要在构造后再检查。下一步改为先指定单点轨迹与终点，直接把端点条件写进设计。

## 先画条件轨迹，再求条件速度

固定一个数据点 $x_0$，选择一族从 $x_0$ 到任意终点 $x_T$ 的光滑且不交叉轨迹，用隐式方程表示为

$$
\varphi_t(x_t\mid x_0)=x_T.
\label{eq:odefactory-implicit-track}
$$

沿同一条轨迹，右侧 $x_T$ 是积分常数。对 $t$ 求全导数：

$$
\partial_x\varphi_t\frac{dx_t}{dt}
+\partial_t\varphi_t=0,
$$

如果 $\partial_x\varphi_t$ 可逆，条件速度为

$$
f_t(x_t\mid x_0)
=-(\partial_x\varphi_t)^{-1}\partial_t\varphi_t.
\label{eq:odefactory-conditional-velocity}
$$

这样构造的 ODE 轨迹必然经过指定 $x_0$，初值条件不再需要事后验证。再给定易采样的终点密度 $p_T(x_T)$，沿公式 \eqref{eq:odefactory-conditional-velocity} 的特征线求解连续性方程：

$$
p_t(x_t\mid x_0)
=p_T(x_T)
\exp\left(
\int_t^T\nabla\cdot f_s(x_s\mid x_0)ds
\right).
\label{eq:odefactory-characteristic-density}
$$

公式 \eqref{eq:odefactory-characteristic-density} 的指数项是沿轨迹累计的体积变化。把公式 \eqref{eq:odefactory-implicit-track} 解出的 $x_T$ 代入后，条件密度只依赖 $(t,x_t,x_0)$，于是初值、终值与归一化同时闭合。

## 从条件速度得到真实边缘速度

单点条件速度依赖未知数据 $x_0$，不能直接用于生成。把全部点源混合后，真实边缘速度是后验加权平均：

$$
\begin{aligned}
f_t(x_t)
&=\frac{\int f_t(x_t\mid x_0)
p_t(x_t\mid x_0)p_0(x_0)dx_0}{p_t(x_t)}\\
&=\mathbb E\left[
f_t(x_t\mid x_0)\mid x_t
\right].
\end{aligned}
\label{eq:odefactory-marginal-velocity}
$$

因此用平方损失回归可计算的条件速度：

$$
\mathcal L_{\text{CFM}}
=\mathbb E_{t,x_0,x_t}
\left[
\|v_\theta(x_t,t)-f_t(x_t\mid x_0)\|^2
\right].
\label{eq:odefactory-cfm-loss}
$$

给定 $(x_t,t)$ 时，平方损失的最优解是条件期望，正好等于公式 \eqref{eq:odefactory-marginal-velocity}。这就是 Conditional Flow Matching 的核心：训练标签来自人为条件路径，但网络学到的是所有条件路径混合后的边缘速度。

> 单点轨迹选直线，训练出的真实 ODE 轨迹也一定是直线吗？

不一定。网络在同一个 $x_t$ 处平均来自不同 $x_0$ 的条件速度，积分这个平均场通常会产生曲线。直线只描述训练用的条件耦合路径，不是每个生成样本最终严格遵循的边缘轨迹。

## 线性路径包含高斯，也允许非高斯先验

令 $T=1$，选择

$$
x_t=\mu_t(x_0)+\sigma_tx_1,
$$

其中 $\mu_0(x_0)=x_0$、$\mu_1(x_0)=0$，$\sigma_0=0$、$\sigma_1=1$。条件速度和条件密度分别为

$$
f_t(x_t\mid x_0)
=\dot\mu_t(x_0)
+\frac{\dot\sigma_t}{\sigma_t}
(x_t-\mu_t(x_0)),
$$

$$
p_t(x_t\mid x_0)
=\frac{1}{\sigma_t^d}
p_1\!\left(
\frac{x_t-\mu_t(x_0)}{\sigma_t}
\right).
$$

只要 $p_1$ 易于采样，就能先采样 $x_1\sim p_1$ 再构造 $x_t$，不要求先验是高斯。最简单的直线取 $\mu_t=(1-t)x_0$、$\sigma_t=t$：

$$
x_t=(1-t)x_0+tx_1,
\qquad
f_t(x_t\mid x_0)=x_1-x_0
=\frac{x_t-x_0}{t}.
\label{eq:odefactory-straight-conditional-path}
$$

若 $p_1=\mathcal N(0,I)$，则 $p_t(x_t\mid x_0)=\mathcal N((1-t)x_0,t^2I)$；若 $p_1$ 是均匀分布或其他简单分布，同一构造仍成立。

## Rectified Flow 省掉显式密度求解

Rectified Flow 再向前简化一步：既然已经能直接采样端点 $(x_0,x_1)$ 并计算任意插值 $x_t=\Phi_t(x_0,x_1)$，就直接回归插值速度 $\partial_t\Phi_t$：

$$
\boxed{
\mathcal L_{\text{RF}}
=\mathbb E_{t,x_0,x_1}
\left[
\left\|
v_\theta(\Phi_t(x_0,x_1),t)
-\partial_t\Phi_t(x_0,x_1)
\right\|^2
\right]
}
\label{eq:odefactory-reflow-objective}
$$

对直线路径，公式 \eqref{eq:odefactory-reflow-objective} 就是回归 $x_1-x_0$。它与公式 \eqref{eq:odefactory-cfm-loss} 相同，但构造时不必先显式写出条件密度或用特征线法求解。

严格性来自测试函数。伪轨迹的一小步满足

$$
x_{t+\Delta t}
=x_t+\partial_t\Phi_t(x_0,x_1)\Delta t+o(\Delta t).
$$

对任意光滑测试函数 $h$，先按给定 $x_t$ 条件化并做一阶展开，未观测端点只通过条件均值出现：

$$
\mathbb E[h(x_{t+\Delta t})]
=\mathbb E\left[
h\!\left(
x_t+\Delta t\,
\mathbb E[\partial_t\Phi_t\mid x_t]
\right)
\right]+o(\Delta t).
$$

所以具有相同边缘演化的真实 ODE 是

$$
\frac{dx_t}{dt}
=\mathbb E[\partial_t\Phi_t(x_0,x_1)\mid x_t].
\label{eq:odefactory-reflow-true-ode}
$$

而公式 \eqref{eq:odefactory-reflow-objective} 的最优解正是这个条件均值。这里保证的是各时刻边缘分布以及端点运输正确，不是模型复原了每一对训练端点的原始耦合轨迹。

三篇文章形成的是同一问题的逐步收敛：格林函数把 ODE 构造变成守恒场求解；特征线法允许先指定条件轨迹和先验，再推出条件密度；Rectified Flow 发现只需采样轨迹并回归速度，条件期望就会自动给出正确的边缘速度。工程上最终只剩“选端点耦合、选插值、回归速度、积分 ODE”，但端点条件和分布正确性的依据仍来自前两层结构。

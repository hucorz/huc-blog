---
title: 生成扩散模型漫谈 05-06：得分扩散的 SDE 与概率流 ODE
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: true
tags:
  - AIGC
  - Diffusion
  - Score Matching
  - Differential Equation
description: 从反向 SDE、得分匹配和 Fokker–Planck 方程推导概率流 ODE。
---

> [《生成扩散模型漫谈（五）：一般框架之SDE篇》](https://spaces.ac.cn/archives/9209) —— 苏剑林
>
> [《生成扩散模型漫谈（六）：一般框架之ODE篇》](https://spaces.ac.cn/archives/9228) —— 苏剑林

## 连续时间把过程与离散步数分开

离散 DDPM 需要事先指定 $T$，连续时间框架则先定义一个前向随机微分方程：

$$
dx=f_t(x)\,dt+g_t\,dw.
\label{eq:score-sde-forward}
$$

$f_t(x)$ 是 drift，决定确定性运动方向；$g_t$ 是 diffusion coefficient，$w$ 是标准 Wiener 过程。理解公式 \eqref{eq:score-sde-forward} 时，可以先看它的 Euler–Maruyama 离散形式：

$$
x_{t+\Delta t}-x_t
=f_t(x_t)\Delta t
+g_t\sqrt{\Delta t}\,\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I).
\label{eq:score-sde-forward-discrete}
$$

随机项必须是 $\mathcal O(\sqrt{\Delta t})$。若它也是 $\mathcal O(\Delta t)$，把长度固定的时间区间分成 $n$ 段后，独立噪声和的方差大约是
$n(1/n)^2=1/n$，极限会消失；取 $\sqrt{\Delta t}$ 时，方差约为
$n(1/\sqrt n)^2=1$，随机效应才能留在连续极限中。

因此 $T$ 不再是模型定义的一部分，而是数值求解精度的一部分。理论上分析同一条连续轨迹，实现时再选择时间网格和求解器。

## 从局部贝叶斯推到反向 SDE

公式 \eqref{eq:score-sde-forward-discrete} 对应的单步转移核是

$$
p(x_{t+\Delta t}\mid x_t)
=\mathcal N\left(
x_{t+\Delta t};
x_t+f_t(x_t)\Delta t,
g_t^2\Delta t\,I
\right).
\label{eq:score-sde-local-kernel}
$$

反向局部核由贝叶斯公式给出：

$$
p(x_t\mid x_{t+\Delta t})
=p(x_{t+\Delta t}\mid x_t)
\exp\left[
\log p_t(x_t)-\log p_{t+\Delta t}(x_{t+\Delta t})
\right]
\times C,
$$

其中 $C$ 与 $x_t$ 无关。由于高斯局部核只在
$x_{t+\Delta t}-x_t=\mathcal O(\sqrt{\Delta t})$ 时有显著密度，可以在 $(x_t,t)$ 附近展开：

$$
\begin{aligned}
\log p_{t+\Delta t}(x_{t+\Delta t})
\approx{}&\log p_t(x_t)
+(x_{t+\Delta t}-x_t)^\top\nabla_x\log p_t(x_t)\\
&+\Delta t\,\partial_t\log p_t(x_t).
\end{aligned}
\label{eq:score-sde-density-taylor}
$$

时间偏导不能省略，因为密度本身也随扩散时间变化。把公式 \eqref{eq:score-sde-density-taylor} 代回局部贝叶斯表达式并对 $x_t$ 配方，保留决定高斯均值和方差的最低阶项，反向一步的均值相对 $x_{t+\Delta t}$ 多出
$-[f_t-g_t^2\nabla_x\log p_t]\Delta t$。连续极限就是

$$
dx=\left[f_t(x)-g_t^2\nabla_x\log p_t(x)\right]dt
+g_t\,d\bar w,
\label{eq:score-sde-reverse}
$$

其中公式沿 $t:T\to0$ 积分，$dt$ 为负；$\bar w$ 表示反向时间的 Wiener 过程。前向与反向使用相同的瞬时扩散强度，额外出现的
$-g_t^2\nabla_x\log p_t(x)$ 把样本推向当时的高密度区域。

这也说明网络真正缺失的量不是“噪声”这个特定参数化，而是边缘分布的 score：

$$
s^*(x,t)=\nabla_x\log p_t(x).
$$

## 条件 score 为什么能学到边缘 score

边缘分布是数据分布经过前向核后的混合：

$$
p_t(x_t)=\int p(x_t\mid x_0)\tilde p(x_0)\,dx_0.
$$

对 $x_t$ 求梯度并除以 $p_t(x_t)$：

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t)
&=\frac{\int p(x_t\mid x_0)\tilde p(x_0)
\nabla_{x_t}\log p(x_t\mid x_0)\,dx_0}
{p_t(x_t)}\\
&=\mathbb E_{p(x_0\mid x_t)}
\left[\nabla_{x_t}\log p(x_t\mid x_0)\right].
\end{aligned}
\label{eq:score-sde-fisher-identity}
$$

也就是说，给定带噪样本 $x_t$，条件 score 的后验均值恰好等于难以直接计算的边缘 score。平方损失的最优回归函数是条件均值，因此训练

$$
L_{\mathrm{DSM}}
=\mathbb E_{x_0,t,x_t\sim p(x_t\mid x_0)}
\left[
\left\|s_\theta(x_t,t)
-\nabla_{x_t}\log p(x_t\mid x_0)
\right\|^2
\right]
\label{eq:score-sde-denoising-score-matching}
$$

就能得到 $s_\theta(x_t,t)\approx\nabla_{x_t}\log p_t(x_t)$。公式 \eqref{eq:score-sde-fisher-identity} 是这个结论的关键条件：训练标签是条件 score，模型输入却只有 $x_t,t$，平方回归自动对不可见的 $x_0$ 做后验平均。

若选择高斯扰动核

$$
x_t=\bar\alpha_t x_0+\bar\beta_t\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I),
$$

则条件 score 有解析式

$$
\nabla_{x_t}\log p(x_t\mid x_0)
=-\frac{x_t-\bar\alpha_t x_0}{\bar\beta_t^2}
=-\frac{\varepsilon}{\bar\beta_t}.
\label{eq:score-sde-gaussian-conditional-score}
$$

令 $s_\theta(x_t,t)=-\varepsilon_\theta(x_t,t)/\bar\beta_t$，公式 \eqref{eq:score-sde-denoising-score-matching} 就变成带权噪声预测：

$$
L_{\mathrm{DSM}}
=\mathbb E\left[
\frac{1}{\bar\beta_t^2}
\left\|\varepsilon-\varepsilon_\theta(x_t,t)\right\|^2
\right].
$$

去掉 $1/\bar\beta_t^2$ 后得到 DDPM 常用的 simple loss，但不同噪声强度的相对权重已经改变。

## 先指定扰动核，再反推线性 SDE

从公式 \eqref{eq:score-sde-forward} 出发，未必容易求出 $p(x_t\mid x_0)$。更适合训练的路线是先设计一个可直接采样、条件 score 可解析的高斯扰动核，再求与之匹配的 SDE。

设

$$
p(x_t\mid x_0)
=\mathcal N(x_t;\bar\alpha_t x_0,\bar\beta_t^2I),
$$

并寻找线性过程 $dx=f_t x\,dt+g_t\,dw$。短时间转移为

$$
x_{t+\Delta t}=(1+f_t\Delta t)x_t
+g_t\sqrt{\Delta t}\,\varepsilon_2.
$$

将 $x_t=\bar\alpha_t x_0+\bar\beta_t\varepsilon_1$ 代入，并分别匹配
$x_{t+\Delta t}$ 的条件均值与方差：

$$
\begin{aligned}
\bar\alpha_{t+\Delta t}
&=(1+f_t\Delta t)\bar\alpha_t,\\
\bar\beta_{t+\Delta t}^2
&=(1+f_t\Delta t)^2\bar\beta_t^2+g_t^2\Delta t.
\end{aligned}
$$

取 $\Delta t\to0$，得到

$$
f_t=\frac{d}{dt}\log\bar\alpha_t,
\qquad
g_t^2=\bar\alpha_t^2
\frac{d}{dt}\left(
\frac{\bar\beta_t^2}{\bar\alpha_t^2}
\right).
\label{eq:score-sde-kernel-to-linear-sde}
$$

当 $\bar\alpha_t\equiv1$ 时只有方差增长，对应 VE-SDE；当
$\bar\alpha_t^2+\bar\beta_t^2=1$ 时总体尺度保持稳定，对应 VP-SDE。Noise Schedule 不只是离散超参数，它完整决定了连续 drift 和 diffusion coefficient。

## Fokker–Planck 方程只追踪边缘密度

反向 SDE 仍然是随机路径。如果只关心每个时刻的边缘分布 $p_t(x)$，需要把路径方程转成密度演化方程。

用 Dirac 函数表示密度：

$$
p_t(x)=\mathbb E[\delta(x-x_t)].
$$

把公式 \eqref{eq:score-sde-forward-discrete} 代入
$\delta(x-x_{t+\Delta t})$，围绕 $x-x_t$ 做二阶展开。对噪声求期望后，一阶随机项因 $\mathbb E\varepsilon=0$ 消失，二阶项由
$\mathbb E[\varepsilon\varepsilon^\top]=I$ 留下。利用 Dirac 导数在期望中的转移关系，可以得到

$$
\partial_t p_t(x)
=-\nabla_x\cdot\left[f_t(x)p_t(x)\right]
+\frac12g_t^2\Delta_x p_t(x).
\label{eq:score-sde-fokker-planck}
$$

第一项描述 drift 搬运概率质量，第二项描述噪声导致的密度扩散。Fokker–Planck 方程把大量随机样本路径压缩成一个确定性的边缘密度演化规律。

## 同一组边缘分布对应一族随机过程

利用 $\nabla p_t=p_t\nabla\log p_t$，对任意满足
$0\le\sigma_t^2\le g_t^2$ 的函数，都可以把公式 \eqref{eq:score-sde-fokker-planck} 改写为

$$
\begin{aligned}
\partial_t p_t
={}&-\nabla\cdot\left[
\left(f_t-\frac12(g_t^2-\sigma_t^2)
\nabla\log p_t\right)p_t
\right]
+\frac12\sigma_t^2\Delta p_t.
\end{aligned}
\label{eq:score-sde-equivalent-fokker-planck}
$$

因此下面整族前向 SDE 都产生完全相同的边缘分布 $p_t$：

$$
dx=\left[
f_t(x)-\frac12(g_t^2-\sigma_t^2)
\nabla_x\log p_t(x)
\right]dt
+\sigma_t\,dw.
\label{eq:score-sde-equivalent-forward-family}
$$

对公式 \eqref{eq:score-sde-equivalent-forward-family} 应用反向 SDE 公式，drift 中原有的修正项再减去
$\sigma_t^2\nabla\log p_t$，得到反向族：

$$
dx=\left[
f_t(x)-\frac12(g_t^2+\sigma_t^2)
\nabla_x\log p_t(x)
\right]dt
+\sigma_t\,d\bar w.
\label{eq:score-sde-equivalent-reverse-family}
$$

这里的“等价”仅指同一时刻的边缘分布相同；不同 $\sigma_t$ 对应不同联合路径、不同样本相关性和不同数值性质。这正是连续时间版本的 DDIM 自由度。

## 概率流 ODE 与 DDIM

取 $\sigma_t=0$，公式 \eqref{eq:score-sde-equivalent-forward-family} 和
\eqref{eq:score-sde-equivalent-reverse-family} 都退化为同一个确定性 ODE：

$$
\frac{dx}{dt}
=f_t(x)-\frac12g_t^2\nabla_x\log p_t(x).
\label{eq:score-sde-probability-flow-ode}
$$

它叫 probability flow ODE。虽然单条轨迹不再注入噪声，但其边缘密度仍与原 SDE 完全一致。用 $s_\theta$ 代替真实 score 后，正向积分把数据确定性映射到先验，反向积分把先验确定性映射回数据。

这带来三层作用：

- 可以使用成熟的自适应、高阶 ODE 求解器减少离散误差；
- 确定性可逆轨迹提供稳定的 latent representation 与编辑路径；
- 通过 ODE 的瞬时变量变换公式，可以像 continuous normalizing flow 一样计算似然。

在线性高斯扰动核下，把公式 \eqref{eq:score-sde-kernel-to-linear-sde} 和
$s_\theta=-\varepsilon_\theta/\bar\beta_t$ 代入公式
\eqref{eq:score-sde-probability-flow-ode}，整理可得

$$
\frac{d}{dt}\left(\frac{x_t}{\bar\alpha_t}\right)
=\varepsilon_\theta(x_t,t)
\frac{d}{dt}\left(
\frac{\bar\beta_t}{\bar\alpha_t}
\right).
\label{eq:score-sde-ddim-ode-equivalence}
$$

这正是确定性 DDIM 离散更新的连续极限。因此关系不是“DDIM 恰好像一个 ODE”，而是：DDIM 在高斯线性扰动路径上离散化了 probability flow ODE；原始随机采样则离散化了 reverse-time SDE。二者共享同一个 score 网络和同一组边缘分布，但沿着不同路径把噪声运回数据。

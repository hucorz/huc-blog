---
title: Score Matching 如何控制 Wasserstein 距离
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - Score Matching
  - Optimal Transport
description: 从耦合轨迹的稳定性出发，理解扩散模型损失为何能上界生成分布的 Wasserstein 距离。
---

> [《生成扩散模型漫谈（十六）：W距离 ≤ 得分匹配》](https://spaces.ac.cn/archives/9467) —— 苏剑林

## 需要控制的是生成终点的分布误差

二阶 Wasserstein 距离定义为

$$
\mathcal W_2[p,q]
=\left(
\inf_{\gamma\in\Pi(p,q)}
\mathbb E_{(x,y)\sim\gamma}\|x-y\|^2
\right)^{1/2},
$$

其中 $\Pi(p,q)$ 是边缘分布分别为 $p,q$ 的全部联合分布。扩散模型用近似 Score 代替真实 Score 后，真正关心的不是每个时间步的局部回归误差本身，而是这些误差累计到 $t=0$ 后，生成分布 $q_0$ 与数据分布 $p_0$ 相差多远。

原论文给出的核心结论可写成

$$
\boxed{
\mathcal W_2[p_0,q_0]
\le
\int_0^T
g_t^2I_t
\left(
\mathbb E_{x_t\sim p_t}
\|\nabla\log p_t(x_t)-s_\theta(x_t,t)\|^2
\right)^{1/2}dt
+I_T\mathcal W_2[p_T,q_T]
}
\label{eq:wscore16-main-bound}
$$

$g_t$ 是前向 SDE 的扩散系数，$I_t$ 是由动力系统稳定性条件产生的非负放大因子。若真实反向过程和模型反向过程从相同先验出发，即 $p_T=q_T$，最后一项为零。于是所有时刻的 Score 误差经过 $g_t^2I_t$ 加权积分，控制了最终生成误差。

这不是说 Score Matching 数值上等于 Wasserstein 距离，也不是说任意网络、任意动力系统都无条件满足这个上界。它依赖速度场的正则性与单侧 Lipschitz 一类稳定性条件；原论文的一般 SDE 证明还使用了最优传输的性质。

## 先看两个确定性流的误差怎样累计

从同一个随机初值 $z$ 出发，考虑两条 ODE：

$$
\frac{dx_t}{dt}=f_t(x_t),
\qquad
\frac{dy_t}{dt}=h_t(y_t),
\qquad x_T=y_T=z.
$$

同一个 $z$ 诱导出 $(x_t,y_t)$ 的一个耦合，因此

$$
\mathcal W_2[p_t,q_t]
\le
\widetilde{\mathcal W}_2(t)
\triangleq
\left(\mathbb E_z\|x_t-y_t\|^2\right)^{1/2}.
\label{eq:wscore16-synchronous-coupling}
$$

右侧未必是最优耦合，所以只能作为上界。令 $\delta_t=x_t-y_t$，把速度差拆成同一点上的模型误差与状态偏移引起的误差：

$$
f_t(x_t)-h_t(y_t)
=\big(f_t(x_t)-h_t(x_t)\big)
+\big(h_t(x_t)-h_t(y_t)\big).
$$

假设 $h_t$ 满足单侧 Lipschitz 条件

$$
(x-y)\cdot(h_t(x)-h_t(y))
\le L_t\|x-y\|^2.
\label{eq:wscore16-one-sided-lipschitz}
$$

对 $\mathbb E\|\delta_t\|^2$ 求导，再分别使用公式 \eqref{eq:wscore16-one-sided-lipschitz} 和 Cauchy–Schwarz 不等式，可以得到

$$
-\frac{d}{dt}\widetilde{\mathcal W}_2(t)
\le
\left(
\mathbb E\|f_t(x_t)-h_t(x_t)\|^2
\right)^{1/2}
+L_t\widetilde{\mathcal W}_2(t).
\label{eq:wscore16-distance-differential-bound}
$$

负号来自这里按扩散生成方向从 $T$ 积分到 $0$。对公式 \eqref{eq:wscore16-distance-differential-bound} 使用积分因子或 Grönwall 不等式，就得到“终点距离不超过沿途速度误差的加权积分”。这已经揭示主结论的结构：局部速度回归越准、动力系统越稳定，最终分布越接近。

## 把速度误差换成 Score 误差

同一个前向扩散可以对应一族反向过程。其 probability flow ODE 的真实速度含有

$$
-\frac12g_t^2\nabla_x\log p_t(x),
$$

模型速度则以 $s_\theta(x,t)$ 替代真实 Score。两者在同一点的差为

$$
\frac12g_t^2
\left(s_\theta(x,t)-\nabla_x\log p_t(x)\right).
$$

把它代入公式 \eqref{eq:wscore16-distance-differential-bound}，便得到公式 \eqref{eq:wscore16-main-bound} 的 ODE 版本；系数中的 $1/2$ 可以随所采用的反向 SDE 或 probability flow ODE 约定吸收到 $I_t$。推广到随机反向 SDE 时，还需比较最优耦合下的分布演化，不能只靠固定共同初值的同步耦合完成全部证明。

原文特别指出一个未能用初等方法跨过的步骤。若直接比较两个分布各自的 Score，会遇到

$$
\mathbb E
\left[
(x_t-y_t)\cdot
(\nabla\log p_t(x_t)-\nabla\log q_t(y_t))
\right]\ge0
$$

是否成立的问题；对任意同步耦合它并不成立。原论文是在最优传输耦合下调用专门结论处理，不能通过强行假设所有分布的对数梯度都满足单侧 Lipschitz 来替代，因为这类假设本身对一般分布是假的。

## 训练使用的条件目标仍然能控制上界

边缘 Score 有恒等式

$$
\nabla_{x_t}\log p_t(x_t)
=\mathbb E\left[
\nabla_{x_t}\log p_t(x_t\mid x_0)
\mid x_t
\right].
\label{eq:wscore16-fisher-identity}
$$

因此由 Jensen 不等式，边缘 Score Matching 损失不超过条件得分匹配损失：

$$
\begin{aligned}
&\mathbb E_{x_t}
\|\nabla\log p_t(x_t)-s_\theta(x_t,t)\|^2\\
&\le
\mathbb E_{x_0,x_t}
\|\nabla\log p_t(x_t\mid x_0)-s_\theta(x_t,t)\|^2.
\end{aligned}
\label{eq:wscore16-conditional-upper-bound}
$$

所以实际可计算的条件目标依然是公式 \eqref{eq:wscore16-main-bound} 中局部误差的一层上界。更精确地说，两种损失之差与模型参数无关；这一点会在第十八篇的笔记中单独展开。

## 时间权重透露了 simple loss 的作用

在高斯扰动参数化中，Score Matching 常被改写为噪声预测：

$$
\frac{1}{\bar\beta_t^2}
\mathbb E
\left[
\|\epsilon_\theta(\bar\alpha_tx_0+\bar\beta_t\varepsilon,t)
-\varepsilon\|^2
\right].
$$

DDPM 的 simple loss 删除 $1/\bar\beta_t^2$，相当于给原 Score 损失额外乘上随 $t$ 增大的 $\bar\beta_t^2$。公式 \eqref{eq:wscore16-main-bound} 中的 $g_t^2I_t$ 也通常随噪声时刻变化，因此原文把 simple loss 理解为一种更接近 Wasserstein 终点误差上界的重加权：它降低了小噪声处原本过大的 $1/\bar\beta_t^2$ 权重。

这个解释是设计动机，不是对 simple loss 最优性的证明。真实上界含有平方根、连续时间积分和依赖动力系统的 $I_t$，而 simple loss 通常只是对离散时刻做均匀 MSE；二者不能直接画等号。

WGAN 直接围绕 $\mathcal W_1$ 的对偶形式训练判别器，Score-based diffusion 则用局部 Score 误差间接控制 $\mathcal W_2$。两者的网络结构和训练过程不同，但这个上界说明它们并非在完全无关的几何目标上工作。

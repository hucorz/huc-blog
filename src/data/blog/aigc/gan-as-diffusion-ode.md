---
title: 把 GAN 看成参数空间中的扩散 ODE
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - GAN
  - Diffusion
  - Optimal Transport
description: 从密度比梯度流与 Rectified Flow 两条路线解释 GAN 的交替训练和 WGAN-GP 形式。
---

> [《生成扩散模型漫谈（十九）：作为扩散ODE的GAN》](https://spaces.ac.cn/archives/9662) —— 苏剑林
>
> [《生成扩散模型漫谈（二十）：从ReFlow到WGAN-GP》](https://spaces.ac.cn/archives/9668) —— 苏剑林

## 渐变发生在生成器训练时间里

普通扩散模型显式维护 $p_t$ 并让样本沿许多时刻逐渐运动；GAN 的生成器 $g_\theta(z)$ 看起来一次就把噪声映射成样本。联系两者的关键是把优化进度 $\tau$ 当作另一种时间：

$$
x_\tau=g_{\theta_\tau}(z),
\qquad z\sim\mathcal N(0,I).
$$

随着交替训练更新 $\theta_0,\theta_1,\ldots$，生成分布 $q_\tau$ 也形成一条渐变路径。于是 GAN 可以理解为：先在样本空间确定 $q_\tau$ 应该怎样向真实分布 $p$ 移动，再让生成器参数更新去拟合这一步样本运动。

这里必须区分两个时间变量：扩散或 Flow 的路径时间 $t$ 描述一次样本运输，$\tau$ 描述生成器训练进度。后面“在 $t=0$ 前进一步”会变成一次 $\theta_\tau\to\theta_{\tau+1}$，二者不能混用。

## MonoFlow：判别器估计密度比方向

令密度比

$$
r_\tau(x)=\frac{p(x)}{q_\tau(x)}.
$$

最小化 $q_\tau$ 到 $p$ 的 KL 散度所对应的一条 Wasserstein 梯度流满足

$$
\partial_\tau q_\tau(x)
=-\nabla\cdot
\left(q_\tau(x)\nabla\log r_\tau(x)\right).
$$

由连续性方程，样本速度是

$$
\frac{dx}{d\tau}=\nabla_x\log r_\tau(x).
\label{eq:ganode19-density-ratio-flow}
$$

真实密度与生成密度都未知，但能分别采样。Vanilla GAN 的判别目标

$$
\max_D
\mathbb E_{x\sim p}\log\sigma(D(x))
+\mathbb E_{x\sim q_\tau}\log(1-\sigma(D(x)))
$$

在无限容量和精确最优化下有

$$
D^*(x)=\log\frac{p(x)}{q_\tau(x)}.
\label{eq:ganode19-optimal-discriminator}
$$

所以当前判别器给出了公式 \eqref{eq:ganode19-density-ratio-flow} 的速度势。它只估计当前 $q_\tau$ 下的密度比，不能一次得到未来所有时刻的 ODE；但可以先做一个小 Euler 步：

$$
x^+=x+\epsilon\nabla_xD(x),
\qquad x=g_{\theta_\tau}(z).
\label{eq:ganode19-sample-euler-step}
$$

## 把样本移动蒸馏回生成器参数

下一组生成器参数应让同一个潜变量 $z$ 产生公式 \eqref{eq:ganode19-sample-euler-step} 的目标：

$$
\mathcal L_{\text{transport}}(\theta)
=\mathbb E_z
\left[
\|g_\theta(z)-g_{\theta_\tau}(z)
-\epsilon\nabla_xD(g_{\theta_\tau}(z))\|^2
\right].
\label{eq:ganode19-generator-transport-loss}
$$

在 $\theta=\theta_\tau$ 处求梯度，前两项抵消，通过链式法则得到

$$
\nabla_\theta\mathcal L_{\text{transport}}
\propto
-\nabla_\theta D(g_\theta(z))
\quad\text{at }\theta=\theta_\tau.
$$

因此若判别器固定且生成器只更新一小步，公式 \eqref{eq:ganode19-generator-transport-loss} 与常用生成器损失

$$
\mathcal L_G(\theta)
=\mathbb E_z[-D(g_\theta(z))]
\label{eq:ganode19-standard-generator-loss}
$$

给出相同方向，差一个正比例系数。等价只在当前参数处的一阶梯度意义上成立，不表示两个 loss 在有限距离内具有相同函数值或相同最优解。这也解释了为什么固定判别器后不宜让生成器连续走很多步：离开 $\theta_\tau$ 后，公式 \eqref{eq:ganode19-standard-generator-loss} 不再忠实拟合最初的局部运输目标；若要多步拟合，应保留带旧生成器锚点的公式 \eqref{eq:ganode19-generator-transport-loss}。

MonoFlow 还允许把 $\log r$ 换成它的单调递增变换 $h(\log r)$。这会改变速度大小，且通常也会改变参数化后的轨迹，但上升方向仍由密度比排序决定。很多 f-GAN 判别器最优解都是密度比的单调函数，因此可以放进同一梯度流视角。

这条推导与 GAN 的实际交替训练相吻合：当前生成器采样，判别器估计当前密度比，样本前进一步，再把这一步拟合回生成器。传统“对任意生成分布先把判别器完全解到最优，再把结果代入某个散度”的论证，更接近嵌套优化，并不直接描述有限步交替训练。

## 从 Rectified Flow 重建同一个交替过程

第二条路线不从 KL 的 Wasserstein 梯度流出发。令当前生成器样本为

$$
x_0=g_{\theta_\tau}(z),
$$

真实样本为 $x_1\sim p$，用直线连接：

$$
x_t=(1-t)x_0+tx_1,
\qquad
\frac{dx_t}{dt}=x_1-x_0.
\label{eq:ganode20-straight-path}
$$

Rectified Flow 训练速度场

$$
\mathcal L_v(\varphi)
=\mathbb E_{x_0,x_1,t}
\left[
\frac12
\|v_\varphi(x_t,t)-(x_1-x_0)\|^2
\right].
\label{eq:ganode20-reflow-loss}
$$

训练完成后，$v_{\varphi^*}(x,0)$ 给出当前生成分布向真实分布运动的初始边缘速度。于是同样可以构造生成器局部运输目标：

$$
\mathbb E_z
\left[
\|g_\theta(z)-g_{\theta_\tau}(z)
-\epsilon v_{\varphi^*}(g_{\theta_\tau}(z),0)\|^2
\right].
\label{eq:ganode20-reflow-generator-step}
$$

更新生成器后重新训练速度场，便得到与 GAN 类似的交替过程。不同于 MonoFlow，这里速度通过真假样本之间的条件位移回归得到，不需要先证明密度比梯度流。

## 梯度速度场把 ReFlow 变成 WGAN-GP

展开公式 \eqref{eq:ganode20-reflow-loss} 并删去与 $\varphi$ 无关的 $\|x_1-x_0\|^2/2$：

$$
\frac12\|v_\varphi(x_t,t)\|^2
-\langle v_\varphi(x_t,t),x_1-x_0\rangle.
$$

接下来需要两个额外假设：忽略显式时间输入，并把向量场限制为某个标量 critic 的梯度，$v_\varphi(x)=\nabla_xD_\varphi(x)$。由公式 \eqref{eq:ganode20-straight-path} 和链式法则，

$$
\left\langle
\nabla D_\varphi(x_t),x_1-x_0
\right\rangle
=\frac{d}{dt}D_\varphi(x_t).
$$

如果对 $t\sim U[0,1]$ 求期望，使用微积分基本定理可以精确写成端点差：

$$
\int_0^1\frac{d}{dt}D_\varphi(x_t)dt
=D_\varphi(x_1)-D_\varphi(x_0).
$$

于是速度回归目标化为

$$
\boxed{
\mathcal L_D
=\mathbb E
\left[
D_\varphi(x_0)-D_\varphi(x_1)
+\frac12\|\nabla D_\varphi(x_t)\|^2
\right]
}
\label{eq:ganode20-zero-centered-wgan-gp}
$$

这就是带零中心梯度惩罚的 WGAN 形式，而且惩罚位置 $x_t$ 正好是真假样本线性插值。原始 WGAN-GP 常惩罚 $(\|\nabla D\|-1)^2$，公式 \eqref{eq:ganode20-zero-centered-wgan-gp} 则惩罚 $\|\nabla D\|^2$；二者相关但不是同一个正则项。

当 $v=\nabla D$ 时，公式 \eqref{eq:ganode20-reflow-generator-step} 在当前参数处的一步梯度又等价于公式 \eqref{eq:ganode19-standard-generator-loss}。因此 WGAN 的 critic 损失与生成器损失都能从 ReFlow 的局部运输得到。

## 两条推导的共同闭环

MonoFlow 和 ReFlow 选择的速度来源不同：前者用 $\nabla\log(p/q_\tau)$，后者回归人为端点耦合下的条件位移；只有加入特定参数化与理想化假设后，它们才落到熟悉的 GAN loss。共同结构则很稳定：

1. 当前生成器定义可采样分布 $q_\tau$；
2. 判别器或速度网络估计 $q_\tau$ 向 $p$ 的局部运动方向；
3. 生成器用一步参数更新吸收这段样本运动；
4. 分布改变后重新估计局部方向。

因此“GAN 是扩散 ODE”不是说推理时也要多步去噪，而是说它把原本发生在单个样本上的连续运输，压进了生成器训练轨迹。训练完成后仍由 $g_\theta(z)$ 一步生成；渐进性已经储存在参数历史中。

---
title: AMED：用平均方向加速 ODE 采样
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - ODE Solver
description: 从积分平均速度、轨迹低维性和轻量蒸馏推导 AMED 的低步数采样方法。
---

> [《生成扩散模型漫谈（二十一）：中值定理加速ODE采样》](https://spaces.ac.cn/archives/9881) —— 苏剑林

## 从欧拉法的误差开始

扩散 ODE 把噪声端 $x_T$ 送到数据端 $x_0$：

$$
\frac{d x_t}{dt}=v_\theta(x_t,t).
\label{eq:amed-ode}
$$

在反向网格 $0=t_0<t_1<\cdots<t_N=T$ 上，欧拉法只用右端点速度近似整段速度：

$$
x_{t_n}\approx x_{t_{n+1}}-(t_{n+1}-t_n)v_\theta(x_{t_{n+1}},t_{n+1}).
$$

它每步只需一次网络调用，但局部误差是一阶的。Heun 法先用欧拉法得到预测点 $\tilde x_{t_n}$，再平均区间两端的速度：

$$
\begin{aligned}
\tilde x_{t_n}&=x_{t_{n+1}}-\Delta t_n v_\theta(x_{t_{n+1}},t_{n+1}),\\
x_{t_n}&\approx x_{t_{n+1}}-\frac{\Delta t_n}{2}
\left[v_\theta(x_{t_{n+1}},t_{n+1})+v_\theta(\tilde x_{t_n},t_n)\right],
\end{aligned}
$$

其中 $\Delta t_n=t_{n+1}-t_n$。高阶方法减少了截断误差，却增加每步 NFE；总 NFE 极小时，被迫使用的大步长仍会让通用高阶 Solver 失效。

## 精确目标其实是区间平均速度

对公式 \eqref{eq:amed-ode} 在 $[t_n,t_{n+1}]$ 上积分：

$$
x_{t_{n+1}}-x_{t_n}
=\int_{t_n}^{t_{n+1}}v_\theta(x_t,t)\,dt.
\label{eq:amed-integral-step}
$$

因此一步真正需要的不是端点瞬时速度，而是

$$
\bar v_n=\frac{1}{\Delta t_n}
\int_{t_n}^{t_{n+1}}v_\theta(x_t,t)\,dt.
\label{eq:amed-mean-direction}
$$

标量连续函数满足积分中值定理，存在 $s_n\in(t_n,t_{n+1})$ 使 $\bar v_n=v_\theta(x_{s_n},s_n)$。但速度是高维向量，通常不存在一个共同的 $s_n$ 让所有坐标同时满足中值等式。AMED 使用的是近似假设：短区间内轨迹足够接近直线，使平均速度可由某个中间点的速度近似。

## 把未知中间点变成可学习量

如果已知 $s_n$，仍然不知道轨迹上的 $x_{s_n}$。AMED 先用右端点速度做一次欧拉预测：

$$
\tilde x_{s_n}
=x_{t_{n+1}}-(t_{n+1}-s_n)v_\theta(x_{t_{n+1}},t_{n+1}).
$$

再用小网络 $g_\phi$ 根据 U-Net 的中间特征 $h_{t_{n+1}}$ 与时间预测 $s_n$，最终一步为

$$
x_{t_n}\approx x_{t_{n+1}}
-\Delta t_n v_\theta(\tilde x_{s_n},s_n),
\qquad
s_n=g_\phi(h_{t_{n+1}},t_{n+1}).
\label{eq:amed-update}
$$

训练 $g_\phi$ 时，先用高 NFE Solver 产生较精确的端点对，再最小化公式 \eqref{eq:amed-update} 的一步误差。教师提供的是局部轨迹端点，不是生成样本数据集；被训练的也只是很小的时间点预测器，所以蒸馏成本远低于把整个扩散模型蒸馏成生成器。

## 为什么高维中值近似仍可能有效

向量积分中值等式若严格成立，轨迹方向必须非常受限；一般曲线没有理由满足它。AMED 的经验依据是：高精度采样轨迹做 PCA 后，前一两个主成分已解释绝大多数变化，说明轨迹常落在近二维子空间，并且接近其中的一条直线。

这也与 ReFlow 一类方法的训练路径有关。训练时常用噪声与数据的线性插值作为伪轨迹，学出的速度场会受到“走直线”的偏置。于是 AMED 并非把标量定理直接推广到向量，而是利用扩散轨迹近低维、近直线这一额外结构。

## NFE 与适用边界

AMED 一步需要端点和中间点两次主网络计算，因此 NFE 近似为 2；小网络 $g_\phi$ 的成本通常忽略。比较 Solver 时必须固定总 NFE，而不是固定步数。极低 NFE 下，二阶通用方法可能因为步长过大反而不如一阶 DDIM，AMED 的优势来自对扩散轨迹的定制，而不只是形式上的阶数。

第一步还可利用高噪声端速度近似可解析的 AFS 技巧省掉一次 NFE。它依赖模型参数化在噪声端的特性，并非对所有扩散 ODE 都自动成立。

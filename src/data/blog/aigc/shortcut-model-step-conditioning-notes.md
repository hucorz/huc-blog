---
title: Shortcut Model：把步长作为生成条件
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
description: 从两小步等于一大步的自洽关系推导 Shortcut Model 的训练目标。
---

> [《生成扩散模型漫谈（二十七）：将步长作为条件输入》](https://spaces.ac.cn/archives/10617) —— 苏剑林

## 瞬时速度不能直接承担大步长

令噪声 $x_0\sim p_0$、数据 $x_1\sim p_1$，选直线路径

$$
x_t=(1-t)x_0+t x_1,
\qquad t\in[0,1].
\label{eq:shortcut-linear-path}
$$

Flow Matching 用平方损失学习瞬时速度：

$$
\mathcal L_{\text{FM}}
=\mathbb E\frac12
\left\|v_\theta(x_t,t,0)-(x_1-x_0)\right\|^2.
\label{eq:shortcut-flow-loss}
$$

最后一个输入 $0$ 表示零步长极限。推理只能用有限差分

$$
x_{t+d}=x_t+d\,v_\theta(x_t,t,0),
$$

它是欧拉近似，$d$ 越大误差越大。一步生成要求 $d=1$，恰好处于瞬时速度最不可靠的区域。

## 网络输出改为区间平均速度

Shortcut Model 让网络同时接收步长 $d$：

$$
x_{t+d}=x_t+d\,v_\theta(x_t,t,d).
\label{eq:shortcut-step-update}
$$

此时 $v_\theta(x_t,t,d)$ 的目标不再是 $t$ 点的瞬时速度，而是从 $t$ 到 $t+d$ 的有效平均速度。$d=0$ 的锚点仍由公式 \eqref{eq:shortcut-flow-loss} 监督；有限步长则从组合一致性得到监督。

从 $x_t$ 连走两次长度 $d$ 的步：

$$
\begin{aligned}
\tilde x_{t+d}&=x_t+d\,v_\theta(x_t,t,d),\\
\tilde x_{t+2d}&=\tilde x_{t+d}
+d\,v_\theta(\tilde x_{t+d},t+d,d).
\end{aligned}
$$

它应等于一次长度 $2d$ 的更新。因此平均速度必须满足

$$
v_\theta(x_t,t,2d)
=\frac12\left[
v_\theta(x_t,t,d)
+v_\theta(\tilde x_{t+d},t+d,d)
\right].
\label{eq:shortcut-self-consistency}
$$

## 用自举目标覆盖所有步长

将公式 \eqref{eq:shortcut-self-consistency} 变成损失：

$$
\mathcal L_{\text{SC}}
=\mathbb E
\left\|
v_\theta(x_t,t,2d)
-\operatorname{sg}\!\left(
\frac{v_\theta(x_t,t,d)+v_\theta(\tilde x_{t+d},t+d,d)}{2}
\right)
\right\|^2.
\label{eq:shortcut-bootstrap-loss}
$$

stop-gradient 把两个小步分支作为自举教师，避免一次更新同时移动目标。训练在二进制步长层级上从小到大传播：Flow Matching 锚定 $d=0$，较小 $d$ 的模型监督 $2d$，最终覆盖 $d=1$。若小步模型本身有偏差，误差也会逐层累积，所以锚点损失与各尺度采样比例都不能省略。

## 一套模型支持一步与多步

训练后可直接用

$$
x_1=x_0+v_\theta(x_0,0,1)
$$

一步生成；也可选 $d=1/N$ 迭代 $N$ 次。与普通 Solver 不同，改变步数时网络会明确知道当前 $d$，而不是拿同一个瞬时速度场硬套不同离散误差。

步长输入与“区间起点、终点”输入本质等价：给定 $t$ 与 $d$ 就得到区间 $[t,t+d]$。该方法不需要预训练教师和对抗训练，但公式 \eqref{eq:shortcut-bootstrap-loss} 依赖自举及 stop-gradient，理论保证不如直接监督瞬时 Flow 完整。

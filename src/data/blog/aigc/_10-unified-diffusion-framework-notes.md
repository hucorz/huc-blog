---
title: 生成扩散模型漫谈 10：统一扩散模型的自由与约束
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: true
tags:
  - AIGC
  - Diffusion
  - Generative Modeling
description: 用可采样破坏过程与边缘一致性组织统一扩散模型框架。
---

> [《生成扩散模型漫谈（十）：统一扩散模型（理论篇）》](https://spaces.ac.cn/archives/9262) —— 苏剑林

## 前向过程只需定义可采样的破坏

DDPM 从单步核 $p(x_t\mid x_{t-1})$ 出发，但训练数据实际由
$p(x_t\mid x_0)$ 产生，DDIM 也说明单步正向耦合并不是唯一选择。统一扩散模型因此直接用重参数定义任意时刻的前向样本：

$$
\varepsilon\sim q(\varepsilon),
\qquad
x_t=F_t(x_0,\varepsilon).
\label{eq:udm-forward-reparameterization}
$$

$F_t$ 可以表示加噪、模糊、遮掩、删除等变换；$x_0$ 和 $x_t$ 可以是连续数据，也可以是离散数据；$t$ 可以取离散网格，也可以连续变化。框架不要求
$q(\varepsilon)$ 一定是高斯分布。

这种自由度仍有实际约束：

- 给定训练样本 $x_0$ 和随机源 $\varepsilon$，$F_t$ 必须容易计算，才能直接构造训练对；
- $t$ 增大时，$x_t$ 中关于 $x_0$ 的信息应整体减少，使反向恢复能被分解为由易到难的小步；
- 终点分布 $p_T(x_T)$ 应容易采样，并且最好近似不依赖具体的 $x_0$；
- 若要使用后面“复用同一随机源”的确定性反向核，还需要
  $F_t(x_0,\cdot)$ 对 $\varepsilon$ 可逆。

“渐进破坏”不必理解为每个样本路径都严格单调，但相邻时间的恢复难度不能剧烈跳变，否则多步修正失去优势。

## 反向一步是预估与修正的组合

目标反向核总能按潜在的干净数据分解：

$$
p(x_{t-1}\mid x_t)
=\int
p(x_{t-1}\mid x_t,x_0)
p(x_0\mid x_t)\,dx_0.
\label{eq:udm-reverse-decomposition}
$$

若 $x_0$ 离散，积分改为求和。公式 \eqref{eq:udm-reverse-decomposition} 给出一个可以直接实施的两阶段采样过程：

$$
\hat x_0\sim q_\theta(x_0\mid x_t),
\qquad
x_{t-1}\sim
p(x_{t-1}\mid x_t,x_0=\hat x_0).
\label{eq:udm-predict-correct-sampling}
$$

第一步从当前破坏状态预估最终干净样本，第二步不要求这个预估已经正确，只用已知前向结构把状态推进到稍干净的 $t-1$。下一时刻再重新预估、重新修正，因此扩散生成的核心不是“网络每步直接预测相邻状态”，而是反复聚合对终点 $x_0$ 的不完美估计。

$q_\theta(x_0\mid x_t)$ 是对未知真实后验的学习近似。它与
$p(x_{t-1}\mid x_t,x_0)$ 的职责不同：前者必须从数据学习，后者可以在满足边缘约束的前提下根据 $F_t$ 设计。

## 后验模型决定统一训练目标

前向重参数已经能生成联合训练样本

$$
x_0\sim\tilde p(x_0),
\quad
\varepsilon\sim q(\varepsilon),
\quad
x_t=F_t(x_0,\varepsilon).
$$

因此后验模型可以直接用交叉熵训练：

$$
L_{\mathrm{UDM}}
=\mathbb E_{x_0,t,\varepsilon}
\left[
-\log q_\theta\left(
x_0\mid F_t(x_0,\varepsilon),t
\right)
\right].
\label{eq:udm-posterior-training}
$$

这个写法不要求 $x_0$ 连续。若 $x_0$ 是离散序列，
$q_\theta(x_0\mid x_t,t)$ 可以是自回归或非自回归序列模型；若
$x_0$ 连续，可以取固定或事后估计方差的条件高斯：

$$
q_\theta(x_0\mid x_t,t)
=\mathcal N\left(
x_0;G_\theta(x_t,t),\bar\sigma_t^2I
\right).
$$

忽略与 $\theta$ 无关的常数，公式 \eqref{eq:udm-posterior-training} 变为

$$
L_{\mathrm{UDM}}
=\mathbb E_{x_0,t,\varepsilon}
\left[
\frac{
\left\|x_0-G_\theta(F_t(x_0,\varepsilon),t)\right\|^2
}{2\bar\sigma_t^2}
\right].
\label{eq:udm-continuous-reconstruction}
$$

DDPM 的噪声预测只是这个后验学习的一种重参数化：当
$F_t(x_0,\varepsilon)=\bar\alpha_t x_0+\bar\beta_t\varepsilon$ 时，从
$x_0$ 预测和从 $\varepsilon$ 预测可以通过已知线性关系互换。一般的
$F_t$ 未必允许这种互换，所以统一目标应先写成概率意义明确的
$-\log q_\theta(x_0\mid x_t,t)$。

## 条件反向核必须满足边缘一致性

$p(x_{t-1}\mid x_t,x_0)$ 可以自由设计，但必须与给定的前向边缘分布兼容：

$$
\int
p(x_{t-1}\mid x_t,x_0)
p(x_t\mid x_0)\,dx_t
=p(x_{t-1}\mid x_0).
\label{eq:udm-marginal-consistency}
$$

否则，即使后验模型完美，反向一步所使用的状态分布也与前向训练分布不一致，误差会沿采样链累积。

一个永远满足公式 \eqref{eq:udm-marginal-consistency} 的简单选择是

$$
p(x_{t-1}\mid x_t,x_0)
=p(x_{t-1}\mid x_0).
\label{eq:udm-independent-correction}
$$

此时修正步骤完全忽略当前 $x_t$：先预测 $\hat x_0$，再从
$p(x_{t-1}\mid\hat x_0)$ 重新破坏一次。它在理论上合法，实际却容易受两个问题影响：

- $x_{t-1}$ 几乎完全依赖不准确的 $\hat x_0$，没有利用当前状态携带的信息；
- 每步重新采样的随机性可能冲淡刚得到的预估进展，使误差难以稳定减少。

好的条件核应在满足边缘一致性的同时，让 $x_t$ 分担对
$x_{t-1}$ 的解释，并尽量保留已有轨迹信息。

## 复用前向随机源得到确定性修正

由公式 \eqref{eq:udm-forward-reparameterization}，

$$
x_t=F_t(x_0,\varepsilon),
\qquad
x_{t-1}=F_{t-1}(x_0,\varepsilon).
$$

如果固定 $x_0$ 时 $F_t$ 关于 $\varepsilon$ 可逆，就能从
$(x_0,x_t)$ 恢复随机源：

$$
\varepsilon=F_t^{-1}(x_0,x_t).
\label{eq:udm-invert-noise}
$$

将同一 $\varepsilon$ 代入较早时刻，得到

$$
x_{t-1}
=F_{t-1}\left(
x_0,F_t^{-1}(x_0,x_t)
\right).
\label{eq:udm-shared-noise-transition}
$$

相应的条件核是一个 Dirac 分布：

$$
p(x_{t-1}\mid x_t,x_0)
=\delta\left(
x_{t-1}-
F_{t-1}(x_0,F_t^{-1}(x_0,x_t))
\right).
\label{eq:udm-deterministic-correction}
$$

为什么它满足边缘一致性？在固定 $x_0$ 下，
$x_t$ 是由 $\varepsilon\sim q$ 推出的；公式
\eqref{eq:udm-invert-noise} 把 $x_t$ 拉回同一个 $\varepsilon$，再经
$F_{t-1}$ 推出 $x_{t-1}$，所以最终分布正是
$p(x_{t-1}\mid x_0)$。这里保持的是完整的随机耦合，不只是两个边缘分布分别相同。

采样时真实 $x_0$ 会被 $\hat x_0$ 替换。公式
\eqref{eq:udm-shared-noise-transition} 同时使用当前 $x_t$ 与预估
$\hat x_0$：$x_t$ 通过恢复的随机源保留已有轨迹，$\hat x_0$ 则引导轨迹向新的干净样本移动。这比公式
\eqref{eq:udm-independent-correction} 每步重新抽噪声更容易积累进展。

可逆条件并非自动成立。若 $F_t$ 丢弃了随机源的一部分，或
$\varepsilon$ 维度与输出不匹配，$F_t^{-1}$ 可能不存在或不唯一，需要另行设计耦合或引入辅助随机变量。

## 高斯随机源允许在确定性与随机性间插值

当 $q(\varepsilon)=\mathcal N(0,I)$ 时，可以利用标准高斯的旋转不变性：

$$
\sqrt{1-\tilde\sigma_t^2}\,\varepsilon_1
+\tilde\sigma_t\varepsilon_2
\sim\mathcal N(0,I),
\qquad
\varepsilon_1,\varepsilon_2\overset{\mathrm{iid}}{\sim}\mathcal N(0,I).
$$

用从 $(x_0,x_t)$ 恢复的随机源替换 $\varepsilon_1$，保留新噪声
$\varepsilon_2$，得到一族条件采样：

$$
x_{t-1}
=F_{t-1}\left(
x_0,
\sqrt{1-\tilde\sigma_t^2}\,
F_t^{-1}(x_0,x_t)
+\tilde\sigma_t\varepsilon
\right),
\quad
\varepsilon\sim\mathcal N(0,I).
\label{eq:udm-stochastic-correction-family}
$$

$\tilde\sigma_t=0$ 时完全复用现有随机源，是确定性修正；增大
$\tilde\sigma_t$ 会减少与当前轨迹的耦合并注入新随机性；
$\tilde\sigma_t=1$ 时退化为忽略 $x_t$ 的重新采样。无论取值如何，括号中的随机源边缘分布仍是标准高斯，所以公式
\eqref{eq:udm-marginal-consistency} 保持成立。

这与 DDIM 中自由选择反向方差是同一个结构：边缘分布由前向扰动核规定，联合耦合仍有自由度。确定性和随机性并不是两套模型，而是同一边缘约束下不同的轨迹设计。

## 框架中的自由项与约束项

统一框架可以按下面的依赖关系复习：

1. 自由选择数据类型、时间类型、随机源 $q(\varepsilon)$ 和破坏函数
   $F_t$，但必须保证可采样、渐进破坏以及终点先验易采样。
2. 用 $F_t$ 直接构造训练对，通过公式
   \eqref{eq:udm-posterior-training} 学习
   $q_\theta(x_0\mid x_t,t)$；输出分布要适配 $x_0$ 的数据类型。
3. 自由设计 $p(x_{t-1}\mid x_t,x_0)$，但必须满足公式
   \eqref{eq:udm-marginal-consistency}。
4. 生成时按公式 \eqref{eq:udm-predict-correct-sampling} 反复预估
   $x_0$ 并做一步修正，直到回到 $t=0$。

DDPM、DDIM、离散扩散和非高斯破坏的差别主要落在
$F_t$、后验模型族和满足边缘一致性的条件核上。统一之处不是某一条高斯公式，而是“可采样前向边缘 + 可学习干净后验 + 边缘一致的预估修正”这三个部件。

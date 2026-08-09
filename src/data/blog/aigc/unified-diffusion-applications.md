---
title: 统一扩散框架的几种落地方式
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - Generative Modeling
description: 从统一反向分解出发，串起热扩散、冷扩散、离散编辑与可学习编码器。
---

> [《生成扩散模型漫谈（十一）：统一扩散模型（应用篇）》](https://spaces.ac.cn/archives/9271) —— 苏剑林

## 反向生成先拆成两个问题

统一扩散模型（Unified Diffusion Model，UDM）不先规定“必须加高斯噪声”，而是把前向破坏抽象为

$$
x_t=\mathcal F_t(x_0,\varepsilon),
\qquad \varepsilon\sim q(\varepsilon).
\label{eq:udm11-forward-transform}
$$

$\mathcal F_t$ 可以是加噪、模糊、遮掩或离散 token 替换；$t$ 也不必局限于离散时间。真正共同的结构在反向过程：给定 $x_t$，先恢复一份干净数据候选，再按照已知前向机制回到较轻的破坏程度：

$$
\hat x_0\sim q_\theta(x_0\mid x_t),
\qquad
x_{t-1}\sim p(x_{t-1}\mid x_t,x_0=\hat x_0).
\label{eq:udm11-reverse-factorization}
$$

第一步是学习问题。连续数据常用条件高斯分布建模 $q_\theta$，对应平方误差；离散序列可以用自回归或非自回归分类器，对应交叉熵。第二步主要由 $\mathcal F_t$ 的结构决定。

最保守的选择是不利用当前状态 $x_t$ 中关于破坏路径的信息，直接重新执行 $t-1$ 时刻的前向变换：

$$
p(x_{t-1}\mid x_t,x_0)=p(x_{t-1}\mid x_0),
\qquad
x_{t-1}=\mathcal F_{t-1}(x_0,\varepsilon').
$$

这总是可用，但可能毁掉刚刚恢复出的内容。若给定 $(x_0,x_t)$ 后能唯一反解出本次使用的噪声，即 $\mathcal F_t$ 对 $\varepsilon$ 可逆，则可以保留原有破坏路径：

$$
x_{t-1}
=\mathcal F_{t-1}\!\left(
x_0,\mathcal F_t^{-1}(x_0,x_t)
\right).
\label{eq:udm11-path-preserving-step}
$$

这里的“可逆”是指对噪声变量可逆，不要求 $x_0\mapsto x_t$ 没有信息损失。如果噪声还是标准高斯，还能把反解出的旧噪声与新噪声按方差守恒混合，用一个超参数在确定性路径和随机采样之间插值。

## 热扩散：DDPM 与 DDIM 只是一个参数选择

主流高斯扩散取

$$
x_t=\bar\alpha_t x_0+\bar\beta_t\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I).
$$

若令 $q_\theta(x_0\mid x_t)=\mathcal N(\bar\mu_\theta(x_t,t),\bar\sigma_t^2I)$，并使用噪声预测参数化

$$
\bar\mu_\theta(x_t,t)
=\frac{x_t-\bar\beta_t\epsilon_\theta(x_t,t)}{\bar\alpha_t},
$$

那么负对数似然中与模型有关的部分为

$$
-\log q_\theta(x_0\mid x_t)
=\frac{\bar\beta_t^2}
{2\bar\sigma_t^2\bar\alpha_t^2}
\left\|
\varepsilon-
\epsilon_\theta(\bar\alpha_t x_0+\bar\beta_t\varepsilon,t)
\right\|^2+C.
$$

常见 simple loss 会删掉只依赖 $t$ 的前置权重。这会改变不同噪声时刻的相对权重，并非严格的代数恒等。

由当前状态可以反解出 $\varepsilon=(x_t-\bar\alpha_tx_0)/\bar\beta_t$。把同一噪声的一部分保留下来，再补一份独立噪声 $z\sim\mathcal N(0,I)$，得到统一的单步形式：

$$
x_{t-1}
=\bar\alpha_{t-1}x_0
+\sqrt{\bar\beta_{t-1}^2-\sigma_t^2}
\frac{x_t-\bar\alpha_tx_0}{\bar\beta_t}
+\sigma_tz.
\label{eq:udm11-hot-reverse-step}
$$

实际采样时用 $\hat x_0$ 替换未知的 $x_0$。$\bar\sigma_t$ 控制干净样本估计本身的不确定性，$\sigma_t$ 控制从 $t$ 到 $t-1$ 额外加入的随机性。原文列出的对应关系是：DDPM 令 $\bar\sigma_t=0$ 且 $\sigma_t=\bar\beta_{t-1}\beta_t/\bar\beta_t$，DDIM 再令 $\sigma_t=0$，Analytical-DPM 则重新估计非零的最优 $\bar\sigma_t$。因此这些方法的共同骨架是公式 \eqref{eq:udm11-reverse-factorization}，差异主要落在两种方差怎样选。

## 冷扩散：无噪声变换的能力与缺口

Cold Diffusion 用模糊、遮挡、池化等确定性变换破坏图像。为了看见它隐含的概率条件，可以先写成带微小噪声的形式，最后再取 $\sigma\to0$：

$$
x_t=\mathcal F_t(x_0)+\sigma\varepsilon.
$$

令恢复网络为 $\mathcal G_t$，使用 $L_1$ 重建时，训练目标是

$$
\left\|
x_0-\mathcal G_t(\mathcal F_t(x_0))
\right\|_1.
$$

基准反向采样先预测 $\hat x_0=\mathcal G_t(x_t)$，再重新做一次较轻的破坏：

$$
x_{t-1}=\mathcal F_{t-1}(\hat x_0).
$$

若把前一步留下的残差 $x_t-\mathcal F_t(\hat x_0)$ 继续带到下一步，就得到改进采样：

$$
x_{t-1}
=x_t+\mathcal F_{t-1}(\hat x_0)-\mathcal F_t(\hat x_0).
\label{eq:udm11-cold-improved-step}
$$

公式 \eqref{eq:udm11-cold-improved-step} 的作用不是凭空增加信息，而是保留当前状态相对预测轨迹的偏差，避免每一步都从 $\hat x_0$ 重新出发。

> 既然前向过程可以完全确定，为什么生成时反而需要随机性？

如果模糊把一张 $w\times w\times3$ 图像最终压到近似一个三维颜色向量，前向映射就是从 $3w^2$ 维到 $3$ 维的多对一映射。确定性的三维输入不可能唯一还原所有丢失细节；反向过程若也确定，就只能为同一个终点选出一种重建。噪声提供的是条件分布中缺失的自由度，使一个低维终点能对应多个合理样本。Cold Diffusion 实验中给终点加入轻微高维随机噪声能改善生成，正说明“Without Noise”并不是可以无条件坚持的原则。

## 离散编辑：关键在能否识别破坏位置

对长度为 $l$ 的定长序列，可以把前向过程定义为随机替换 $t$ 个 token。训练 $q_\theta(x_0\mid x_t)$ 用交叉熵恢复原序列并不困难，难点仍在公式 \eqref{eq:udm11-reverse-factorization} 的第二步。

若替换时允许抽中原 token，那么仅比较 $x_0$ 与 $x_t$ 无法判断哪些位置执行过替换，噪声路径不可逆。基准采样只能在每轮预测完整 $\hat x_0$ 后，再随机破坏其中 $t-1$ 个位置，前一轮已经预测正确的 token 也可能被毁掉。

若规定每个被选位置必须换成不同 token，比较 $(x_0,x_t)$ 就能找出所有修改位置。反向时要求 $\hat x_0$ 与 $x_t$ 恰有 $t$ 个位置不同，再只选其中一个位置改回预测 token。这样 $x_t\to x_{t-1}$ 只改变一个位置，已确认的部分会一直保留，也不再受从左到右的自回归顺序限制。

随机 Mask 是更直观的可逆例子：$x_t$ 中的 `[MASK]` 直接暴露了破坏位置。于是可以从全 Mask 序列开始，每一步只在剩余 Mask 中选一个位置，按照 MLM 给出的条件概率采样 token。它与基于 MLM 的 Gibbs 采样非常接近。这里并非说离散编辑在数值上等同于高斯扩散，而是二者共享“预测 $x_0$，再沿已知破坏路径少退一步”的结构。

## 把编码器也放进前向过程

前向变换不必是固定算子。令 $\mathcal E_\phi$ 为可学习编码器，可以构造

$$
x_t
=\bar\alpha_t\mathcal E_\phi(x_0)
+\bar\beta_t\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I).
\label{eq:udm11-learned-encoder}
$$

此时 $q_\theta(x_0\mid x_t)$ 是解码器，$\phi$ 与 $\theta$ 一起通过

$$
-\log q_\theta\!\left(
x_0\mid
\bar\alpha_t\mathcal E_\phi(x_0)+\bar\beta_t\varepsilon
\right)
$$

训练。因为原始对象先经过编码，$x_0$ 可以是连续数据，也可以是离散数据；高斯扩散发生在表示空间，最后一步由解码分布直接返回原对象。这与 VAE 的“编码到简单隐变量，再解码回来”有相似目标，但这里仍保留多时刻的渐进破坏与反向生成。

UDM 的价值不只是把几种算法换一套记号。它给出了一个检查新扩散方案的顺序：先说明破坏算子与噪声，再确定如何从 $x_t$ 建模 $x_0$，最后检查噪声路径是否可逆、信息是否丢失，以及反向随机性是否足以覆盖条件分布。热扩散、冷扩散和离散编辑的差异，都能在这几个问题上定位。

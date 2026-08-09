---
title: 扩散模型的反向方差估计
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - Analytic DPM
description: 从后验不确定性推导 Analytic-DPM 与条件反向方差。
---

> [《生成扩散模型漫谈（七）：最优扩散方差估计（上）》](https://spaces.ac.cn/archives/9245) —— 苏剑林
>
> [《生成扩散模型漫谈（八）：最优扩散方差估计（下）》](https://spaces.ac.cn/archives/9246) —— 苏剑林

## 固定 $x_0$ 的随机性不是全部随机性

设正向扰动核为

$$
p(x_t\mid x_0)=\mathcal N(x_t;\bar\alpha_t x_0,\bar\beta_t^2I),
$$

DDIM 给出的条件反向核可以写成

$$
p(x_{t-1}\mid x_t,x_0)
=\mathcal N\left(
x_{t-1};
c_t x_t+\gamma_t x_0,
\sigma_t^2I
\right),
\label{eq:analytic-dpm-conditional-kernel}
$$

其中

$$
c_t=\frac{\sqrt{\bar\beta_{t-1}^2-\sigma_t^2}}{\bar\beta_t},
\qquad
\gamma_t=\bar\alpha_{t-1}-c_t\bar\alpha_t.
$$

DDIM 通常用单点估计 $\bar\mu_t(x_t)$ 替换 $x_0$。这会保留公式
\eqref{eq:analytic-dpm-conditional-kernel} 中显式的方差 $\sigma_t^2I$，却丢掉“给定 $x_t$ 后，真实 $x_0$ 仍不确定”带来的随机性。

严格的边缘化应是

$$
p(x_{t-1}\mid x_t)
=\int p(x_{t-1}\mid x_t,x_0)p(x_0\mid x_t)\,dx_0.
\label{eq:analytic-dpm-reverse-marginalization}
$$

用各向同性高斯近似未知后验

$$
p(x_0\mid x_t)
\approx\mathcal N(x_0;\bar\mu_t(x_t),\bar\sigma_t^2I),
\label{eq:analytic-dpm-posterior-approximation}
$$

并写成 $x_0=\bar\mu_t(x_t)+\bar\sigma_t\varepsilon_2$。将它代入条件采样式：

$$
\begin{aligned}
x_{t-1}
&=c_t x_t+\gamma_t x_0+\sigma_t\varepsilon_1\\
&\approx c_t x_t+\gamma_t\bar\mu_t(x_t)
+\gamma_t\bar\sigma_t\varepsilon_2
+\sigma_t\varepsilon_1.
\end{aligned}
$$

$\varepsilon_1$、$\varepsilon_2$ 相互独立，所以两部分随机性的方差相加：

$$
p(x_{t-1}\mid x_t)
\approx\mathcal N\left(
x_{t-1};
c_t x_t+\gamma_t\bar\mu_t(x_t),
\left(\sigma_t^2+\gamma_t^2\bar\sigma_t^2\right)I
\right).
\label{eq:analytic-dpm-corrected-reverse-variance}
$$

公式 \eqref{eq:analytic-dpm-corrected-reverse-variance} 中的
$\gamma_t^2\bar\sigma_t^2$ 是 Analytic-DPM 的核心修正。即便显式取
$\sigma_t=0$，只要 $p(x_0\mid x_t)$ 没有退化成单点，边缘化后的反向过程仍有非零方差。确定性 DDIM 相当于额外忽略了这部分后验不确定性。

## 均值仍由噪声预测器给出

平方损失下，后验均值是给定 $x_t$ 后预测 $x_0$ 的最优函数：

$$
\bar\mu_t(x_t)
=\mathbb E[x_0\mid x_t]
=\arg\min_{\mu(x_t)}
\mathbb E\left[
\|x_0-\mu(x_t)\|^2
\right].
\label{eq:analytic-dpm-posterior-mean}
$$

使用扩散模型常见的参数化

$$
\bar\mu_t(x_t)
=\frac{1}{\bar\alpha_t}
\left[x_t-\bar\beta_t\varepsilon_\theta(x_t,t)\right]
\label{eq:analytic-dpm-mean-parameterization}
$$

后，公式 \eqref{eq:analytic-dpm-posterior-mean} 就是噪声预测目标。Analytic-DPM 不改变这个均值网络，而是要在它训练完成后估计
$p(x_0\mid x_t)$ 中尚未被条件均值解释的方差。

## 用全方差公式估计各向同性后验

先假设公式 \eqref{eq:analytic-dpm-mean-parameterization} 精确等于真实条件均值。对任意常向量 $\mu_0$，条件协方差满足

$$
\begin{aligned}
\Sigma(x_t)
&=\mathbb E\left[
(x_0-\bar\mu_t)(x_0-\bar\mu_t)^\top
\mid x_t
\right]\\
&=\mathbb E\left[
(x_0-\mu_0)(x_0-\mu_0)^\top
\mid x_t
\right]
-(\bar\mu_t-\mu_0)(\bar\mu_t-\mu_0)^\top.
\end{aligned}
\label{eq:analytic-dpm-conditional-covariance-shift}
$$

为了得到只依赖时间 $t$ 的单个标量方差，先对 $x_t$ 求平均，再取协方差矩阵的迹并除以维度 $d$：

$$
\bar\sigma_t^2
=\frac1d\mathbb E\|x_0-\mu_0\|^2
-\frac1d\mathbb E\|\bar\mu_t(x_t)-\mu_0\|^2.
\label{eq:analytic-dpm-total-variance-estimator}
$$

取 $\mu_0=\mathbb E[x_0]$ 时，第一项就是数据的平均逐维方差。公式
\eqref{eq:analytic-dpm-total-variance-estimator} 是全方差公式的迹版本：总方差等于“条件均值之间的方差”与“条件内剩余方差”之和。

这条估计直观，但需要统计数据方差和去噪均值。还可以把
$\bar\mu_t$ 的噪声参数化直接代入，得到只依赖噪声网络输出的 Analytic-DPM 形式。

## 从噪声网络输出得到解析方差

由公式 \eqref{eq:analytic-dpm-mean-parameterization}，在 Perfect Mean 假设下有

$$
\begin{aligned}
\Sigma(x_t)
={}&\frac{1}{\bar\alpha_t^2}
\mathbb E\left[
(x_t-\bar\alpha_t x_0)(x_t-\bar\alpha_t x_0)^\top
\mid x_t
\right]\\
&-\frac{\bar\beta_t^2}{\bar\alpha_t^2}
\varepsilon_\theta(x_t,t)\varepsilon_\theta(x_t,t)^\top.
\end{aligned}
\label{eq:analytic-dpm-covariance-from-noise}
$$

对 $x_t$ 求平均后，公式第一项中的嵌套期望可以换回
$x_0\sim\tilde p(x_0)$、$x_t\sim p(x_t\mid x_0)$ 的联合采样。由于
$x_t-\bar\alpha_t x_0=\bar\beta_t\varepsilon$，其二阶矩就是
$\bar\beta_t^2I$，于是

$$
\mathbb E_{x_t}[\Sigma(x_t)]
=\frac{\bar\beta_t^2}{\bar\alpha_t^2}
\left[
I-\mathbb E_{x_t}
\left[
\varepsilon_\theta(x_t,t)
\varepsilon_\theta(x_t,t)^\top
\right]
\right].
\label{eq:analytic-dpm-average-covariance}
$$

取迹并除以 $d$，得到各向同性估计：

$$
\boxed{
\bar\sigma_t^2
=\frac{\bar\beta_t^2}{\bar\alpha_t^2}
\left(
1-\frac1d\mathbb E_{x_t}
\|\varepsilon_\theta(x_t,t)\|^2
\right)
}
\label{eq:analytic-dpm-isotropic-estimator}
$$

它可以在均值模型训练完成后，对每个 $t$ 采样一批 $x_t$ 离线统计，不需要重新训练扩散模型。理论上
$\bar\sigma_t^2\le\bar\beta_t^2/\bar\alpha_t^2$；有限样本、模型误差或数值误差可能让括号中的估计越界，实际实现需要保证方差非负。

如果不再强迫所有维度共享同一方差，只取公式
\eqref{eq:analytic-dpm-average-covariance} 的对角线，就得到向量方差：

$$
\bar{\boldsymbol\sigma}_t^2
=\frac{\bar\beta_t^2}{\bar\alpha_t^2}
\left(
\boldsymbol 1-
\mathbb E_{x_t}[\varepsilon_\theta(x_t,t)^2]
\right),
\label{eq:analytic-dpm-diagonal-perfect-mean}
$$

平方为逐元素平方。完整 $d\times d$ 协方差虽然也有公式，但图像维度下存储、预测和采样成本过高，对角近似是更现实的折中。

## Imperfect Mean 下应估计残差二阶矩

上面的“总方差减去已解释方差”依赖
$\bar\mu_t(x_t)=\mathbb E[x_0\mid x_t]$。真实网络并不精确，若均值固定但可能有偏，最合适的高斯方差应直接从似然求出。

对各向同性近似
$\mathcal N(x_0;\bar\mu_t(x_t),\bar\sigma_t^2I)$，平均负对数似然中与方差有关的部分是

$$
\mathcal L(\bar\sigma_t^2)
=\frac{\mathbb E\|x_0-\bar\mu_t(x_t)\|^2}
{2\bar\sigma_t^2}
+\frac d2\log\bar\sigma_t^2.
$$

对 $\bar\sigma_t^2$ 求导并令其为零：

$$
\bar\sigma_t^2
=\frac1d\mathbb E_{x_0,x_t}
\|x_0-\bar\mu_t(x_t)\|^2.
\label{eq:analytic-dpm-imperfect-mean-mle}
$$

把 $x_t=\bar\alpha_t x_0+\bar\beta_t\varepsilon$ 和公式
\eqref{eq:analytic-dpm-mean-parameterization} 代入，可得

$$
\boxed{
\bar\sigma_t^2
=\frac{\bar\beta_t^2}{\bar\alpha_t^2d}
\mathbb E_{x_0,\varepsilon}
\left\|
\varepsilon-
\varepsilon_\theta(
\bar\alpha_t x_0+\bar\beta_t\varepsilon,t
)
\right\|^2
}
\label{eq:analytic-dpm-imperfect-mean-noise-residual}
$$

Perfect Mean 时，残差只表示不可约的后验不确定性；Imperfect Mean 时，它还包含均值模型的系统误差。因此公式
\eqref{eq:analytic-dpm-imperfect-mean-noise-residual} 估计的是“相对于当前固定均值，最优高斯近似应使用的残差二阶矩”，不能再解释成纯粹的真实条件方差。

逐维版本只需去掉维度平均：

$$
\bar{\boldsymbol\sigma}_t^2
=\frac{\bar\beta_t^2}{\bar\alpha_t^2}
\mathbb E_{x_0,\varepsilon}
\left[
(\varepsilon-\varepsilon_\theta(x_t,t))^2
\right].
\label{eq:analytic-dpm-imperfect-mean-diagonal}
$$

## 条件方差需要学习后验残差

无条件的 $\bar{\boldsymbol\sigma}_t^2$ 对所有 $x_t$ 做了平均。若不同带噪输入有不同的不确定性，应保留 $x_t$ 依赖：

$$
\bar{\boldsymbol\sigma}_t^2(x_t)
=\frac{\bar\beta_t^2}{\bar\alpha_t^2}
\mathbb E\left[
(\varepsilon_t-\varepsilon_\theta(x_t,t))^2
\mid x_t
\right],
\label{eq:analytic-dpm-conditional-variance-target}
$$

其中
$\varepsilon_t=(x_t-\bar\alpha_t x_0)/\bar\beta_t$。条件期望无法为每个 $x_t$ 重复采样真实后验，但仍可利用平方回归：训练网络 $g_\phi(x_t,t)$ 预测逐元素目标

$$
r_t=(\varepsilon_t-\varepsilon_\theta(x_t,t))^2,
$$

并最小化

$$
\mathbb E_{x_0,t,\varepsilon_t}
\left[
\|r_t-g_\phi(x_t,t)\|^2
\right].
\label{eq:analytic-dpm-conditional-variance-regression}
$$

平方损失的最优解是
$g_\phi(x_t,t)=\mathbb E[r_t\mid x_t]$，乘上
$\bar\beta_t^2/\bar\alpha_t^2$ 后就是公式
\eqref{eq:analytic-dpm-conditional-variance-target}。

## 为什么采用两阶段训练

均值决定反向一步往哪里走，方差只是决定在这个方向附近注入多少随机性。若从头联合训练，持续变化的均值会让方差目标本身不断移动，而可学习方差又会改变负对数似然对均值误差的加权，两者容易相互干扰。

Extended-Analytic-DPM 因此先用固定方差训练好均值/噪声网络，再冻结它，离线统计解析方差或训练条件方差头。这样做允许复用已有模型，也让公式
\eqref{eq:analytic-dpm-imperfect-mean-noise-residual} 中的残差目标保持稳定。

方差修正在少步采样时通常更重要：跨步越大，单步条件分布越宽，忽略 $x_0\mid x_t$ 的不确定性造成的误差越明显；接近完整细网格时，均值误差和数值离散误差往往仍是主要因素，精细方差更像次级改进。

还要保留一个经验上的边界：原文提到基于 Perfect Mean 假设的方案在实验中反而优于显式面向 Imperfect Mean 的残差方案。这不等于均值网络数学上真的完美，只说明估计偏差、有限样本方差和训练目标的组合可能让更严格的公式未必给出更好的生成指标。

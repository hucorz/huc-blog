---
title: 生成扩散模型漫谈 02-04：从 VAE 与贝叶斯推导 DDPM 和 DDIM
author: Huc
pubDatetime: 2026-08-09
modDatetime: 2026-08-09
featured: false
draft: true
tags:
  - AIGC
  - Diffusion
  - DDPM
  - DDIM
description: 从联合分布、条件后验和边缘一致性推导 DDPM 与 DDIM。
---

> [《生成扩散模型漫谈（二）：DDPM = 自回归式VAE》](https://spaces.ac.cn/archives/9152) —— 苏剑林
>
> [《生成扩散模型漫谈（三）：DDPM = 贝叶斯 + 去噪》](https://spaces.ac.cn/archives/9164) —— 苏剑林
>
> [《生成扩散模型漫谈（四）：DDIM = 高观点DDPM》](https://spaces.ac.cn/archives/9181) —— 苏剑林

## 从单步 VAE 到多步隐变量

普通 VAE 用一步编码 $x\to z$ 和一步生成 $z\to x$。当编码分布、先验和生成分布都限制为容易计算的高斯分布时，一步映射要独自承担整个复杂数据分布，表达能力容易受限。

DDPM 把一个困难的大变化拆成 $T$ 个局部变化：

$$
x_0\to x_1\to\cdots\to x_T,
\qquad
x_T\to x_{T-1}\to\cdots\to x_0.
$$

每一步仍然可以是条件高斯分布，但多个局部高斯转移复合后不再等同于单个简单高斯映射。这里的“自回归”发生在扩散时间轴上：生成 $x_{t-1}$ 时只依赖 $x_t$，是一个一阶马尔可夫生成过程。

沿用原文第二篇的记号，$p$ 表示固定的正向编码分布，$q$ 表示需要学习的反向生成分布；这与 DDPM 原论文常见的 $p_\theta$、$q$ 记号刚好相反：

$$
\begin{aligned}
p(x_{0:T})
&=\tilde p(x_0)\prod_{t=1}^{T}p(x_t\mid x_{t-1}),\\
q_\theta(x_{0:T})
&=q(x_T)\prod_{t=1}^{T}q_\theta(x_{t-1}\mid x_t).
\end{aligned}
\label{eq:ddpm-vae-joint-distributions}
$$

其中 $\tilde p(x_0)$ 是数据分布，$q(x_T)=\mathcal N(0,I)$ 是生成起点。训练可以理解为最小化两个完整轨迹分布之间的
$\operatorname{KL}(p\Vert q_\theta)$。

## 联合 KL 怎样拆成逐步去噪

正向过程固定为

$$
p(x_t\mid x_{t-1})
=\mathcal N(x_t;\alpha_t x_{t-1},\beta_t^2I),
\qquad \alpha_t^2+\beta_t^2=1.
\label{eq:ddpm-vae-forward-kernel}
$$

反向过程则设为

$$
q_\theta(x_{t-1}\mid x_t)
=\mathcal N(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2I),
$$

其中只有均值网络含可训练参数。把公式 \eqref{eq:ddpm-vae-joint-distributions} 代入联合 KL 后，正向分布自身的对数项、固定先验 $q(x_T)$ 和高斯归一化常数都不依赖 $\theta$。与第 $t$ 步模型有关的部分只剩

$$
\mathbb E_{p(x_0,x_{t-1},x_t)}
\left[
\frac{1}{2\sigma_t^2}
\left\|x_{t-1}-\mu_\theta(x_t,t)\right\|^2
\right].
\label{eq:ddpm-vae-step-reconstruction}
$$

从完整轨迹降到 $(x_0,x_{t-1},x_t)$，用到的是马尔可夫结构：$x_{t+1:T}$ 的条件分布可积分为 $1$，$x_{1:t-2}$ 则可边缘化为 $p(x_{t-1}\mid x_0)$。因此这不是把联合 KL 直接假设成若干独立 loss，而是逐项积分后的结果。

令 $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$、$\bar\beta_t=\sqrt{1-\bar\alpha_t^2}$，正向边缘分布为

$$
p(x_t\mid x_0)=\mathcal N(x_t;\bar\alpha_t x_0,\bar\beta_t^2I),
\qquad
x_t=\bar\alpha_t x_0+\bar\beta_t\varepsilon.
\label{eq:ddpm-vae-forward-marginal}
$$

用单步关系 $x_{t-1}=\alpha_t^{-1}(x_t-\beta_t\varepsilon_t)$ 参数化反向均值，公式 \eqref{eq:ddpm-vae-step-reconstruction} 就会变成噪声回归。再把两份相关噪声旋转成正交高斯变量、对不出现在模型输入中的那一维精确求期望，最终得到常见的简化目标：

$$
L_{\mathrm{simple}}
=\mathbb E_{x_0,t,\varepsilon}
\left[
\left\|
\varepsilon-\varepsilon_\theta(
\bar\alpha_t x_0+\bar\beta_t\varepsilon,t
)
\right\|^2
\right].
\label{eq:ddpm-vae-simple-objective}
$$

从联合 KL 严格推下来的目标在公式 \eqref{eq:ddpm-vae-simple-objective} 前还有只依赖 $t$ 的权重。实践中把它去掉，会改变各时间步的相对权重，因此这是训练目标的简化，不是纯代数恒等变换。

这种视角也解释了 DDPM 为什么更像“只保留生成能力的 VAE”：正向核没有可训练编码器，而且 Noise Scheduler 要让 $\bar\alpha_T\approx0$，使 $p(x_T\mid x_0)$ 几乎与 $x_0$ 无关。模型得到的是从公共噪声先验出发的生成器，而不是能保留单个输入身份的语义编码器。

## 贝叶斯后验直接给出反向一步

联合 KL 说明了目标来自哪里，但要看清反向分布的结构，更直接的路线是条件贝叶斯。不能直接计算
$p(x_{t-1}\mid x_t)$，因为真实边缘分布 $p(x_{t-1})$ 和 $p(x_t)$ 都依赖未知数据分布；给定 $x_0$ 后，三项却都已知：

$$
p(x_{t-1}\mid x_t,x_0)
=\frac{p(x_t\mid x_{t-1})p(x_{t-1}\mid x_0)}
{p(x_t\mid x_0)}.
\label{eq:ddpm-bayes-posterior-start}
$$

三个分布都是高斯。把它们的指数项合并并对 $x_{t-1}$ 配方，可得

$$
p(x_{t-1}\mid x_t,x_0)
=\mathcal N(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\sigma_t^2I),
\label{eq:ddpm-bayes-posterior}
$$

其中

$$
\begin{aligned}
\tilde\mu_t(x_t,x_0)
&=\frac{\alpha_t\bar\beta_{t-1}^2}{\bar\beta_t^2}x_t
+\frac{\bar\alpha_{t-1}\beta_t^2}{\bar\beta_t^2}x_0,\\
\tilde\sigma_t^2
&=\frac{\bar\beta_{t-1}^2\beta_t^2}{\bar\beta_t^2}.
\end{aligned}
\label{eq:ddpm-bayes-posterior-parameters}
$$

关键限制是生成时没有 $x_0$。因此先训练去噪器，由当前状态估计最终干净样本：

$$
\hat x_{0,\theta}(x_t,t)
=\frac{1}{\bar\alpha_t}
\left[x_t-\bar\beta_t\varepsilon_\theta(x_t,t)\right].
\label{eq:ddpm-bayes-x0-estimator}
$$

把公式 \eqref{eq:ddpm-bayes-x0-estimator} 代替后验中的真实 $x_0$，就得到只依赖 $x_t$ 的近似反向核：

$$
p_\theta(x_{t-1}\mid x_t)
\approx
\mathcal N\left(
x_{t-1};
\frac{1}{\alpha_t}
\left[x_t-\frac{\beta_t^2}{\bar\beta_t}
\varepsilon_\theta(x_t,t)\right],
\frac{\bar\beta_{t-1}^2\beta_t^2}{\bar\beta_t^2}I
\right).
\label{eq:ddpm-bayes-reverse-kernel}
$$

这里的 $\hat x_{0,\theta}$ 不需要一次就准确。它只是先给出终点的粗估计，再借助精确条件后验结构从 $x_t$ 前进一步；下一步重新估计、重新修正。逐步采样可以看成不断执行“远期预估 + 局部修正”。

两个极端数据分布还能解释 DDPM 常见的两种固定方差：若数据分布退化为单点，后验方差就是公式 \eqref{eq:ddpm-bayes-posterior-parameters} 中的 $\tilde\sigma_t^2$；若数据本身就是标准高斯，则正反向平稳，反向方差为 $\beta_t^2$。真实数据介于这些特例之外，所以两者是有理论来源的可用选择，不是对一般数据的最优性证明。

## DDIM 只保留真正需要的边缘分布

DDPM 的训练只用到 $p(x_t\mid x_0)$，采样只用到反向核。于是可以不再把单步正向核 $p(x_t\mid x_{t-1})$ 当作出发点，只要求不同时间的边缘分布保持为公式 \eqref{eq:ddpm-vae-forward-marginal}。

在未知正向联合耦合的情况下，设一个更一般的条件后验：

$$
p(x_{t-1}\mid x_t,x_0)
=\mathcal N(x_{t-1};\kappa_t x_t+\lambda_t x_0,\sigma_t^2I).
$$

它必须满足边缘一致性

$$
\int p(x_{t-1}\mid x_t,x_0)p(x_t\mid x_0)\,dx_t
=p(x_{t-1}\mid x_0).
\label{eq:ddim-marginal-consistency}
$$

将 $x_t=\bar\alpha_t x_0+\bar\beta_t\varepsilon_1$ 代入待定的条件采样式后，得到

$$
x_{t-1}
=(\kappa_t\bar\alpha_t+\lambda_t)x_0
+\kappa_t\bar\beta_t\varepsilon_1
+\sigma_t\varepsilon_2.
$$

要让它与
$x_{t-1}=\bar\alpha_{t-1}x_0+\bar\beta_{t-1}\varepsilon$ 同分布，只需匹配均值系数和噪声方差：

$$
\bar\alpha_{t-1}=\kappa_t\bar\alpha_t+\lambda_t,
\qquad
\bar\beta_{t-1}^2=\kappa_t^2\bar\beta_t^2+\sigma_t^2.
\label{eq:ddim-coefficient-constraints}
$$

两个方程只有三个未知量，所以 $\sigma_t$ 成为自由参数，并有

$$
\kappa_t=\frac{\sqrt{\bar\beta_{t-1}^2-\sigma_t^2}}{\bar\beta_t},
\qquad
\lambda_t=\bar\alpha_{t-1}
-\frac{\bar\alpha_t\sqrt{\bar\beta_{t-1}^2-\sigma_t^2}}
{\bar\beta_t}.
\label{eq:ddim-coefficient-solution}
$$

这说明 DDPM 的那一种后验只是满足相同边缘分布的一种耦合。只要每个 $p(x_t\mid x_0)$ 不变，公式 \eqref{eq:ddpm-vae-simple-objective} 就不变，训练好的噪声预测器可以直接复用；改变的是反向采样轨迹。

把 $x_0$ 替换成公式 \eqref{eq:ddpm-bayes-x0-estimator} 后，一步采样可写为更有解释力的形式：

$$
x_{t-1}
=\bar\alpha_{t-1}\hat x_{0,\theta}
+\sqrt{\bar\beta_{t-1}^2-\sigma_t^2}\,
\varepsilon_\theta(x_t,t)
+\sigma_t z,
\quad z\sim\mathcal N(0,I).
\label{eq:ddim-general-sampling}
$$

三项分别是预测的干净样本、沿当前预测噪声方向保留的部分，以及新注入的随机噪声。取
$\sigma_t=\bar\beta_{t-1}\beta_t/\bar\beta_t$ 会回到 DDPM 的后验方差；取 $\sigma_t=0$，最后一项消失，从 $x_T$ 到 $x_0$ 成为确定性映射，这才是通常狭义所称的 DDIM。

## 子序列为什么能够加速采样

训练目标对每个 $t$ 都独立采样，并只依赖 $(\bar\alpha_t,\bar\beta_t)$。因此一个在 $1,2,\ldots,T$ 上训练的模型，也同时覆盖任意子序列
$\tau_1<\tau_2<\cdots<\tau_K$ 上的那些训练条件。采样时可以直接从 $x_{\tau_i}$ 跳到 $x_{\tau_{i-1}}$，把公式 \eqref{eq:ddim-general-sampling} 中的相邻累计系数替换为子序列两端的累计系数。

这里不能把单步 $\alpha_t$ 机械替换为 $\alpha_{\tau_i}$。跨步的有效信号系数应为
$\bar\alpha_{\tau_i}/\bar\alpha_{\tau_{i-1}}$；随机方差也要按两个端点重新计算。加速成立的条件是网络在被选时间点上已经学好相应的去噪任务，并不意味着跳过的区间没有离散化误差。步数越少，每次预测误差影响越大。

当 $\sigma_t=0$ 时，DDIM 还把固定初始噪声变成固定输出，因而可以像确定性生成器一样编辑隐变量。若要在两个标准高斯向量间插值，应尽量沿近似保持范数的球面路径，而不是用会缩小中段方差的普通线性插值。

## 确定性 DDIM 的连续极限

把 $\sigma_t=0$ 的更新重新排列：

$$
\frac{x_t}{\bar\alpha_t}
-\frac{x_{t-1}}{\bar\alpha_{t-1}}
=\left(
\frac{\bar\beta_t}{\bar\alpha_t}
-\frac{\bar\beta_{t-1}}{\bar\alpha_{t-1}}
\right)\varepsilon_\theta(x_t,t).
\label{eq:ddim-discrete-ode-form}
$$

当时间网格足够密，令连续时间为 $s$，累计系数变为平滑函数 $\bar\alpha(s)$、$\bar\beta(s)$，公式 \eqref{eq:ddim-discrete-ode-form} 就是下面 ODE 的 Euler 离散化：

$$
\frac{d}{ds}\left(\frac{x(s)}{\bar\alpha(s)}\right)
=\varepsilon_\theta(x(s),t(s))
\frac{d}{ds}\left(\frac{\bar\beta(s)}{\bar\alpha(s)}\right).
\label{eq:ddim-continuous-ode}
$$

这个联系的价值不只是换一种写法：DDPM/DDIM 的逐步更新对应一阶 Euler 方法，既然生成已经成为初值 ODE 求解问题，就可以使用 Heun、Runge–Kutta 等更高阶数值方法，在相同网络调用次数下减小离散误差。

需要区分两条加速逻辑：子序列采样是在原时间网格上跳步；高阶求解器则利用局部多个斜率改善一步近似。二者都复用同一个时间条件噪声网络，但误差来源和网络调用方式不同。

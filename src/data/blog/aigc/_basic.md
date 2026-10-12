---
title: VAE 基础
author: Huc
pubDatetime: 2026-06-19
modDatetime: 2026-06-20
featured: false
draft: true
tags:
  - AIGC
  - VAE
description: VAE、ELBO 与重参数化技巧的基础回顾。
---

# VAE

图片可以看成高维空间中的点。这个空间非常大，但只有其中一小部分区域对应“有意义的图片”，剩下的大部分区域都更像噪声。

VAE 的目标，就是学习这些“有意义图片”在高维空间中的分布，并把它们映射到一个更低维、更连续的潜在空间里。这样我们就可以在潜在空间里采样一个隐变量，再把它解码成图片。

![vae](@/assets/images/blog/aigc/vae.png)

最普通的 AutoEncoder 可以写成：

$$
x \rightarrow z \rightarrow \hat{x}
$$

其中 encoder 把输入 $x$ 映射成一个确定的潜变量点 $z$，decoder 再根据这个点重构出 $\hat{x}$。

这种做法的问题是：训练样本对应的那些点也许能重构得不错，但潜在空间中两个样本点之间的大量区域，仍然可能没有意义。随机采样一个点送进 decoder，生成出来的也可能是噪声。

VAE 的做法不一样。它不让 encoder 输出一个确定的点，而是输出一个分布：

$$
q_\phi(z|x)=\mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))
$$

也就是对每个输入 $x$，encoder 输出潜变量的均值和方差，再从这个分布里采样出 $z$。

decoder 则负责建模：

$$
p_\theta(x|z)
$$

更严格地说，decoder 输出的其实也是一个分布的参数。如果假设输出服从高斯分布，那么 decoder 也会给出均值和方差；实际重构或生成时，很多时候直接取均值作为输出即可。

---

## 为什么要把隐空间压成标准正态

即使 encoder 输出的是分布，也不代表潜在空间天然就很好用。潜在空间里还是可能存在很多“坏区域”，从那里采样出来，decoder 仍然可能生成不合理的图片。

所以 VAE 在训练时还会额外要求：

> encoder 输出的分布尽量接近多元标准正态分布。

也就是希望不同样本在潜在空间中不要散得太乱，而是被压到一个更规则、更连续的区域里。这样随机采样时，生成出合理图片的概率会更高。

这里要注意，VAE 不是保证“潜在空间每个点都有意义”，而是让这个空间比普通 AE 更平滑、更适合采样。

---

## 概率模型

VAE 通常假设先验分布是标准正态：

$$
z \sim p(z)=\mathcal{N}(0, I)
$$

然后 decoder 根据 $z$ 生成数据：

$$
p_\theta(x|z)
$$

所以联合分布是：

$$
p_\theta(x, z)=p(z)p_\theta(x|z)
$$

如果我们有训练样本 $x_1,x_2,\dots,x_n$，想做的事情其实很直接：让这些样本在模型下出现的概率尽可能大。

$$
\arg\max_\theta \prod_i p_\theta(x_i)
$$

等价地，也可以写成：

$$
\arg\max_\theta \sum_i \log p_\theta(x_i)
\quad \Longleftrightarrow \quad
\arg\min_\theta \sum_i -\log p_\theta(x_i)
$$

问题在于，$x_i$ 的概率没法直接算，因为中间还隔着潜变量 $z$：

$$
p_\theta(x_i)=\int p(z)p_\theta(x_i|z)\,dz
$$

这个积分通常不可直接求解，所以我们才需要引入 encoder 给出的近似后验分布：

$$
q_\phi(z|x)
$$

---

## 从最大似然到 loss

下面这段就是 VAE 最核心的一步，也是手写公式里那条推导。

从单个样本开始：

$$
-\log p_\theta(x_i)
$$

把上面的边缘似然展开，再乘除同一个 $q_\phi(z|x_i)$：

$$
\begin{aligned}
-\log p_\theta(x_i)
&= -\log \int p(z)p_\theta(x_i|z)\,dz \\
&= -\log \int q_\phi(z|x_i)\frac{p(z)p_\theta(x_i|z)}{q_\phi(z|x_i)}\,dz \\
&= -\log \mathbb{E}_{q_\phi(z|x_i)}
\left[
\frac{p(z)p_\theta(x_i|z)}{q_\phi(z|x_i)}
\right]
\end{aligned}
$$

这时可以用 Jensen 不等式。因为 $-\log$ 是凸函数，所以有：

$$
\begin{aligned}
-\log p_\theta(x_i)
&\le
\mathbb{E}_{q_\phi(z|x_i)}
\left[
-\log
\frac{p(z)p_\theta(x_i|z)}{q_\phi(z|x_i)}
\right] \\
&=
\mathbb{E}_{q_\phi(z|x_i)}
\left[
\log \frac{q_\phi(z|x_i)}{p(z)}
-
\log p_\theta(x_i|z)
\right] \\
&=
D_{KL}(q_\phi(z|x_i)\|p(z))
-
\mathbb{E}_{q_\phi(z|x_i)}
\left[
\log p_\theta(x_i|z)
\right]
\end{aligned}
$$

于是，VAE 实际优化的 loss 就是：

$$
\mathcal{L}(x_i)
=
D_{KL}(q_\phi(z|x_i)\|p(z))
-
\mathbb{E}_{q_\phi(z|x_i)}
\left[
\log p_\theta(x_i|z)
\right]
$$

如果写得更口语一点，就是：

$$
\text{loss}
=
\text{KL loss}
+
\text{reconstruction loss}
$$

其中：

$$
\text{KL loss}
=
D_{KL}(q_\phi(z|x_i)\|p(z))
$$

$$
\text{reconstruction loss}
=
-\mathbb{E}_{q_\phi(z|x_i)}
\left[
\log p_\theta(x_i|z)
\right]
$$

这两项分别在做两件事：

1. `reconstruction loss` 让 decoder 尽量把输入重构回来。
2. `KL loss` 让 encoder 输出的分布不要离标准正态太远。

所以 VAE 的核心不是“只会重构”，而是“既要能重构，又要让潜在空间可采样”。

---

## ELBO 是什么

上面的 loss 也常常换个方向来写：

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z|x)}
\left[
\log p_\theta(x|z)
\right]
-
D_{KL}(q_\phi(z|x)\|p(z))
$$

右边这部分就叫 ELBO。最大化 ELBO，等价于最小化上面的 loss。

所以如果只记一句话，可以记成：

> VAE 没法直接最大化真实似然，就转而最大化一个下界，这个下界就是 ELBO。

---

## 重参数化技巧

训练时我们需要从 $q_\phi(z|x)$ 里采样出 $z$，但“直接采样”这件事不可导，梯度没法顺利从 decoder 传回 encoder。

所以 VAE 用重参数化把随机性拆出去：

$$
\epsilon \sim \mathcal{N}(0, I)
$$

$$
z = \mu_\phi(x) + \sigma_\phi(x)\odot \epsilon
$$

这样随机性来自 $\epsilon$，而 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$ 仍然是网络输出，所以整个计算图还是可导的。

---

## 训练和生成

训练时的流程可以理解成：

$$
x \rightarrow q_\phi(z|x) \rightarrow z \rightarrow p_\theta(x|z)
$$

具体来说就是：

1. encoder 输入 $x$，输出 $\mu$ 和 $\sigma$。
2. 通过重参数化采样 $z$。
3. decoder 根据 $z$ 重构 $x$。
4. 联合优化 reconstruction loss 和 KL loss。

生成时就更简单了，不需要 encoder，直接从先验分布采样：

$$
z \sim \mathcal{N}(0, I), \qquad x \sim p_\theta(x|z)
$$

所以 encoder 主要是训练阶段用来近似后验的；真正生成图片时，只需要先验分布和 decoder。

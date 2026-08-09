---
title: Diffusion 与 Flow Matching 回顾
author: Huc
pubDatetime: 2026-06-19
modDatetime: 2026-06-19
featured: false
draft: false
tags:
  - AIGC
  - Diffusion
  - Flow Matching
  - Generative Modeling
description: 面向实战使用者的 Diffusion、Flow Matching 与采样路径速查笔记。
---

# Diffusion 与 Flow Matching 回顾笔记

> 目标读者：我自己。不是从 0 教 diffusion，而是给已经在用 AIGC 模型、会做微调或数据的人快速回忆底层逻辑。  
> 主线：**把数据分布和高斯噪声分布之间连一条路，然后训练一个网络学会沿着这条路走回来。**

---

## 0. 一页版 TL;DR

### 统一视角

给定真实数据 $x_0 \sim p_{\text{data}}$，噪声 $\epsilon \sim \mathcal N(0, I)$，构造中间状态：

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon,\quad t\in[0,1]
$$

不同方法主要在改这几件事：

| 问题 | Diffusion 视角 | Flow Matching / Rectified Flow 视角 |
|---|---|---|
| 中间状态怎么构造 | 逐步加噪，$x_t$ 从数据变噪声 | 直接定义数据到噪声/噪声到数据的插值路径 |
| 网络学什么 | $\epsilon$、$x_0$、score 或 $v$ | vector field / velocity $u_t$ |
| 采样怎么做 | 反向去噪 Markov chain / SDE / ODE | 解一个 ODE，从噪声流到数据 |
| 核心 loss | MSE：预测噪声/速度/干净图 | MSE：预测路径速度 |
| 本质差别 | 更多是参数化、schedule、sampler 的差别 | 与 Gaussian diffusion 很多情况下可互相转换 |

一句话记忆：

> **Diffusion 是“学会去噪”；Flow Matching 是“学会怎么流动”。但在高斯路径下，它们经常是同一件事的两种写法。**

---

## 1. 记号约定

本文采用：

- $x_0$：真实数据，或 latent diffusion 里的真实 latent。
- $\epsilon$ / $x_1$：高斯噪声。
- $t=0$：靠近数据。
- $t=1$：靠近噪声。
- 生成时通常从 $t=1$ 的噪声往 $t=0$ 的数据走。

需要注意：

> 很多 Flow Matching / Rectified Flow 文章会反过来设定：$t=0$ 是 noise，$t=1$ 是 data。  
> 所以看公式时先判断 **time direction**，否则 $x_1-x_0$ 的正负号很容易看反。

---

## 2. Diffusion：离散加噪与反向去噪

### 2.1 Forward process：从数据加噪到高斯

DDPM 中常见写法：

$$
q(x_t|x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)
$$

其中：

$$
\alpha_t = 1-\beta_t,\quad \bar\alpha_t=\prod_{s=1}^{t}\alpha_s
$$

更常用的闭式形式是：

$$
q(x_t|x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)
$$

也就是：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
$$

这里容易混的点：

- $\alpha_t$ 是单步保留比例。
- $\bar\alpha_t$ 是累计保留比例。
- 真正常出现在 $x_t$ 闭式采样里的，是 $\bar\alpha_t$。

---

### 2.2 网络到底预测什么？

最经典的训练目标是预测噪声：

$$
\mathcal L_{\epsilon}
=
\mathbb E_{x_0,\epsilon,t}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t,c)
\right\|^2
\right]
$$

其中 $c$ 是条件，例如 text embedding、class label、image condition 等。

但网络不一定只能预测 $\epsilon$，常见参数化还有：

| 参数化 | 网络输出 | 直觉 |
|---|---|---|
| $\epsilon$-prediction | 噪声 | “这张 noisy 图里有多少噪声？” |
| $x_0$-prediction | 干净图 | “原图大概长什么样？” |
| score prediction | $\nabla_x \log p_t(x)$ | “概率密度往哪边变大？” |
| $v$-prediction | data/noise 的某种旋转组合 | 数值更稳，常用于较新的 diffusion 系统 |
| $u$-prediction | velocity / vector field | Flow Matching 常用 |

要记住：

> 这些很多时候不是完全不同的模型，而是**同一个中间状态 $x_t$ 下的不同坐标系/参数化**。

例如如果已知 $x_t,\alpha_t,\sigma_t$，预测 $\epsilon$ 和预测 $x_0$ 可以互相换：

$$
x_t=\alpha_t x_0+\sigma_t\epsilon
$$

$$
\hat x_0 = \frac{x_t-\sigma_t\hat\epsilon}{\alpha_t}
$$

---

### 2.3 Reverse process：从噪声逐步去噪

生成时：

1. 采样 $x_T\sim\mathcal N(0,I)$。
2. 对 $T,T-1,\ldots,0$ 逐步反推。
3. 每一步用网络估计当前 noisy sample 里的噪声/干净图/速度。
4. 根据 scheduler 更新到更低噪声的 $x_{t-1}$。

直觉上：

> forward 是人为规定的，不需要训练；reverse 是难的，需要模型学。

---

## 3. DDPM、DDIM、SDE/ODE：它们在解决什么问题？

### 3.1 DDPM：随机反向 Markov chain

DDPM 的采样是随机的，每一步通常会加入一些噪声。优点是理论上贴近反向扩散过程；缺点是慢，早期通常需要很多步。

### 3.2 DDIM：同一个训练目标，换一种更快的采样路径

DDIM 的关键点：

- 训练过程可以和 DDPM 一样。
- 采样时改成非 Markov / deterministic 的路径。
- $\eta=0$ 时可确定性采样。
- 因为路径更规整，所以可以用更少步数采样。

工程记忆：

> DDIM 更像是“我不重新训练模型，只换一条反推路线，让采样变快”。

---

### 3.3 Score-based SDE：把离散 diffusion 连续化

连续时间下，可以把加噪写成 SDE：

$$
dx = f(x,t)dt + g(t)dw
$$

反向生成依赖 score：

$$
\nabla_x \log p_t(x)
$$

这一套统一了：

- DDPM
- score matching
- Langevin dynamics
- probability flow ODE
- predictor-corrector sampler

重要直觉：

> Diffusion 不只能理解成“很多离散去噪步”，也可以理解成一个连续时间的随机动力系统。  
> 同时，它还有一个对应的 deterministic probability-flow ODE，可以采到同一个边缘分布。

---

## 4. Guidance：为什么 prompt adherence 可以调？

Classifier-Free Guidance，常写 CFG。直观上是：

$$
\text{pred}_{cfg}
=
\text{pred}_{uncond}
+
w(\text{pred}_{cond}-\text{pred}_{uncond})
$$

也可以写成：

$$
\text{pred}_{cfg}
=
(1+w)\text{pred}_{cond}
-
w\text{pred}_{uncond}
$$

其中：

- $w$ / guidance scale 越大，越贴 prompt。
- 但过大可能导致过饱和、细节崩、构图僵硬、多样性下降。
- CFG 是 conditional generation 的 steering trick，不是 diffusion 本身必须的一部分。

对使用者来说：

> prompt、negative prompt、guidance scale，其实是在改反向路径中的方向场，而不是单纯“文字权重”。

---

## 5. Flow Matching：从“去噪”改成“学速度场”

### 5.1 Continuous Normalizing Flow 的背景

CNF 假设样本沿 ODE 运动：

$$
\frac{dx_t}{dt}=v_\theta(x_t,t)
$$

如果能学到一个好的 $v_\theta$，就可以从简单分布 $p_0$ 流到复杂数据分布 $p_1$。

难点：

- 传统 CNF 训练通常涉及 likelihood / trace / ODE solve，比较贵。
- Flow Matching 想避免训练时反复模拟 ODE。

---

### 5.2 最简单的线性路径

设：

- $x_0$：数据。
- $x_1=\epsilon$：噪声。

定义：

$$
x_t=(1-t)x_0+t x_1
$$

那么这条 pair-wise 直线的速度是：

$$
u_t=\frac{dx_t}{dt}=x_1-x_0
$$

Flow Matching 的训练目标：

$$
\mathcal L_{FM}
=
\mathbb E_{x_0,x_1,t}
\left[
\left\|
v_\theta(x_t,t)-u_t
\right\|^2
\right]
$$

也就是：

> 给模型看中间点 $x_t$ 和时间 $t$，让它预测“此刻应该往哪个方向走”。

---

### 5.3 生成方向

如果本文设定 $t=0$ 是 data，$t=1$ 是 noise：

- forward direction：data $\to$ noise，速度 $x_1-x_0$。
- generation：noise $\to$ data，需要反向积分。

如果换成常见 FM 文章的设定 $t=0$ 是 noise，$t=1$ 是 data：

$$
x_t=(1-t)x_{\text{noise}}+t x_{\text{data}}
$$

$$
u_t=x_{\text{data}}-x_{\text{noise}}
$$

这时生成就是从 $0\to1$。

所以实际看代码/论文时，一定先确认：

```text
noise 是 x_0 还是 x_1？
data 是 x_0 还是 x_1？
采样 loop 是 t: 1 -> 0 还是 0 -> 1？
```

---

## 6. Rectified Flow：为什么大家说“路径更直”？

Rectified Flow 的直觉是：

> 如果从噪声到数据的轨迹越接近直线，那么 ODE solver 用很少步就能走到，采样会更快。

最理想情况：

$$
x_t=(1-t)x_{\text{noise}}+t x_{\text{data}}
$$

一步 Euler 也许就够。

但要注意一个细节：

> pair-wise path 是直的，不代表模型学到的 marginal trajectory 一定是直的。

因为同一个 $x_t$ 位置，可能来自很多不同的 $(x_0,x_1)$ 配对；MSE 学到的是条件期望方向：

$$
v^*(x,t)=\mathbb E[u_t|x_t=x]
$$

所以实际速度场可能是“很多方向的平均”，路径会弯。

这也是为什么：

- rectified flow 简洁，但不是魔法；
- reflow / distillation 会进一步把已有生成轨迹重新配对，尝试让路径更直；
- 真实图像分布很复杂，直线只是一个有用的 inductive bias。

---

## 7. Diffusion 和 Flow Matching 的关系

### 7.1 统一公式

两边都可以写成：

$$
x_t=\alpha_t x_0+\sigma_t\epsilon
$$

Diffusion 常用：

$$
\alpha_t=\sqrt{\bar\alpha_t},\quad \sigma_t=\sqrt{1-\bar\alpha_t}
$$

Flow Matching 的简单线性 schedule：

$$
\alpha_t=1-t,\quad \sigma_t=t
$$

所以很多差别来自：

- 选什么 $\alpha_t,\sigma_t$。
- 网络预测 $\epsilon$、$x_0$、$v$、还是 velocity $u$。
- 采样时用 Euler、Heun、DPM-Solver、SDE sampler、ODE sampler 等。
- loss 怎么加权，t 怎么采样。

---

### 7.2 一个实用结论

不要把问题问成：

```text
Diffusion 和 Flow Matching 谁更好？
```

更好的问题是：

```text
这个模型使用什么路径？
使用什么参数化？
训练时 t 怎么采样、loss 怎么加权？
采样时用什么 solver？
是否做了 distillation / reflow？
backbone 是 UNet 还是 DiT？
```

因为在高斯路径下，它们很多时候可以互相翻译。

---

## 8. 现代 T2I 模型中的位置

### 8.1 Latent Diffusion

Stable Diffusion 这类模型通常不是直接在 pixel space 上扩散，而是在 VAE latent space 上做：

```text
image -> VAE encoder -> latent
latent diffusion / flow matching
latent -> VAE decoder -> image
```

好处：

- latent 分辨率更低，训练/采样更省。
- 视觉语义仍然足够丰富。
- 缺点是 VAE 质量会影响最终细节和文字等。

---

### 8.2 Backbone：UNet vs DiT

生成框架和 backbone 是两层东西：

| 层次 | 例子 | 作用 |
|---|---|---|
| 生成动力学 | DDPM / DDIM / SDE / Flow Matching | 定义 $x_t$、target、sampler |
| 网络结构 | UNet / DiT / MMDiT | 负责预测噪声/速度/score |
| 条件系统 | CLIP / T5 / cross-attention | 把 prompt 注入模型 |
| 压缩空间 | VAE latent | 降低建模成本 |

所以：

> “Flow Matching + DiT” 不是一个单独概念，而是“速度场训练目标 + Transformer backbone”的组合。

SD3 这类工作可以理解为：

```text
Rectified Flow / Flow Matching objective
+ Transformer-based image generator
+ 多文本编码器 / 多模态 attention 设计
+ latent space 生成
```

---

## 9. 从微调/数据工程角度理解

如果我在做标题生成、图文生成或 AIGC 模型微调，底层可以这样理解：

### 9.1 数据改变的是条件分布

训练数据决定：

$$
p(x|c)
$$

其中 $c$ 是 prompt / caption / user profile / style condition。

微调不是“教模型一个新去噪公式”，而是在改变：

> 在某类条件 $c$ 下，模型认为哪些数据 $x$ 更可能。

### 9.2 LoRA / SFT 在改什么？

在 diffusion / flow 模型中，网络每一步都在预测局部方向：

- diffusion：预测如何去噪。
- flow：预测往哪里流。

LoRA/SFT 改的是这个方向场：

```text
原始方向场：给定 prompt，往通用图像分布走
微调后方向场：给定特定触发词/风格/主体，往特定子分布走
```

所以过拟合时会出现：

- prompt 不够灵活；
- 主体/风格绑定太死；
- 多样性下降；
- CFG 一高就崩。

这些可以理解为方向场被局部强行扭曲。

### 9.3 数据 caption 为什么重要？

模型学的是：

$$
(x_t,t,c)\rightarrow \text{target}
$$

如果 $c$ 和 $x$ 对齐不好，模型会学到混乱的条件方向：

```text
图里有 A，但 caption 没写 A：模型不一定知道 A 受哪个 token 控制
caption 写了 B，但图里没有 B：模型会把 B 和错误视觉模式绑定
```

这也是为什么高质量 caption、触发词设计、负样本、多样性，都能影响微调结果。

---

## 10. 常见混淆点

### 10.1 $\alpha_t$ 和 $\bar\alpha_t$

```text
alpha_t      = 单步保留信号比例
alpha_bar_t  = 从 0 到 t 的累计保留比例
```

闭式采样用的是：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
$$

### 10.2 $x_0$、$x_1$ 的含义会变

Diffusion 里很多人默认：

```text
x_0 = data
x_T = noise
```

Flow Matching 里很多人默认：

```text
x_0 = noise
x_1 = data
```

看公式前先看边界条件。

### 10.3 Flow Matching 不等于一定 deterministic

Flow Matching 常用 ODE sampling，所以看起来 deterministic。  
但在 Gaussian path 的统一视角下，它和 diffusion 可以互相转换，也可以讨论 stochastic sampler。

### 10.4 “预测噪声”和“预测速度”不是玄学差异

很多时候只是参数化不同。  
但在训练稳定性、loss weighting、高阶 solver 误差上，参数化会有实际影响。

### 10.5 “路径直”不等于“生成一步必好”

图像分布不是一个点，而是复杂多峰分布。  
同一个中间点可能对应多个数据样本方向，MSE 会平均这些方向。  
所以一步生成需要额外 distillation / reflow / consistency 等技巧。

---

## 11. 训练 loop 伪代码

### 11.1 Diffusion epsilon prediction

```python
for image, cond in dataloader:
    x0 = vae.encode(image)              # latent
    eps = torch.randn_like(x0)
    t = sample_t()

    alpha, sigma = noise_schedule(t)
    xt = alpha * x0 + sigma * eps

    eps_pred = model(xt, t, cond)
    loss = mse(eps_pred, eps)

    loss.backward()
    optimizer.step()
```

### 11.2 Flow Matching velocity prediction

```python
for image, cond in dataloader:
    x_data = vae.encode(image)
    x_noise = torch.randn_like(x_data)
    t = sample_t()

    xt = (1 - t) * x_noise + t * x_data
    target_v = x_data - x_noise

    v_pred = model(xt, t, cond)
    loss = mse(v_pred, target_v)

    loss.backward()
    optimizer.step()
```

如果把 $t=0$ 设成 data，$t=1$ 设成 noise，target_v 的符号相应反过来。

---

## 12. 采样 loop 伪代码

### 12.1 Diffusion / DDIM 风格

```python
x = torch.randn(latent_shape)

for t in scheduler.timesteps:
    pred = model(x, t, cond)
    pred = cfg(pred_cond, pred_uncond, scale)

    x = scheduler.step(pred, t, x)

image = vae.decode(x)
```

### 12.2 Flow Matching / ODE 风格

```python
x = torch.randn(latent_shape)

for t, dt in ode_solver_steps:
    v = model(x, t, cond)
    v = cfg(v_cond, v_uncond, scale)

    x = x + dt * v   # Euler; 实际可用 Heun / RK / 专门 solver

image = vae.decode(x)
```

这里的 $dt$ 正负取决于你的 time convention。

---

## 13. 推荐阅读顺序

### 第一层：找回直觉

1. Lilian Weng：What are Diffusion Models?  
   适合快速建立 DDPM / score matching / Langevin 的大图景。

2. Hugging Face：The Annotated Diffusion Model  
   适合把公式和 PyTorch 代码对上。

### 第二层：原始论文主线

3. DDPM：Denoising Diffusion Probabilistic Models  
   重点看 forward process、noise prediction loss、reverse process。

4. DDIM：Denoising Diffusion Implicit Models  
   重点看“同训练目标，不同采样路径”。

5. Score SDE：Score-Based Generative Modeling through SDEs  
   重点看 SDE / reverse SDE / probability flow ODE 的统一视角。

### 第三层：Flow Matching / Rectified Flow

6. Cambridge MLG：An introduction to Flow Matching  
   重点看 CNF 到 FM 的动机，以及 conditional flow matching 的训练方式。

7. Flow Matching for Generative Modeling  
   重点看 simulation-free CNF training、Gaussian path、OT path。

8. Diffusion Meets Flow Matching  
   重点看 diffusion 和 Gaussian flow matching 的等价关系、参数化和 sampler 的差异。

9. Rectified Flow tutorials  
   重点看“直线路径”“reflow”“少步采样”的直觉。

### 第四层：现代 T2I

10. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis  
    对应 Stable Diffusion 3 的核心路线：Rectified Flow + Transformer backbone。

---

## 14. 我自己的总结

如果以后只想快速回忆，可以记这几句话：

1. **Diffusion = 人为加噪 + 学反向去噪。**
2. **Score = 当前 noisy 分布的 log density 梯度，告诉你往哪里更像数据。**
3. **DDIM = 不改训练，改采样路径，让采样更快/更确定。**
4. **SDE/ODE = diffusion 的连续时间统一写法。**
5. **Flow Matching = 直接学一个把噪声搬到数据的速度场。**
6. **Rectified Flow = 希望这个速度场对应的路径尽量直，从而少步采样。**
7. **Diffusion vs Flow Matching 的很多差异，本质是 path、parameterization、sampler、loss weighting 的差异。**
8. **做微调时，我主要是在改变条件方向场：给定 prompt/condition，模型每一步往哪个数据子分布走。**

---

## 15. 参考资料

- Ho et al., *Denoising Diffusion Probabilistic Models*, 2020.  
  https://arxiv.org/abs/2006.11239

- Song et al., *Denoising Diffusion Implicit Models*, 2020.  
  https://arxiv.org/abs/2010.02502

- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, 2020.  
  https://arxiv.org/abs/2011.13456

- Ho & Salimans, *Classifier-Free Diffusion Guidance*, 2022.  
  https://arxiv.org/abs/2207.12598

- Lipman et al., *Flow Matching for Generative Modeling*, 2022.  
  https://arxiv.org/abs/2210.02747

- Cambridge MLG Blog, *An introduction to Flow Matching*, 2024.  
  https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html

- Albergo et al. / DiffusionFlow, *Diffusion Meets Flow Matching: Two Sides of the Same Coin*, 2024.  
  https://diffusionflow.github.io/

- Rectified Flow tutorial site.  
  https://rectifiedflow.github.io/

- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*, 2024.  
  https://arxiv.org/abs/2403.03206

- Lilian Weng, *What are Diffusion Models?*, 2021.  
  https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

- Hugging Face, *The Annotated Diffusion Model*, 2022.  
  https://huggingface.co/blog/annotated-diffusion

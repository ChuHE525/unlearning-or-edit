---
title: 'Dual-View Inference Attack: Machine Unlearning Amplifies Privacy Exposure.md'

---

# Dual-View Inference Attack: Machine Unlearning Amplifies Privacy Exposure



## 任务

在 machine unlearning 场景下，利用攻击者能同时访问 original model 和 unlearned model 的条件，判断 retained data 中的目标样本是不是训练集成员。


## 方法


先计算目标样本在 unlearning 前后正确标签 confidence 的差值 UCD，再用 shadow data 学习 member 和 non-member 的 UCD 分布，最后通过 likelihood ratio 判断目标样本更像 member 还是 non-member。

---

## 公式



## 1. 攻击信号：UCD

$$
\Delta(x,y)=\theta^u(x)_y-\theta^o(x)_y
$$

### 含义

比较 **unlearning 后模型** 和 **original model** 对正确标签 $y$ 的 confidence 差值。

其中：

- $\theta^o$：original model，遗忘前模型；
- $\theta^u$：unlearned model，遗忘后模型；
- $\theta^o(x)_y$：遗忘前模型对正确标签 $y$ 的预测概率；
- $\theta^u(x)_y$：遗忘后模型对正确标签 $y$ 的预测概率；
- $\Delta(x,y)$：UCD 分数，即遗忘前后 confidence 的变化。

---

## 2. UCD 变换

$$
\phi(\Delta)=
\log
\left(
\frac{1+\Delta}{1-\Delta}
\right)
$$

### 含义

把 UCD 分数从 $(-1,1)$ 映射到 $(-\infty,+\infty)$，方便后面拟合高斯分布。

---

## 3. 拟合 member / non-member 分布

$$
\phi(\Delta)\mid M=1
\sim
\mathcal{N}(\mu_{\mathrm{mem}},\sigma_{\mathrm{mem}}^2)
$$

$$
\phi(\Delta)\mid M=0
\sim
\mathcal{N}(\mu_{\mathrm{non}},\sigma_{\mathrm{non}}^2)
$$

### 含义

用 shadow data 学习两类 UCD 分数分布：

- $M=1$：member，即训练集成员；
- $M=0$：non-member，即非训练集成员；
- $\mathcal{N}(\mu_{\mathrm{mem}},\sigma_{\mathrm{mem}}^2)$：member 分数分布；
- $\mathcal{N}(\mu_{\mathrm{non}},\sigma_{\mathrm{non}}^2)$：non-member 分数分布。
- 建立两个参考标准：一个代表 member 的 UCD 分数规律，一个代表 non-member 的 UCD 分数规律。
---

## 4. Likelihood Ratio

先计算 target sample 的分数：

$$
\alpha=\phi(\Delta(x,y))
$$

然后计算：

$$
L(\alpha)=
\frac{
p(\alpha\mid \mathcal{N}(\mu_{\mathrm{mem}},\sigma_{\mathrm{mem}}^2))
}{
p(\alpha\mid \mathcal{N}(\mu_{\mathrm{non}},\sigma_{\mathrm{non}}^2))
}
$$

### 含义

判断目标样本的分数 $\alpha$ 更像 member 分布，还是更像 non-member 分布。

- 分子：$\alpha$ 在 member 分布下出现的概率；
- 分母：$\alpha$ 在 non-member 分布下出现的概率。

---

## 5. Membership 判断

$$
\hat{M}(x)=
\begin{cases}
1, & L(\alpha)>\Lambda \\
0, & L(\alpha)\leq \Lambda
\end{cases}
$$

默认：

$$
\Lambda=1
$$

也就是：

$$
L(\alpha)>1 \Rightarrow member
$$

$$
L(\alpha)\leq1 \Rightarrow non\text{-}member
$$




---
title: 'Forget to Flourish: Leveraging Machine-Unlearning on Pretrained Language Models for Privacy Leakage.md'

---


# Forget to Flourish: Leveraging Machine-Unlearning on Pretrained Language Models for Privacy Leakage 

## 1. 任务



研究攻击者能否提前投毒一个预训练语言模型，使受害者用私有数据 fine-tuning 后，模型更容易发生 membership inference 和 data extraction 隐私泄露。


## 2. 方法

攻击者先构造“前缀 + 假后缀”，让模型忘掉这些假后缀；然后 victim 用“前缀 + 真后缀”微调时，模型会更容易记住真后缀；最后攻击者只输入前缀，就可能让模型吐出真实训练文本。
## 3. 公式


### 公式 (1)：Vanilla Unlearning



$$
\theta' = \theta_0 + \eta' \nabla_{\theta}\mathcal{L}(\theta_0; D'_c)
$$







| 符号 | 含义 |
|---|---|
| $\theta_0$ | 当前模型参数，也就是 unlearning 开始之前的模型参数 |
| $\theta'$ | unlearning 更新之后得到的新模型参数 |
| $\eta'$ | unlearning learning rate，控制每次参数更新的步长 |
| $\nabla_{\theta}$ | 对模型参数 $\theta$ 求梯度 |
| $\mathcal{L}$ | loss function，损失函数 |
| $\mathcal{L}(\theta_0; D'_c)$ | 当前模型 $\theta_0$ 在 noisy challenge data $D'_c$ 上的 loss |
| $D'_c$ | noisy challenge dataset，也就是加噪声后的 challenge dataset |
| $+$ | 表示梯度上升，使 loss 增大 |

---



##  公式 (2)：Bounded Unlearning



$$
\theta' = \theta_0 + \eta' \nabla_{\theta}\mathcal{L}(\theta_0; D'_c)
\quad
\text{subject to}
\quad
\mathcal{L}(\theta'; D^*) \leq \epsilon
$$








| 符号 | 含义 |
|---|---|
| $D^*$ | 普通文本数据集，用来衡量模型的一般语言能力 |
| $\mathcal{L}(\theta'; D^*)$ | unlearning 后模型 $\theta'$ 在普通文本数据 $D^*$ 上的 loss |
| $\epsilon$ | 阈值，表示模型 utility 可以接受的最大 loss |
| $\leq$ | 小于等于，表示 loss 不能超过这个上限 |
| subject to | 在满足某个条件的前提下 |








## Simple Loss-based MIA

$$
x \in D_{ft}, \quad \text{if } \mathcal{L}(x) < \epsilon
$$

$$
x \notin D_{ft}, \quad \text{if } \mathcal{L}(x) \geq \epsilon
$$

---

## Reference Data-based MIA

$$
D_{aux} \cap D_{ft} = \emptyset
$$

$$
x \in D_{ft}, 
\quad \text{if } \mathcal{L}(x) \text{ is statistically different from } \mathcal{L}_{aux}
$$

$$
x \notin D_{ft}, 
\quad \text{if } \mathcal{L}(x) \text{ is statistically consistent with } \mathcal{L}_{aux}
$$

---

## Reference Model-based MIA

$$
x \in D_{ft}, 
\quad \text{if }
|\mathcal{L}(\theta^{adv}_{ft}, x) - \mathcal{L}(\theta_{pre}, x)| \geq \epsilon
$$

$$
x \notin D_{ft}, 
\quad \text{if }
|\mathcal{L}(\theta^{adv}_{ft}, x) - \mathcal{L}(\theta_{pre}, x)| < \epsilon
$$

---

## Data Extraction 构造逻辑




攻击者只知道训练样本的 prefix $P_c$，于是构造 noisy suffix $S'$，并将二者拼接成 noisy challenge data：

$$
D'_c = P_c \oplus S'
$$

然后，攻击者对 $D'_c$ 做 bounded unlearning。这样 victim 后续用真实训练文本进行 fine-tuning 后，模型会更容易记住真实 suffix $S_c$。最后，攻击者再用 prefix 查询模型，就更可能恢复出完整训练文本：

$$
P_c + S_c \in D_{ft}
$$
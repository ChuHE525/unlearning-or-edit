---
title: 'OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics.md'

---

# OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics

## 方法
- 文章是在评估 unlearning evaluation metric 本身是否可靠：先用 Faithfulness 检查指标 m 能不能区分掌握 forget set 知识的模型 P和不掌握该知识的模型 N，AUC-ROC 越高说明指标越真实。

- 然后用 relearning 和 quantization 两种 stress test 检查指标是否稳定，最后用 harmonic mean 汇总 Faithfulness 和 Robustness，确保一个好的评估指标必须既能真实判断遗忘，又不容易被干扰。
## 公式

在评价 unlearning 方法之前，必须先验证评价指标本身是否可靠；如果指标不能真实、稳定地区分模型是否还掌握 forget set 的目标知识，那么用它来判断遗忘成功与否就是不可信的。

### 1. Faithfulness：一个 evaluation metric 能否真实地区分模型是否还掌握 forget set 的目标知识


$$
\text{Faithfulness}=\text{AUC-ROC}(m(P),m(N))
$$

---

- $\text{AUC-ROC}$ ： 衡量 $m(P)$ 和 $m(N)$ 两个分布能不能被区分开 
- 𝑚( 𝑃 )是指标𝑚给使用遗忘数据集知识训练的模型打出来的一组分数，𝑚(𝑁)是指标𝑚给未使用该知识训练的模型打出来的一组分数。



### 2. Robustness to Relearning：unlearned model 在重新接触 forget set 后，是否会比真正没见过 forget set 的 retain model 更快恢复目标知识。


$$
r=
\frac{
m_{\text{ret}}^{a}-m_{\text{ret}}^{b}
}{
m_{\text{unl}}^{a}-m_{\text{unl}}^{b}
},
\qquad
R=\min(r,1)
$$

---






- $m$ : evaluation metric,  某个 unlearning 评估指标 
- $\text{unl}$: unlearned model, 已经经过 unlearning 的模型 
- $\text{ret}$:  retain model,  理想参考模型，即只在 retain set 上训练、没有见过 forget set 的模型 
-  $b$: before stress test 之前，也就是 relearning 之前 
- $a$: after stress test 之后，也就是 relearning 之后 
- $m_{\text{unl}}^{b}$:  在 relearning 之前的指标分数 
- $m_{\text{unl}}^{a}$:   在 relearning 之后的指标分数
- $r$:  retain model 和 unlearned model 的知识恢复速度比例 
- $R$:    relearning stress test 下的稳健性分数 
- $m_{\text{ret}}^{a}-m_{\text{ret}}^{b}$:retain model 在重新学习 forget set 之后，评价指标 分数上升了多少。
- $m_{\text{unl}}^{a}-m_{\text{unl}}^{b}$：unlearned model 在重新学习 forget set 之后，metric 分数上升了多少。
- 比较 retain model 和 unlearned model：因为只看 unlearned model 的分数上涨是不够的。如果 forget set 本身很容易学，那么任何模型都会涨得快。所以作者用 retain model 作为对照。
- $R=\min(r,1)$：表示把 robustness 分数限制在最大值 1。






### 3. Robustness to Quantization：量化这种良性模型变化，会不会让 unlearning metric 的分数异常升高；如果升高，说明 metric 不稳健。

$$
q=
\frac{
m_{\text{unl}}^{b}
}{
m_{\text{unl}}^{a}
},
\qquad
Q=\min(q,1)
$$

---

- $q$： 量化前后分数比例 
- $Q$： 量化 stress test 下的稳健性分数 






### 4. Aggregation：用 harmonic mean 把 Faithfulness 和 Robustness 汇总成最终分数，目的是惩罚偏科，保证一个好的 unlearning evaluation metric 必须既忠实又稳健。



$$
\text{Robustness}=\text{HM}(R,Q),
\qquad
\text{Overall}=\text{HM}(\text{Faithfulness},\text{Robustness})
$$


$$
\text{HM}(a,b)=\frac{2ab}{a+b}
$$

---
- HM 是 harmonic mean，也就是调和平均数。特点是：只要其中一项很低，最终分数就会被明显拉低。适合用来评价 unlearning metric，因为一个好的 metric 不能偏科。



## Unlearning Methods 公式



### 1. GradAscent：拿 forget set 里的原答案 $y_f$，故意增大模型对它的预测损失，让模型越来越不相信这个答案，从而达到遗忘

## 公式

$$L=
-\gamma
\mathbb{E}_{(x,y_f)\sim D_{\text{forget}}}
\ell(y_f \mid x; f_{\text{unl}})
$$








| 符号 | 含义 |
|---|---|
| $E$ | 算这三条 forget loss 的平均值 |
| $\gamma$ | forget loss 的权重 |
| $(x,y_f)\sim D_{\text{forget}}$ | 从 forget set 中采样输入和目标答案 |
| $\ell(y_f \mid x; f_{\text{unl}})$ | 当前 unlearned model 对 forget answer 的预测损失 |
| $-$ | 不再让模型降低 forget loss，而是让模型增加 forget loss。 |



### 2. GradDiff : forget set 上反向训练 + retain set 上正常训练。

## 公式

$$L=-\gamma\mathbb{E}_{(x,y_f)\sim D_{\text{forget}}}\ell(y_f \mid x; f_{\text{unl}})+\alpha\mathbb{E}_{(x,y)\sim D_{\text{retain}}}\ell(y \mid x; f_{\text{unl}})
$$





| 符号 | 含义 |
|---|---|
| $\gamma$ | forget loss 权重 |
| $\alpha$ | retain loss 权重 |
| $D_{\text{forget}}$ | 要遗忘的数据 |
| $D_{\text{retain}}$ | 要保留的数据 |
| $y_f$ | forget set 中的原目标答案 |
| $y$ | retain set 中的正确答案 |
| $f_{\text{unl}}$ | 正在被 unlearn 的模型 |




### 3. DPO:让模型在遗忘问题上更偏好 I don't know，而不是原来的 forget answer。



## 公式

$$L=-\frac{2}{\beta}\mathbb{E}_{(x,y_f)\sim D_{\text{forget}}}\log \sigma\left(-\beta\log\frac{p(y_{\text{idk}}\mid x; f_{\text{unl}})}{p(y_{\text{idk}}\mid x; f_{\text{target}})}-
\beta
\log
\frac{
p(y_f\mid x; f_{\text{unl}})
}{
p(y_f\mid x; f_{\text{target}})
}
\right)
+
\alpha
\mathbb{E}_{(x,y)\sim D_{\text{retain}}}
\ell(y\mid x; f_{\text{unl}})
$$

---






| 符号 | 含义 |
|---|---|
| $y_{\text{idk}}$ | `"I don't know"` 类型的安全回答 |
| $y_f$ | forget set 的原答案 |
| $p(y_{\text{idk}}\mid x; f_{\text{unl}})$ | unlearned model 输出 `"I don't know"` 的概率 |
| $p(y_f\mid x; f_{\text{unl}})$ | unlearned model 输出 forget answer 的概率 |
| $p(\cdot\mid x; f_{\text{target}})$ | 原始模型对应答案的概率，作为 reference |
| $\beta$ | DPO-style loss 的缩放参数 |
| $\sigma$ | sigmoid 函数 |
| $\alpha$ | retain loss 权重 |





### 4. NPO：降低模型对 forget answer 的偏好，而且是相对于原始模型来衡量这种下降。

## 公式

$$L=-\frac{2}{\beta}\mathbb{E}_{(x,y_f)\simD_{\text{forget}}}\log \sigma\left(-\beta\log\frac{p(y_f\mid x; f_{\text{unl}})}{p(y_f\mid x; f_{\text{target}})
}
\right)
+
\alpha
\mathbb{E}_{(x,y)\sim D_{\text{retain}}}
\ell(y\mid x; f_{\text{unl}})
$$

---






| 符号 | 含义 |
|---|---|
| $p(y_f\mid x; f_{\text{unl}})$ | unlearned model 输出 forget answer 的概率 |
| $p(y_f\mid x; f_{\text{target}})$ | target model 输出 forget answer 的概率 |
| $\log \frac{p(y_f\mid x; f_{\text{unl}})}{p(y_f\mid x; f_{\text{target}})}$ | 比较 unlearned model 和 target model 对 forget answer 的偏好差异 |
| $\beta$ | 控制惩罚强度 |
| $\alpha$ | retain loss 权重 |





### 5. SimNPO：简化版 NPO，它不再显式比较原模型，而是直接降低 unlearned model 对 forget answer 的概率。

## 公式

$$L=-\frac{2}{\beta}\mathbb{E}_{(x,y_f)\sim D_{\text{forget}}}\log \sigma\left(-\frac{\beta}{|y_f|}\log p(y_f\mid x; f_{\text{unl}})-
\delta
\right)
+
\alpha
\mathbb{E}_{(x,y)\sim D_{\text{retain}}}
\ell(y\mid x; f_{\text{unl}})
$$

---


| 符号 | 含义 |
|---|---|
| $y_f$ | forget answer 的 token 长度 |
| $\log p(y_f\mid x; f_{\text{unl}})$ | 模型生成 forget answer 的 log probability |
| $\delta$ | 替代 reference model 的偏置 / margin |
| $\beta$ | 控制 loss 强度 |
| $\alpha$ | retain loss 权重 |



### 6. AltPO：通过让模型偏向替代事实，削弱模型对原始 forget knowledge 的偏好。

## 公式

$$L=-\frac{2}{\beta}\mathbb{E}_{(x,y_f)\sim D_{\text{forget}}}\log \sigma\left(-\beta\log\frac{p(y_{\text{alt}}\mid x; f_{\text{unl}})}{p(y_{\text{alt}}\mid x; f_{\text{target}})}-
\beta
\log
\frac{
p(y_f\mid x; f_{\text{unl}})
}{
p(y_f\mid x; f_{\text{target}})
}
\right)
+
\alpha
\mathbb{E}_{(x,y)\sim D_{\text{retain}}}
\ell(y\mid x; f_{\text{unl}})
$$

---





| 符号 | 含义 |
|---|---|
| $y_f$ | 原始 forget answer |
| $y_{\text{alt}}$ | 替代答案，通常是模型自己生成的 in-domain plausible fact |
| $f_{\text{target}}$ | 原始模型 |
| $f_{\text{unl}}$ | unlearned model |

 



### 7. RMU：在 hidden representation 层面遗忘：forget data 的表示被推向随机方向，retain data 的表示保持接近原始模型。

## 公式

$$L=
\mathbb{E}_{(x,y_f)\sim D_{\text{forget}}}\frac{1}{|y_f|}\sum_{i=1}^{|y_f|}\left\|\phi([x,y_{<i}]; f_{\text{unl}})c\cdot u\right\|_2^2+\mathbb{E}_{(x,y)\sim D_{\text{retain}}}\frac{1}{|y|}\sum_{i=1}^{|y|}\left\|\phi([x,y_{<i}]; f_{\text{unl}})-
\phi([x,y_{<i}]; f_{\text{target}})
\right\|_2^2
$$

---





| 符号 | 含义 |
|---|---|
| $\phi(\cdot; f)$ | 模型 $f$ 的中间层表示 / embedding feature |
| $[x,y_{<i}]$ | 输入 $x$ 加上答案前 $i-1$ 个 token |
| $y_{<i}$ | 答案中第 $i$ 个 token 之前的前缀 |
| $u$ | 随机向量，元素从 $[0,1)$ 采样 |
| $c$ | 缩放超参数 |
| $\|\cdot\|_2^2$ | 平方 L2 距离 |
| $y_f$ | forget answer 的长度 |
| $y$ | retain answer 的长度 |



### 8. UNDIAL:先把 forget answer 的 logit 降低，再让模型模仿这个削弱后的输出分布。

## 公式 1：调整 logits

$$z_{\text{adj}}(x)=z_{\text{orig}}(x)-
\beta \cdot \mathbf{1}_{y_f}
$$

## 公式 2：训练目标

$$L=
\gamma
\mathbb{E}_{(x,y_f)\sim D_{\text{forget}}}
\left[
KL
\left(
\text{softmax}(z_{\text{adj}}(x))
\|
\text{softmax}(z_{\text{unl}}(x))
\right)
\right]
+
\alpha
\mathbb{E}_{(x,y)\sim D_{\text{retain}}}
\ell(y\mid x; f_{\text{unl}})
$$

---






| 符号 | 含义 |
|---|---|
| $z_{\text{orig}}(x)$ | unlearning 前模型对输入 $x$ 的原始 logits |
| $z_{\text{adj}}(x)$ | 调低 forget token 后的 adjusted logits |
| $z_{\text{unl}}(x)$ | 当前 unlearned model 输出的 logits |
| $\mathbf{1}_{y_f}$ | forget answer 对应 token 的指示向量 |
| $\beta$ | 调低 forget token logit 的幅度 |
| $KL(\cdot\|\cdot)$ | 两个概率分布之间的 KL divergence |
| $\text{softmax}$ | 把 logits 转成概率分布 |
| $\gamma$ | forget-side KL loss 权重 |
| $\alpha$ | retain loss 权重 |

### 9. WGA：在 GA 的基础上给每个 forget token 加权：还没忘掉的 token 权重大，已经忘得差不多的 token 权重小，从而缓解过度遗忘。

WGA 的核心是： 
在 forget set 上仍然做反向训练，但不是每个 token 都同等强度遗忘，而是给每个 token 一个权重，防止 GA 过度遗忘。

## 公式

$$L_{\text{WGA}}=
\mathbb{E}_{(x,y)\sim D_u}
\sum_{k=1}^{|y|}
w_{x,y,k}^{\text{wga}}
\log p(y_k \mid y_{<k},x;\theta)
$$

其中：

$w_{x,y,k}^{\text{wga}}=p(y_k \mid y_{<k},x;\theta)^{\beta}$

如果加入 retain regularization，可以写成：

$$L=\mathbb{E}_{(x,y)\sim D_u}\sum_{k=1}^{|y|}w_{x,y,k}^{\text{wga}}\log p(y_k \midy_{<k},x;\theta)-
\lambda
\mathbb{E}_{(x,y)\sim D_r}
\log p(y\mid x;\theta)
$$



| 符号 | 含义 |
|---|---|
| $L_{\text{WGA}}$ | WGA 的 unlearning loss |
| $D_u$ | unlearn set / forget set，要遗忘的数据 |
| $D_r$ | retain set，要保留的数据 |
| $(x,y)$ | 一条要遗忘的输入-输出样本 |
| $y_k$ | 答案 $y$ 中第 $k$ 个 token |
| $y_{<k}$ | 第 $k$ 个 token 之前的前缀 |
| $p(y_k \mid y_{<k},x;\theta)$ | 当前模型生成第 $k$ 个 token 的概率 |
| $w_{x,y,k}^{\text{wga}}$ | WGA 给第 $k$ 个 token 分配的遗忘权重 |
| $\beta$ | 控制权重强度的超参数 |
| $\theta$ | 当前模型参数 |
| $\lambda$ | retain regularization 的权重 |



### 10. SatImp：是一种结合 saturation 和 importance 的 token 加权遗忘方法，它通过 $p^{\beta_1}(1-p)^{\beta_2}$ 给 token 分配权重，让遗忘更平滑、更不容易伤害保留能力。
SatImp 的核心是： 
给 forget set 中每个 token 一个结合 saturation 和 importance 的权重，让模型重点遗忘“中等难度 / 中等 loss”的 token。

## 公式

一般的 token-wise reweighting unlearning 形式是：

$$L_{\text{reweight}}=
\mathbb{E}_{(x,y)\sim D_u}
\sum_{k=1}^{|y|}
w_{x,y,k}
\log p(y_k \mid y_{<k},x;\theta)
$$

SatImp 的权重定义为：

$w_{x,y,k}^{\text{satimp}}=p(y_k \mid y_{<k},x;\theta)^{\beta_1}\cdot\left(1-p(y_k \mid y_{<k},x;\theta)\right)^{\beta_2}$

如果加入 retain regularization，可以写成：

$$L=
\mathbb{E}_{(x,y)\sim D_u}
\sum_{k=1}^{|y|}
w_{x,y,k}^{\text{satimp}}
\log p(y_k \mid y_{<k},x;\theta)-
\lambda
\mathbb{E}_{(x,y)\sim D_r}
\log p(y\mid x;\theta)
$$



| 符号 | 含义 |
|---|---|
| $L_{\text{reweight}}$ | 加权遗忘损失 |
| $D_u$ | unlearn set / forget set |
| $D_r$ | retain set |
| $(x,y)$ | 一条要遗忘的输入-输出样本 |
| $y_k$ | 输出序列 $y$ 中第 $k$ 个 token |
| $y_{<k}$ | 第 $k$ 个 token 之前的上下文 |
| $p(y_k \mid y_{<k},x;\theta)$ | 模型对第 $k$ 个 token 的预测概率 |
| $w_{x,y,k}^{\text{satimp}}$ | SatImp 给第 $k$ 个 token 的权重 |
| $\beta_1$ | saturation 部分的权重控制参数 |
| $\beta_2$ | importance 部分的权重控制参数 |
| $\theta$ | 当前模型参数 |
| $\lambda$ | retain regularization 权重 |


### 11. CE-U：是通过把 forget answer 的真实 token 概率从目标分布中移除，让模型学习一个不再支持原答案的分布，从而实现更稳定的遗忘。


CE-U 的核心是： 
不直接最大化 forget answer 的 loss，而是在 logit 空间中把真实 token 的 logit 设为 $-\infty$，构造一个“不包含正确答案”的 teacher distribution，让模型去模仿这个分布。

## 公式

先定义原始 logits：

$$
z_i
$$

其中 $i$ 表示词表中的第 $i$ 个 token，$y$ 是真实 token 的索引。

CE-U 修改 logits：

$$
z_{i,\text{CE-U}}=
\begin{cases}
-\infty, & i = y \\
z_i, & i \neq y
\end{cases}
$$

然后得到 CE-U 的目标分布：

$$
p_{i,\text{CE-U}}=
\text{softmax}(z_{\text{CE-U}})_i
$$

最后 CE-U loss 为：

$$
L_{\text{CE-U}}=-
\sum_i
\text{sg}(p_{i,\text{CE-U}})
\log p(i)
$$

其中：

$$
p(i)=\text{softmax}(z)_i
$$



## 符号说明

| 符号 | 含义 |
|---|---|
| $L_{\text{CE-U}}$ | CE-U 的 unlearning loss |
| $z_i$ | 模型对第 $i$ 个 token 的原始 logit |
| $z_{i,\text{CE-U}}$ | 修改后的 CE-U logit |
| $i$ | 词表中的 token 索引 |
| $y$ | forget answer 中真实 token 的索引 |
| $-\infty$ | 把真实 token 的 logit 压到极低，使其概率为 0 |
| $p_{i,\text{CE-U}}$ | CE-U 构造出的 teacher distribution |
| $p(i)$ | 当前模型输出分布中第 $i$ 个 token 的概率 |
| $\text{softmax}$ | 把 logits 转成概率分布 |
| $\text{sg}(\cdot)$ | stop-gradient，不让 teacher distribution 参与反向传播 |


### 12. PDU：把“遗忘”和“保留”从简单加权改成约束优化：在 retain loss 不超过预算的前提下，用 logit flattening 让 forget set 上的输出分布变平，从而更稳定地遗忘目标知识。

PDU 的核心是： 
不再简单把 forget loss 和 retain loss 加权相加，而是把 retain performance 写成硬约束：在保证 retain loss 不超过阈值的前提下，最小化 forget loss。

## 公式 1：约束优化形式

$$
\min_{\pi\in\Pi}
L_{\text{fgt}}(\pi,D_{\text{fgt}})
\quad
\text{s.t.}
\quad
L_{\text{rtn}}(\pi,D_{\text{rtn}})
\leq
\epsilon
$$

其中：

$$
\epsilon=
(1+\alpha)
L_{\text{rtn}}(\pi_{\text{ref}},D_{\text{rtn}})
$$

## 公式 2：Lagrangian

$$
\mathcal{L}(\pi,\lambda)=
L_{\text{fgt}}(\pi,D_{\text{fgt}})+
\lambda
\left(
L_{\text{rtn}}(\pi,D_{\text{rtn}})-
\epsilon
\right)
$$

## 公式 3：参数化后的 primal-dual 形式

$$
\max_{\lambda\geq 0}
\min_{\theta\in\Theta}
L_{\text{fgt}}(\pi_\theta,D_{\text{fgt}})+
\lambda
\left(
L_{\text{rtn}}(\pi_\theta,D_{\text{rtn}})-
\epsilon
\right)
$$

## 公式 4：primal-dual 更新

$$
\theta
\leftarrow
\theta-
\eta_\theta
\nabla_\theta
\mathcal{L}(\theta,\lambda)
$$

$$
\lambda
\leftarrow
\left[
\lambda+
\eta_\lambda
\left(
L_{\text{rtn}}(\pi_\theta,D_{\text{rtn}})-
\epsilon
\right)
\right]_+
$$

## 公式 5：PDU 的 forget loss，logit-margin flattening

$$
L_{\text{fgt}}^{\text{LM}}(\pi_\theta,D_{\text{fgt}})=
\mathbb{E}_{(x,y)\sim D_{\text{fgt}}}
\left[
\frac{1}{|y|}
\sum_{t=1}^{|y|}
\left(
\max_k z_{t,k}-
\frac{1}{V}
\sum_{k=1}^{V} z_{t,k}
\right)^2
\right]
$$

其中：

$z_t=\pi_{\theta}^{\text{logits}}(y_t \mid x,y_{<t})$




| 符号 | 含义 |
|---|---|
| $\pi$ | 模型策略 / 模型输出分布 |
| $\pi_\theta$ | 参数为 $\theta$ 的模型 |
| $\Pi$ | 可选模型函数空间 |
| $\Theta$ | 参数空间 |
| $D_{\text{fgt}}$ | forget set |
| $D_{\text{rtn}}$ | retain set |
| $L_{\text{fgt}}$ | forget loss，用来推动遗忘 |
| $L_{\text{rtn}}$ | retain loss，用来衡量保留能力 |
| $\epsilon$ | retain loss 允许的最大阈值 |
| $\alpha$ | 控制 retain loss 允许退化程度的超参数 |
| $\pi_{\text{ref}}$ | reference model，通常是 unlearning 前的模型 |
| $\lambda$ | dual variable / Lagrange multiplier |
| $\eta_\theta$ | 更新模型参数的学习率 |
| $\eta_\lambda$ | 更新 dual variable 的学习率 |
| $[\cdot]_+$ | 投影到非负区间，即 $\max(\cdot,0)$ |
| $z_{t,k}$ | 第 $t$ 个生成位置上第 $k$ 个词表 token 的 logit |
| $V$ | 词表大小 |
| $\max_k z_{t,k}$ | 当前 token 位置上最大的 logit |
| $\frac{1}{V}\sum_{k=1}^V z_{t,k}$ | 当前 token 位置上的平均 logit |





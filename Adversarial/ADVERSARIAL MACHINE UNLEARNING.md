# ADVERSARIAL MACHINE UNLEARNING
模型删数据后，到底删干净没有。
- Unlearner 先改模型，Auditor 再用最强的 membership inference 去查；unlearner 的目标就是既保住 retain/test 性能，又让 auditor 查不出来。

**MIA** 
- 训练过的样本，模型通常更熟悉，它们在模型输出上往往和没见过的样本不一样。比如更高置信度、更低 loss
- MIA 在这里被改造成区分 forget outputs 与 test outputs 的二分类任务，不再问‘它是不是 member’，而是问‘forget 样本的输出还像不像 member’。如果还像，说明没忘干净；如果已经和 test 样本差不多，就说明残迹变弱了。


## Auditing Set 的定义

对于 unlearned model $\theta_u$，定义其 auditing set 为

![9aacf240-4cc4-43ef-acf8-840f7c7d940b](https://github.com/user-attachments/assets/70f7d77a-b887-4537-987f-fa5631fad665)




其中：

$$
s_j^f = S_{\theta_u}(x_j^f), 
\qquad
s_j^{te} = S_{\theta_u}(x_j^{te})
$$


#### 符号说明
- $\tilde{D}_{\theta_u}$: 针对 unlearned model $\theta_u$ 构造的审计数据集
- $S_{\theta_u}(x)$: 模型 $\theta_u$ 对输入 $x$ 的输出
- $s_j^f$: forget 样本经过 $\theta_u$ 后的输出
- $s_j^{te}$: test 样本经过 $\theta_u$ 后的输出
- 上标 $te$: test
- 标签 1: member 类，代表 forget 样本
- 标签 0: non-member 类，代表 test 样本

这里的输出 $s$ 可以是：
- 单个标量，比如每个样本的 cross-entropy loss，也可以是整条类别概率向量
- forget 样本经过模型后的输出，记成 1；test 样本经过模型后的输出，记成 0。变成一个二分类任务：只看模型输出，判断这条输出更像 forget 还是更像 test
- 输入给 auditor 的是模型输出，
标 1 / 标 0 是为了训练一个区分器，
这本质上是把“是否还有残迹”转成二分类问题

### 1.审计者的最优攻击

$$
U_a(\theta_a, \theta_u) = M\left(\widetilde{D}_{\text{val}}^{\theta_u}; \theta_a\right)
\quad \text{where} \quad
\theta_a \in \mathcal{H}_{\theta_u} = \mathop{\arg\max}_{\theta_a' \in \mathcal{H}_a} M\left(\widetilde{D}_{\text{tr}}^{\theta_u}; \theta_a'\right)
$$


#### 第一层：审计者效用函数

$$
U_a(\theta_a, \theta_u) = M\left(\widetilde{D}_{\text{val}}^{\theta_u}; \theta_a\right)
$$

表示审计者的效用 $U_a$，就是攻击模型 $\theta_a$ 在审计验证集上的表现。

---

#### 第二层：审计者最佳响应

$$
\theta_a \in \mathcal{B}_{\theta_u} = \mathop{\arg\max}_{\theta_a' \in \mathcal{H}_a} M\left(\widetilde{D}_{\text{tr}}^{\theta_u}; \theta_a'\right)
$$

表示审计者不会随便选一个攻击器，而是会从所有可选攻击器里，挑一个最强的 best response。

---

#### 符号含义
- $U_a$：auditor 的 utility，审计者效用
- $\theta_a$：auditor 使用的攻击模型参数
- $\theta_u$：unlearned model 参数
- $M(\cdot; \theta_a)$：用于衡量攻击器效果的指标函数
- $\widetilde{D}_{\text{tr}}^{\theta_u}$：审计训练集
- $\widetilde{D}_{\text{val}}^{\theta_u}$：审计验证集
- $\mathcal{B}_{\theta_u}$：给定 unlearned model $\theta_u$ 后，审计者的最佳响应集合
- $\mathop{\arg\max}$：取使目标最大的参数
- $\theta_a'$：搜索时的候选攻击器参数
- $\mathcal{H}_a$：auditor 的假设空间，也就是攻击器可以取哪些模型
- $M$ 可以是 accuracy、ROC/AUC、特定 FPR 下的 TPR 等指标

---
unlearner 面对的不是一个弱攻击者，而是一个**看到你模型后会挑最优攻击方式的人**。
### 2.unlearner 的目标函数

$$
C_u(\theta_u, \theta_a) = L(D_r; \theta_u) + \alpha \cdot M\left(\widetilde{D}_{\text{val}}^{\theta_u}; \theta_a\right)
$$


#### 第一项：保留集损失

$$
L(D_r; \theta_u)
$$

表示模型在 retain set 上的损失。它的作用是：别把该记住的数据也搞坏了。

#### 第二项：审计攻击约束项

$$
\alpha \cdot M\left(\widetilde{D}_{\theta_u}^{\text{val}}; \theta_a\right)
$$

表示审计者攻击成功的程度。因为 unlearner 在最小化这个式子，所以它希望这个值越小越好，也就是让攻击者越难分辨 forget/test 越好。

---

#### 符号含义
- $C_u$：unlearner 的 cost function
- $L(D_r; \theta_u)$：模型 $\theta_u$ 在 retain set 上的学习损失
- $\alpha \in \mathbb{R}_+$：权衡系数，正实数
- $\mathbb{R}_+$：正实数集合
- $M\left(\widetilde{D}_{\theta_u}^{\text{val}}; \theta_a\right)$：auditor 在审计验证集上的区分能力

---
- 第一项管“性能”
- 第二项管“忘干净没有”

- $\alpha$ 越大，越重视“让 auditor 失败”；
- $\alpha$ 越小，越重视“保住模型精度”。
### 3. 双层优化 / Stackelberg Game

$$
\mathop{\arg\min}_{\theta_u \in \mathcal{H}_u} L(D_r; \theta_u) + \alpha \cdot M\left(\widetilde{D}_{\theta_u}^{val}; \theta_a\right)
\quad \text{s.t.} \quad
\theta_a \in \mathcal{B}_{\theta_u}
$$




#### 符号含义
- $\min\limits_{\theta_u \in \mathcal{H}_u}$：在 unlearner 的模型空间里找最优的 $\theta_u$
- $\mathcal{H}_u$：unlearner 的假设空间
- $\text{s.t.}$：subject to，满足约束
- $\theta_a \in \mathcal{B}_{\theta_u}$：auditor 总是用对当前 $\theta_u$ 的最优攻击

---

#### 核心直觉
这是一个**「我先走，你后走」的博弈**。

- **上层**：unlearner 先决定把模型改成什么样
- **下层**：auditor 看到后，用最佳 MIA 来查你

所以 unlearner 训练时必须提前考虑：
**如果对手拿最强攻击来打，我还藏不藏得住 forget set 的痕迹？**

### 4. 对 unlearner 目标求梯度

$$
\frac{\partial C_u}{\partial \theta_u} = \frac{\partial L(D_r; \theta_u)}{\partial \theta_u} + \frac{\partial M\left(\widetilde{D}_{\theta_u}^{val}; \theta_a\right)}{\partial \theta_a} \cdot \frac{\partial \theta_a}{\partial \widetilde{D}_{\theta_u}^{tr}} \cdot \frac{\partial \widetilde{D}_{\theta_u}^{tr}}{\partial \theta_u}
$$

#### 第一项

$$
\frac{\partial L(D_r; \theta_u)}{\partial \theta_u}
$$

就是普通训练里的梯度，保证 retain set 性能。

#### 第二项（关键）
表示：
$\theta_u$ 改了，会改变审计集；审计集改了，会改变最优攻击器 $\theta_a$；攻击器改了，会改变审计指标 $M$。

所以这是一条链式传导：

$$
\theta_u \longrightarrow \widetilde{D}_{\theta_u}^{\text{tr}} \longrightarrow \theta_a \longrightarrow M
$$

---

#### 符号含义
- $\partial/\partial$：偏导数
- $\dfrac{\partial C_u}{\partial \theta_u}$：unlearner 总目标对模型参数的梯度
- $\dfrac{\partial L}{\partial \theta_u}$：retain 损失对模型参数的梯度
- $\dfrac{\partial M}{\partial \theta_a}$：审计指标对攻击器参数的梯度
- $\dfrac{\partial \theta_a}{\partial \widetilde{D}_{\theta_u}^{\text{tr}}}$：最优攻击器对审计训练集的敏感度
- $\dfrac{\partial \widetilde{D}_{\theta_u}^{\text{tr}}}{\partial \theta_u}$：审计训练集输出对 unlearned model 参数的敏感度

---


- 普通 fine-tune 只看“我自己的损失怎么变”，而这篇论文还看“我的改动会不会让对手更难攻击”。
### 5.把攻击者最优解写成隐式方程

$$
f\left(\tilde{D}_{\theta_u}^{tr}, \theta_a\right) = 0
$$
#### 攻击者最优解的隐式表达（KKT 条件）

攻击者最优解 $\theta_a$ 往往没有显式公式，不能直接写成：

$$
\theta_a = g\left(\widetilde{D}_{\theta_u}^{\text{tr}}\right)
$$

但如果**下层优化是凸的**，比如线性 SVM，那么最优解满足 KKT 条件。
于是就可以写成一个**隐式系统**：

$$
f\left(\widetilde{D}_{\theta_u}^{\text{tr}}, \theta_a\right) = 0
$$

---

#### 字母含义
- $f$：由最优性条件组成的方程组
- 方程组通常包含：stationarity（平稳性）、primal feasibility（原始可行性）、dual feasibility（对偶可行性）、complementary slackness（互补松弛性）
- $\widetilde{D}_{\theta_u}^{\text{tr}}$：审计训练集
- $\theta_a$：攻击器最优参数

---
- 虽然没有显式解，但我们有「最优时必须满足的条件」。
这就足够做**隐式求导**了。
### 6.用隐函数定理做反传

$$
\frac{\partial \theta_a}{\partial \tilde{D}_{\theta_u}^{tr}} = -\left( \frac{\partial f(\tilde{D}_{\theta_u}^{tr}, \theta_a)}{\partial \tilde{D}_{\theta_u}^{tr}} \right)^{-1} \frac{\partial f(\tilde{D}_{\theta_u}^{tr}, \theta_a)}{\partial \theta_a}
$$

可以通过对 KKT 系统做隐式求导，得到**攻击器最优解怎样随审计集变化**。这是双层优化梯度推导中最核心、难度最高的部分。

---

#### 符号含义
- $\dfrac{\partial \theta_a}{\partial \widetilde{D}_{\theta_u}^{\text{tr}}}$：攻击器最优参数对审计训练集的导数
- $f\left(\widetilde{D}_{\theta_u}^{\text{tr}}, \theta_a\right)$：KKT 条件组成的方程组（描述攻击器最优解的约束条件）
- $(\cdot)^{-1}$：矩阵的逆运算

---


1. 该公式基于**隐函数定理**，无需显式写出攻击器最优参数 $\theta_a$ 的表达式；
2. 仅依赖 KKT 最优性条件 $f(\cdot)=0$，即可完成梯度推导；
3. 矩阵求逆 $(\cdot)^{-1}$ 是该公式的关键运算，对应凸优化下 KKT 系统的可逆性假设。

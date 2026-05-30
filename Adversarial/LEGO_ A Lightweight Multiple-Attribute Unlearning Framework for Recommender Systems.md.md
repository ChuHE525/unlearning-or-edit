# LEGO: A Lightweight Multiple-Attribute Unlearning Framework for Recommender Systems

## 方法

- 任务是解决推荐系统中的多敏感属性动态遗忘问题。它首先在 Embedding Calibration 阶段，针对每个敏感属性分别最小化用户 embedding 与该属性之间的互信息，并利用 vCLUB 估计这种信息泄露，同时通过参数空间约束限制 embedding 的改动幅度，保持推荐性能；随后在 Flexible Combination 阶段，学习多个已校准 embedding 的组合权重，将它们加权融合成一个最终 embedding，使该 embedding 同时降低与多个目标敏感属性之间的互信息。

---

## 公式

### 1. 真实的互信息

这个公式通过计算变量 $x$ 和 $y$ 实际共同发生的概率与假设它们完全独立时同时发生的概率之间的比值，来量化这两个变量之间的绑定程度（即已知 $x$ 能让你对 $y$ 减少多少不确定性）。

$$
I(x;y) = \mathbb{E}_{p(x,y)} \left[ \log \frac{p(x,y)}{p(x)p(y)} \right]
$$

$$\log \frac{p(x,y)}{p(x)p(y)}=\log \frac{p(y|x)p(x)}{p(x)p(y)}=\log \frac{p(y|x)}{p(y)}=
\log p(y|x) - \log p(y)
$$

---

### 2. CLUB 互信息上界

这个公式通过计算真实配对数据的预测得分减去随机盲抽数据的预测得分所得到的差值，来估算特征 $x$ 中究竟包含了多少关于标签 $y$ 的有效隐私信息（因为乱搭导致得分更低，减去的数值更小，公式就变成了上限）。

$$
\begin{aligned}
I_{\text{CLUB}}(\boldsymbol{x}; \boldsymbol{y})
&= \mathbb{E}_{p(\boldsymbol{x},\boldsymbol{y})}
[\log p(\boldsymbol{y} | \boldsymbol{x})] \\
&\quad -
\mathbb{E}_{p(\boldsymbol{x})}
\mathbb{E}_{p(\boldsymbol{y})}
[\log p(\boldsymbol{y} | \boldsymbol{x})].
\end{aligned}
$$

---

### 3. vCLUB

引入神经网络 $q_\phi$ 替代真实概率 $p$。

$$
\begin{aligned}
I_{\text{vCLUB}}(\boldsymbol{x}; \boldsymbol{y})
&= \mathbb{E}_{p(\boldsymbol{x},\boldsymbol{y})}
[\log q_\phi(\boldsymbol{y} | \boldsymbol{x})] \\
&\quad -
\mathbb{E}_{p(\boldsymbol{x})}
\mathbb{E}_{p(\boldsymbol{y})}
[\log q_\phi(\boldsymbol{y} | \boldsymbol{x})].
\end{aligned}
$$

---

### 4. vCLUB 作为互信息上界的条件

只要 $q_\phi(\boldsymbol{x}, \boldsymbol{y})$ 更像真实配对的数据分布 $p(\boldsymbol{x}, \boldsymbol{y})$，而不是更像随机乱配的数据分布 $p(\boldsymbol{x})p(\boldsymbol{y})$ ，那么用 $q_\phi$ 构造出来的 vCLUB 可以被当作互信息 $I(\boldsymbol{x}; \boldsymbol{y})$ 的上界。

$$
KL(p(\boldsymbol{x},\boldsymbol{y}) \| q_\phi(\boldsymbol{x},\boldsymbol{y}))
\le
KL(p(\boldsymbol{x})p(\boldsymbol{y}) \| q_\phi(\boldsymbol{x},\boldsymbol{y})).
$$

---

### 公式 (5)：最小化 KL 与最大化对数似然

最小化 $KL(p(\boldsymbol{x}, \boldsymbol{y}) \| q_\phi(\boldsymbol{x}, \boldsymbol{y}))$ 最后等价于最大化：

$$
\mathbb{E}_{p(\boldsymbol{x},\boldsymbol{y})}
[\log q_\phi(\boldsymbol{y}|\boldsymbol{x})]
$$

也就是让预测器 $q_\phi(\boldsymbol{y}|\boldsymbol{x})$ 在真实配对 $(\boldsymbol{x}, \boldsymbol{y})$ 上预测得越准越好。

$$
\begin{aligned}
&\min_{\phi}
KL(p(\boldsymbol{x}, \boldsymbol{y}) \| q_\phi(\boldsymbol{x}, \boldsymbol{y})) \\
&=
\min_{\phi}
\mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}
\left[
\log(p(\boldsymbol{y}|\boldsymbol{x})p(\boldsymbol{x}))-
\log(q_\phi(\boldsymbol{y}|\boldsymbol{x})p(\boldsymbol{x}))
\right] \\
&=
\min_{\phi}
\mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}
[\log p(\boldsymbol{y}|\boldsymbol{x})]-
\mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}
[\log q_\phi(\boldsymbol{y}|\boldsymbol{x})].
\end{aligned}
$$

---

### 6. Batch 上的对数似然函数

**公式 (6)：** 用一个 batch 里的 **$B$** 个真实样本，计算预测器 **$q_\phi(\boldsymbol{y}|\boldsymbol{x})$** 在真实配对 **$(\boldsymbol{x}_i, \boldsymbol{y}_i)$** 上的平均对数概率；训练时要**最大化它**，让 **$q_\phi$** 预测 **$\boldsymbol{y}$** 更准。

$$
\mathcal{L}(\phi)=
\frac{1}{B}
\sum_{i=1}^{B}
\log q_\phi(\boldsymbol{y}_i|\boldsymbol{x}_i).
$$

- $\mathcal{L}(\phi)$：代表对数似然函数（Log-Likelihood Function）越大，说明预测器给真实标签的概率越高，预测越准。
- $B$：代表 Batch Size，一次训练中使用的样本数量。
- $\frac{1}{B}\sum_{i=1}^{B}$：代表求平均值。把这一个 Batch 里所有 $B$ 个样本的得分加起来，然后除以 $B$ 算个平均分。
- $\log q_\phi(\boldsymbol{y}_i|\boldsymbol{x}_i)$：对于第 $i$ 个样本，神经网络在看了特征 $\boldsymbol{x}_i$ 后，准确预测出它真实的标签 $\boldsymbol{y}_i$ 的对数概率。

---

### 7. Batch 内的 vCLUB 估计

**公式 (7)：** 批次（Batch）内真实标签的 log 分数 − 所有标签的平均 log 分数。

- 在一个 batch 中，比较每个 $\boldsymbol{x}_i$ 对自己真实标签 $\boldsymbol{y}_i$ 的预测分数，和对所有随机标签 $\boldsymbol{y}_j$ 的平均预测分数；**差值越大**，说明 $\boldsymbol{x}$ 中包含的 $\boldsymbol{y}$ 信息越多。

$$
\hat{I}_{vCLUB}=
\frac{1}{B}
\sum_{i=1}^{B}
\left[
\log q_\phi(y_i|x_i)-
\frac{1}{B}
\sum_{j=1}^{B}
\log q_\phi(y_j|x_i)
\right]
$$

---

## 步骤一：嵌入校准（Embedding Calibration）

### 公式 (8)

找一个新的用户 embedding $U_t^*$，让它和敏感属性 $A_t$ 的互信息 $I(U_t; A_t)$ 最小，也就是尽量“忘掉”这个属性。

$$
U_t^*=
\arg\min_{U}
I(U; A_t)
$$

---

### 公式 (9)

因为真实互信息 $I(U_t; A_t)$ 难算，所以用可计算的 $I_{vCLUB}(U_t; A_t)$ 来近似替代它。

$$
U_t^*=
\arg\min_{U_t}
I_{vCLUB}(U_t; A_t)
$$

---

### 公式 (10)

攻防交替优化，也就是：先训练预测器 $q_\phi$，让它尽量从用户 embedding $\boldsymbol{U}_t$ 中预测出敏感属性 $\boldsymbol{A}_t$；再反过来更新 embedding $\boldsymbol{U}_t$，让 vCLUB 估计出的互信息尽量小。

$$
\begin{aligned}
\phi
&=
\arg \max_{\phi}
\mathbb{E}_{p(\boldsymbol{U}_t, \boldsymbol{A}_t)}
\mathcal{L}(\phi),
\\
\boldsymbol{U}_t^*
&=
\arg \min_{\boldsymbol{U}_t}
\mathbb{E}_{p(\boldsymbol{U}_t, \boldsymbol{A}_t)}
\hat{I}_{\text{vCLUB}}.
\end{aligned}
$$

- $\phi$：预测器的参数。
- $\arg\max_{\phi}$：寻找一组最优的参数 $\phi$，使得后面的那个函数（得分）达到最大值。
- $\mathbb{E}_{p(\boldsymbol{U}_t, \boldsymbol{A}_t)}$：从真实的用户特征 - 敏感属性配对数据集中抽取样本算期望（求平均）。
- $\mathcal{L}(\phi)$：预测器的对数似然损失函数（也就是公式 6）。它代表了模型预测隐私的准确度。
- $\boldsymbol{U}_t$：第 $t$ 步时，正在被修改的用户特征向量（Embedding）。
- $\boldsymbol{U}_t^*$：经过这一轮对抗后，产生的更安全的新特征向量。
- $\arg\min_{\boldsymbol{U}_t}$：保持预测器不变，反向传播去更新特征向量 $\boldsymbol{U}_t$ 的数值，使得后面的那个函数达到最小值。
- $\hat{I}_{\text{vCLUB}}$：用 batch 估计出来的 vCLUB 互信息上界。

---

### 公式 (11)

更新后的 embedding 如果离原始 embedding $U_0$ 太远，就把它拉回半径为 $\epsilon$ 的范围内，防止推荐性能下降。

$$
U_t =
\begin{cases}
\quad \boldsymbol{U}_t,
& \text{if } \|\boldsymbol{U}_t - \boldsymbol{U}_0\|_2 \le \epsilon,
\\
\text{proj}(\boldsymbol{U}_t)=
\boldsymbol{U}_0
+
\frac{\epsilon}{\|\boldsymbol{U}_t - \boldsymbol{U}_0\|_2}
(\boldsymbol{U}_t - \boldsymbol{U}_0),
& \text{otherwise.}
\end{cases}
$$

- $\boldsymbol{U}_0$：原始用户 embedding，也就是还没做遗忘之前的用户向量。
- $\boldsymbol{U}_t$：更新后的用户 embedding，正在为了遗忘第 $t$ 个敏感属性而被修改。
- $\boldsymbol{U}_t - \boldsymbol{U}_0$：新 embedding 和原始 embedding 的差值。
- $\|\boldsymbol{U}_t - \boldsymbol{U}_0\|_2$：二范数，表示 $\boldsymbol{U}_t$ 离 $\boldsymbol{U}_0$ 有多远。
- $\epsilon$：允许 embedding 最大偏离原始 embedding 的距离。
- $\text{proj}(\boldsymbol{U}_t)$：投影操作，意思是把超出范围的 $\boldsymbol{U}_t$ 拉回合法范围内。从 $U_0$ 出发，沿着原来 $U_t$ 的方向走，但只走 $\epsilon$ 这么远。

---

## 步骤二：灵活组合（Flexible Combination）

### 公式 (12)

给多个已经分别遗忘不同属性的 embedding 分配权重，把它们加权组合成一个新 embedding，并让这个新 embedding 和所有敏感属性的互信息都尽可能小。

$$
\begin{aligned}
&\min_{\boldsymbol{\alpha}}
\sum_{i=1}^{k}
I(U(\boldsymbol{\alpha}); A_i),
\\
&\text{s.t.}
\quad
\alpha_i > 0,
\ i = 1, \dots, k,
\quad
\|\boldsymbol{\alpha}\|_1 = 1.
\end{aligned}
$$

- $k$：需要保护的敏感属性数量。比如要保护年龄、性别、职业，则 $k=3$。

- $A_i$：第 $i$ 个敏感属性。例如：

  $$
  A_1 = \text{年龄}, \quad A_2 = \text{性别}, \quad A_3 = \text{职业}$$

- $\boldsymbol{U}_i^*$：前一步 **Embedding Calibration** 得到的 embedding。例如：

  - $\boldsymbol{U}_1^*$：已经尽量遗忘年龄的 embedding；
  - $\boldsymbol{U}_2^*$：已经尽量遗忘性别的 embedding；
  - $\boldsymbol{U}_3^*$：已经尽量遗忘职业的 embedding。

- $\alpha_i$：第 $i$ 个 embedding 的组合权重。

- $\boldsymbol{\alpha}$：所有权重组成的向量：

  $$
  \boldsymbol{\alpha}=
  [\alpha_1, \alpha_2, \dots, \alpha_k]
  $$

- $\boldsymbol{U}(\boldsymbol{\alpha})$：最终组合出来的新 embedding，可以理解为加权求和：

  $$
  \boldsymbol{U}(\boldsymbol{\alpha})=
  \sum_{i=1}^{k}
  \alpha_i \boldsymbol{U}_i^*
  $$

- $I(\boldsymbol{U}(\boldsymbol{\alpha}); A_i)$：最终 embedding $\boldsymbol{U}(\boldsymbol{\alpha})$ 和第 $i$ 个敏感属性 $A_i$ 的互信息。

- $\sum_{i=1}^{k} I(\boldsymbol{U}(\boldsymbol{\alpha}); A_i)$：最终 embedding 和所有敏感属性之间的互信息总和。

- $\min_{\boldsymbol{\alpha}}$：通过调整权重向量 $\boldsymbol{\alpha}$，让这个总互信息最小。

- $\text{s.t.}$：subject to，表示“满足约束条件”。

- $\alpha_i > 0$：每个权重都要大于 0。

- $\|\boldsymbol{\alpha}\|_1 = 1$：所有权重的 L1 范数（绝对值之和）等于 1；因为约束了权重大于 0，所以展开就是：

  $$
  \alpha_1 + \alpha_2 + \dots + \alpha_k = 1
  $$
---

### 公式 (13)

用 softmax 把任意权重 $\alpha$ 转换成一组合法权重，使每个权重大于 0，并且所有权重加起来等于 1。

$$
\text{proj}(\boldsymbol{\alpha})=
\text{softmax}(\boldsymbol{\alpha})=
\left[
\frac{\exp(\alpha_1)}{\sum_{j=1}^{k}\exp(\alpha_j)},
\dots,
\frac{\exp(\alpha_k)}{\sum_{j=1}^{k}\exp(\alpha_j)}
\right].
$$

- $\text{proj}(\boldsymbol{\alpha})$：投影操作，把不合法的权重强行变成合法权重。
- $\text{softmax}(\boldsymbol{\alpha})$：softmax 函数，用来把一组任意的实数转换成一组加起来等于 1 的正数权重。
- $\exp(\alpha_i)$：对 $\alpha_i$ 取指数运算，它的作用是保证输出的结果一定大于 0（无论原始 $\alpha_i$ 是正是负）。
- $\sum_{j=1}^{k}\exp(\alpha_j)$：所有指数值的总和，用来做底部的分母进行“归一化”。
- $\frac{\exp(\alpha_i)}{\sum_{j=1}^{k}\exp(\alpha_j)}$：第 $i$ 个归一化后的最终权重。它代表了整体的一个百分比。

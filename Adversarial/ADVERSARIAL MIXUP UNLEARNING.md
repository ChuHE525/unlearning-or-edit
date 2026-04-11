---
title: ADVERSARIAL MIXUP UNLEARNING

---

# ADVERSARIAL MIXUP UNLEARNING

## *方法*
- **任务**：克服灾难性遗忘（模型在遗忘目标信息时，把remaining data中本该保留的知识也一起破坏掉了，不能像之前那样正确识别这些样本），从一个已经训练好的模型里，删除 Forgetting data 对应的知识,同时保留 Remaining data 对应的知识。
- **方法**：提出了 MixUnlearn方法。它先从 Forgetting 和 Remaining 中各取一个样本，在特征层通过 MixBlock 构造 mixed sample；然后用 generator loss 训练一个对抗生成器，故意生成最容易让 unlearner 出错的 hard mixed samples；最后再用对比损失函数对 unlearner 进行训练，让它在 mixed samples 和真实样本上都做到远离 Forgetting、接近 Remaining，从而减轻 catastrophic unlearning。
## *公式*
- 第一步：构造 mixed sample
- 第二步：训练 generator 去故意制造更难的 mixed samples
- 第三步：训练 unlearner 去纠正这些 hardest cases
### 定义混合样本怎么来
#### 从定义上说明“困难混合样本”是怎么构造的

$$
x_{ij}^{\text{mix}} = g(x_i, x_j, \lambda)
$$

- 这个公式的目的是生成一个混合样本  $x^{mix}_{ij}$ ，它位于遗忘集和保留集之间。具体来说， $x_i$ 是遗忘样本， $x_j$  是保留样本，而  $\lambda$  是一个从 Beta 分布中采样的混合比例，控制两者在混合样本中的占比。接着，生成器 g 是一个可学习的模块，它根据  $x_i$ 、 $x_j$  的特征以及  $\lambda$ ，生成一个既包含遗忘部分又包含保留部分的中间样本，从而生成最容易让unlearner出错的混合样本。
- 生成中间样本的目的是为了更好的模拟出遗忘和保留数据之间的边界（特征上带有遗忘集和保留集的共同属性，同时保留了俩类样本中一部分对模型判断有用的关键信息， 更容易引发灾难性遗忘），然后用这些样本去训练unlearner。
- 生成器的具体任务是输入一个遗忘样本、一个保留样本和混合比例后，构造一个 mixed sample，并通过对抗优化把这个 mixed sample 变成 hard sample，也就是让当前 unlearner 在这个样本上更容易暴露 Forgetting 信息、同时更容易丢掉 Remaining 知识。

生成器的具体作用是主动制造出 Forgetting 和 Remaining 之间最容易发生 catastrophic unlearning 的边界情况，把 unlearner 的弱点显式暴露出来，并为后续训练 unlearner 提供最有针对性的 mixed samples。
#### 在初始模型的特征层上用 MixBlock 

$$
x_{ij}^{\text{mix}} = \mathrm{MixBlock}\big(h_D(x_i),\, h_D(x_j),\, \lambda\big)
$$

-  $\mathrm{MixBlock}$ ：论文采用的可学习混合模块。一个轻量级生成器，用于学习如何混合两个样本。

$$
x_{ij}^{\text{mix}} = x_i \odot M_i + x_j \odot (1 - M_i)
$$

- MixBlock 的输入是从初始模型 $f_d$ 前几层（例如前面的编码层和卷积层）提取的特征(中间语义表示)  $h_D(x_i)$  和  $h_D(x_j)$ ，以及混合比例  $\lambda$ 。它的做法是在特征层按位置加权混合：每个位置用一个权重  $M_i$  决定更偏向遗忘还是保留。这样做的目的是在模型真正“理解”的空间里生成边界样本，让后续的 unlearning更精准。
- 选用 MixBlock，样本经过初始模型后，会被转换为一个由许多具体数值组成的特征矩阵（或特征向量），其中的每一个数值都代表着一个细微的语义维度。MixBlock 的做法，就是给这成千上万个特征数值，每一个都单独分配一个独立的权重系数进行相乘。这样就能打破传统的一刀切混合，在极其微观的层面上，精准地调配遗忘特征和保留特征的比例，从而缝合出最完美的边界样本

### 定义生成器怎么训练
#### 训练生成器的损失函数
目的：generator 的损失函数,用这个损失来更新生成器参数，让生成器朝更会造难样本的方向优化，让当前 unlearner 更容易：暴露 Forgetting 信息，丢掉 Remaining 知识。

$$
\mathcal{L}_{\mathrm{gen}}
= - \sum_{x_j \in B_r}
\log
\frac{
\exp\!\left((1-\lambda)\,\mathrm{SimLoss}\big(f_U(x^{\text{mix}}_{ij}),\, p(x_j)\big)\right)
}{
\sum_{x_i \in B_f}
\exp\!\left(
\lambda\,\mathrm{SimLoss}\big(f_U(x^{\text{mix}}_{ij}),\, p(x_i)\big)\;/\;\tau_{\mathrm{gen}}
\right)
}
$$

- 分子就是表示e的指数次方，指数上写着 $1-\lambda$  * simloss(就是指1−cosine similarity，cos看他们俩个向量的方向夹角，越相似夹角越小）
（用于衡量混合样本  $x_{ij}^{\text{mix}}$  经 unlearner  $f_U$  后的输出，与 Remaining 样本  $x_j$  的目标分布  $p(x_j)$  之间的不相似程度。)
   -  $\mathrm{SimLoss}$  小：
  表明模型输出与目标分布  $p(x_j)$  在语义上非常相似。

  -  $\mathrm{SimLoss}$  大： 
  表明模型输出与目标分布  $p(x_j)$  在语义上差异较大。
- Bf和Br分别表示来自于遗忘集和保留集，分母中指数部分还有一个温度参数，Tgen用于控制分母指数的缩放程度，决定了分母对不相似度的反应速度，从而影响生成器最终挑选的样本难度。
- 混合样本输出和 Forgetting 样本目标分布 𝑝(𝑥𝑖 ）的不相似程度。
同样地：
  - SimLoss 小：说明输出更像 Forgetting
  - SimLoss 大：说明输出远离 Forgetting
- 分子负责让 mixed sample 输出远离 Remaining；
分母负责让 mixed sample 输出接近 Forgetting；
加上前面的负号后，generator 就会专门生成一种样本，使 unlearner 保不住 Remaining、又露出 Forgetting。
- 这个负号让整个优化方向变成对抗式的，也就是故意让 generator 生成让 unlearner 更容易出错的样本。
- 求和就是 把一个 batch 里所有保留样本对应的损失都累加起来。
#### 目标分布 𝑝(𝑥)

$$
p(x) =
\begin{cases}
y, & \text{if the label } y \text{ of } x \text{ is available (label-aware)} \\
\mathrm{Sharpen}\!\left(f_D(x)\right), & \text{otherwise (label-agnostic)}
\end{cases}
$$

目的是为了让方法既支持有标签也支持无标签，统一定义训练时比较用的目标分布
-  $p(x)$ ：目标分布（target distribution）。作为与模型输出进行对齐的参考目标。

-  $y$ ：真实标签对应的 one-hot 向量。在有真实标签的情况下，目标分布直接取真实标签。

-  $f_D(x)$ ：初始模型对样本 $x$ 的预测分布。在无标签场景下，用初始模型的输出作为伪标签。

-  $\mathrm{Sharpen}(\cdot)$ ：锐化操作。使预测分布更加尖锐，从而增强类别判别性。
#### 训练 unlearner 在混合样本上表现正确的损失

$$
\mathcal{L}_{\mathrm{mix}}=
\sum_{x_j \in B_r}
\log
\frac{
\exp\!\left((1-\lambda)\,\mathrm{SimLoss}\big(f_U(x^{\text{mix}}_{ij}),\, p(x_j)\big)\right)
}{
\sum_{x_i \in B_f}
\exp\!\left(
\lambda\,\mathrm{SimLoss}\big(f_U(x^{\text{mix}}_{ij}),\, p(x_i)\big)\;/\;\tau_{\mathrm{mix}}
\right)
}
$$

- 公式的目的，是让 unlearner 在 generator 制造出来的困难混合样本上，学会远离 Forgetting、接近 Remaining，从而减少灾难性 unlearning。
#### 真实样本上的对比损失

$$
\mathcal{L}_{\mathrm{real}}=
\sum_{x_j \in B_r}
\log
\frac{
\exp\!\left(\mathrm{SimLoss}\big(f_U(x_j),\, p(x_j)\big)\right)
}{
\sum_{x_i \in B_f}
\exp\!\left(
\mathrm{SimLoss}\big(f_U(x_i),\, p(x_i)\big)\;/\;\tau_{\mathrm{real}}
\right)
}
$$

- 是在真实样本上训练 unlearner，在真正的原始数据上也能做到：

对 Remaining 样本保留知识
对 Forgetting 样本忘掉知识，帮助模型在原始真实样本上进行 unlearning
- 公式  $L_{real}$ 就是在真实的保留集和遗忘集上计算损失。Xj  $\in B_r$  是保留集里的样本，这些样本的相似度是 unlearner 输出与其真实目标分布之间的距离。相似度是通过 1 减去余弦相似度来衡量的，也就是看两个向量夹角有多大。分母是对遗忘集样本的相似度求和， $\tau_{real}$  就是用来调节分母里 Forgetting 项的敏感程度。
#### 最终训练 unlearner 的总损失

$$
L_{\text{unlearn}} = L_{\text{mix}} + \omega L_{\text{real}}
$$

- 公式的目的，是把困难混合样本上的鲁棒遗忘和真实样本上的准确遗忘结合起来，形成最终完整的 unlearning 训练目标




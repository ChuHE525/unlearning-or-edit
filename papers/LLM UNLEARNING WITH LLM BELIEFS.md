# LLM UNLEARNING WITH LLM BELIEFS

* 这篇文章指出传统 unlearning 会把概率质量挤到语义相近的改写答案上，造成伪遗忘；为了解决这个问题，他们引入 model beliefs，并同时压制 target response 和 beliefs，从而更彻底地忘掉目标内容。
* 现有 LLM unlearning 方法虽然能压低目标答案概率，但常常只造成“伪遗忘”，因为概率质量会被挤到语义相近的高概率改写上。为了解决这个问题，作者提出利用模型自身的高置信生成（model beliefs）作为辅助遗忘目标，构建 bootstrapping unlearning 框架
* LLM 会记住有害或敏感信息，因此需要通过 unlearning 直接从模型参数中移除这些知识；现有 unlearning 方法只是把“原答案”压下去，但知识并没有被真正移除，而是逃到高概率的改写表达中；引入 model beliefs：既然概率质量会被“挤”到模型自己最容易生成的高概率区域，
那 unlearning 时就不能只压 target response，
还要一起压制这些模型自己高置信生成的内容。

作者把这些内容叫做：

model beliefs

也就是模型当前最相信、最可能生成的 token 或 sequence。

- 于是他们提出一个：

bootstrapping (BS) framework

意思是：

利用模型自己生成出来的高置信内容

把这些内容再反过来作为 unlearning 信号

一起参与“遗忘”

第一节里还提前概括了两个版本：

BS-T：token level
把 target token 和模型高概率 token 一起压制。

BS-S：sequence level
直接把模型生成的高置信完整序列也加入 forget set，一起删。

## Preliminaries: From Concepts to Practices 预备知识
- 先把任务形式化：

词表记作 V

输入  记作 x

输出  记作 y

模型参数记作 θ

模型按自回归方式，一个 token 一个 token 地生成回答。

然后定义 LLM unlearning 的数据和目标：

训练数据：
𝐷𝑡



要忘掉的数据：
𝐷𝑢⊆𝐷𝑡



保留数据：
𝐷𝑟


目标有两个：

Unlearning：让 unlearned model 对 forget set 以及它们的改写版本赋予低概率

Retention：对其他无关输入，输出分布尽量和原模型保持接近。
- 基本符号

V：词表（vocabulary）

x：输入 prompt

y：模型输出 response

|y|：response 的长度

θ：当前模型参数

θo：原始模型参数（unlearning 前）

θu：unlearned 后模型参数

Dt：原始训练集

Du：forget set，要忘掉的数据

Dr：retain set，要保留行为的数据

(xu, yu)：一条 forget 数据，输入是 
𝑥𝑢

，目标回答是 
𝑦𝑢


(xr, yr)：一条 retain 数据

πθ(·|x, y<i)：在给定 prompt 和前缀的条件下，模型下一 token 的条件分布

yi：response 第 
𝑖
i 个 token

y<i：第 
𝑖
i 个 token 之前的前缀

πθ(y|x)：整个序列 
𝑦
y 在 prompt 
𝑥
x 下的概率。

- 义 1：单个 token 的条件概率

- 四个公式的核心含义

GA：直接降低 forget response 的 likelihood。

GradDiff：在 GA 基础上加 retain loss，试图保护正常能力。

NPO：把当前模型与原模型在 forget sample 上的概率做比值，实现 instance-wise reweighting。

WGA：进一步细化到 token 级别做加权，实现更细粒度的 unlearning。

- 第三节的核心是在说明：现有 unlearning 经常只是“表面忘记”，真正原因是压低目标答案后，概率质量被 softmax 挤到高概率的语义相近改写区域，因此模型仍会通过 paraphrase 泄漏知识。
- 第四节提出 belief-aware unlearning：不只忘原答案，还要忘模型会逃到的高置信改写区域；方法上通过 bootstrapping 把模型自己的高概率预测变成辅助 unlearning 信号。







# 第四节笔记：Bootstrapping-Based Unlearning

## 1. 第四节在讲什么

第四节是这篇文章最核心的方法部分。作者在前面已经说明，传统 unlearning 方法通常只会压制原始 target response 的概率，但这样做会带来一个问题：

- 被压下去的概率质量不会消失；
- 它会转移到模型当前本来就很有信心的高概率区域；
- 这些高概率区域往往对应语义相近的改写表达；
- 于是模型表面上像是忘了，实际上只是换了一种说法继续输出。

作者把这个现象称为 **squeezing effect**。

为了解决这个问题，第四节提出一个新的思路：

> 不仅要压制原始 target，还要压制模型自己当前最可能生成的那些高置信内容，也就是 model beliefs。

这个方法叫做 **Bootstrapping-Based Unlearning**，包括两个层次：

- **BS-T**：token level 的 bootstrapping
- **BS-S**：sequence level 的 bootstrapping

---

## 2. 第四节的核心思路

第四节的思路可以概括为两句话：

1. 模型会把概率质量挤到它自己最相信的高概率 token 或高概率序列上。
2. 所以 unlearning 时，不仅要删原答案，还要删这些“模型自己会逃去的答案”。

也就是说，作者的方法不是只盯住人工给定的 target，而是把模型自己的高置信输出也一起拉进遗忘目标。

---

## 3. 局部 belief：top-k 高概率 token 集合

### 公式

$$
H_k^{(i)} = \mathrm{Top}\text{-}k(\pi_\theta(\cdot \mid x_u, y_u^{<i}))
$$

### 公式含义

这个公式表示：

在第 $i$ 个 token 位置上，给定 forget prompt $x_u$ 和前缀 $y_u^{<i}$，模型会输出一个“下一个 token 的概率分布”。  
从这个分布中，取概率最高的前 $k$ 个 token，组成一个集合，这个集合就记作 $H_k^{(i)}$。

### 字母解释

- $H_k^{(i)}$：第 $i$ 个位置上的 top-k 高概率 token 集合
- $k$：取前多少个高概率 token
- $i$：当前 token 位置
- $\pi_\theta(\cdot \mid x_u, y_u^{<i})$：模型在给定输入和前缀条件下，对下一个 token 的条件概率分布
- $\theta$：当前模型参数
- $x_u$：forget prompt，也就是需要遗忘的输入
- $y_u^{<i}$：目标序列在第 $i$ 个位置之前的前缀
- $\mathrm{Top}\text{-}k(\cdot)$：从概率分布中选出概率最高的前 $k$ 个 token

### 这个公式的用途

这个公式的作用是定义 **局部高概率 belief 区域**。  
作者认为，传统方法压低 target token 后，概率最容易流向这里。因此，后面的 BS-T 会针对这些 top-k 高概率 token 一起做 suppression，而不是只压一个原始 target token。

---

## 4. 全局 belief：高置信序列采样

### 公式

$$
\hat{y}_u \sim \pi_\theta(\cdot \mid x_u)
$$

### 公式含义

这个公式表示：

对于 forget prompt $x_u$，模型不只是会在某个 token 位置上有高概率 token，它还会生成整条高置信的完整回答。  
作者把这样的完整高置信回答看作 **sequence-level 的 belief**。

### 字母解释

- $\hat{y}_u$：模型基于 forget prompt $x_u$ 生成的一条高置信完整序列
- $\pi_\theta(\cdot \mid x_u)$：模型在输入 $x_u$ 下，对所有可能输出序列的分布
- $\theta$：当前模型参数
- $x_u$：forget prompt
- $\sim$：表示“从该分布中采样得到”

### 这个公式的用途

这个公式的作用是定义 **全局 belief**。  
后面的 BS-S 就是把这些模型自己最可能说出来的完整回答，也加入 forget set，一起做遗忘。

---

## 5. BS-T 的 soft target

### 公式

$$
t_u^i = \lambda_{\mathrm{BST}} \, \mathrm{sg}(\pi_\theta(\cdot \mid x_u, y_u^{<i})_{H_k^{(i)}}) + (1 - \lambda_{\mathrm{BST}}) e_{y_u^i}
$$

### 公式含义

这个公式是 BS-T 的核心。

在传统方法里，第 $i$ 个位置只会针对原始 target token $y_u^i$ 做压制。  
而这里作者构造了一个新的 **soft target**，记作 $t_u^i$。

这个 soft target 由两部分组成：

1. 模型当前 top-k 高概率 token 的 belief 分布
2. 原始 target token 的 one-hot 表示

也就是说，BS-T 在第 $i$ 个位置上不只是压制原始 target token，还会一起压制模型当前最可能逃过去的那些高概率 token。

### 字母解释

- $t_u^i$：第 $i$ 个位置上的 soft unlearning target
- $\lambda_{\mathrm{BST}}$：BS-T 的混合系数，用来控制 belief 分布所占的权重
- $\mathrm{sg}(\cdot)$：stop-gradient，表示这一项只作为目标使用，不让梯度反向传回去
- $\pi_\theta(\cdot \mid x_u, y_u^{<i})_{H_k^{(i)}}$：当前位置的模型概率分布，但只取 top-k 高概率 token 对应的部分
- $H_k^{(i)}$：第 $i$ 个位置上的 top-k 高概率 token 集合
- $e_{y_u^i}$：目标 token $y_u^i$ 的 one-hot 向量
- $y_u^i$：forget target 序列在第 $i$ 个位置上的 token
- $x_u$：forget prompt
- $y_u^{<i}$：当前位置之前的前缀
- $\theta$：当前模型参数

### 这个公式的用途

这个公式的作用是构造一个 **belief-aware 的 soft target**。  
它让 unlearning 从“只压一个正确 token”变成“同时压一片高概率 token 区域”，从而缓解 token-level 的 squeezing effect。

---

## 6. BS-T 的 token-level loss

### 公式

$$
L_{\mathrm{BST}}(\theta; D_u) = \mathbb{E}_{D_u} \left[ \sum_{i=1}^{|y_u|} \langle t_u^i, \log \pi_\theta(\cdot \mid x_u, y_u^{<i}) \rangle \right]
$$

### 公式含义

这个公式表示：

对于 forget set $D_u$ 中的每个样本，在每个 token 位置上，都使用前面构造的 soft target $t_u^i$，去和当前位置的 log 概率分布做加权匹配，然后把所有位置加总，最后对整个 forget set 求平均。

因为 $t_u^i$ 不只是 one-hot target，而是包含了 belief 分布，所以这个 loss 会同时压制：

- 原始 target token
- top-k 的高概率 belief token

### 字母解释

- $L_{\mathrm{BST}}(\theta; D_u)$：BS-T 在 forget set 上的总损失
- $\theta$：当前模型参数
- $D_u$：forget set，也就是需要遗忘的数据集合
- $\mathbb{E}_{D_u}[\cdot]$：对 forget set 中所有样本求平均
- $\sum_{i=1}^{|y_u|}$：对目标序列中每个 token 位置求和
- $|y_u|$：forget target 序列的长度
- $t_u^i$：第 $i$ 个位置上的 soft target
- $\log \pi_\theta(\cdot \mid x_u, y_u^{<i})$：当前位置整个词表上的 log 概率分布
- $\langle a, b \rangle$：向量内积，表示用 soft target 对 log 概率进行加权
- $x_u$：forget prompt
- $y_u^{<i}$：当前位置之前的前缀

### 这个公式的用途

这个公式是 BS-T 真正执行 unlearning 的目标函数。  
它的作用是把 forgetting pressure 从单个 target token 扩展到局部高概率 belief 区域，从而减少“换个近义 token 继续输出”的问题。

---

## 7. BS-S 的辅助 forget set

### 公式

$$
\hat{D}_u = \{ (x_u, \hat{y}_u^{(j)}) \}_{j=1}^{N}
$$

以及

$$
\hat{y}_u^{(j)} \sim \pi_\theta(\cdot \mid x_u)
$$

### 公式含义

这个公式表示：

对于每个 forget prompt $x_u$，模型会从当前分布中生成若干条高置信完整回答。  
作者把这些回答和原 prompt 配对，形成一个新的辅助 forget set，记作 $\hat{D}_u$。

### 字母解释

- $\hat{D}_u$：辅助 forget set，由模型自己生成的高置信序列构成
- $(x_u, \hat{y}_u^{(j)})$：第 $j$ 条辅助 forget 样本
- $x_u$：forget prompt
- $\hat{y}_u^{(j)}$：模型采样得到的第 $j$ 条高置信回答
- $j$：第几条采样序列
- $N$：每个 forget prompt 采样出的 belief sequence 数量
- $\pi_\theta(\cdot \mid x_u)$：当前模型在 prompt $x_u$ 下的输出分布
- $\theta$：当前模型参数

### 这个公式的用途

这个公式的作用是构造 **sequence-level 的 bootstrapped forget data**。  
作者不满足于只删原始答案，而是把模型当前最可能继续说出来的改写句子、整句续写也一起加入遗忘数据中。

---

## 8. BS-S 的总目标

### 公式

$$
L_{\mathrm{BSS}} = (1 - \lambda_{\mathrm{BSS}}) L(\theta; D_u) + \lambda_{\mathrm{BSS}} L(\theta; \hat{D}_u)
$$

如果写成优化目标，就是：

$$
\min_\theta L_{\mathrm{BSS}}
$$

### 公式含义

这个公式表示：

BS-S 的总损失由两部分组成：

1. 在原始 forget set $D_u$ 上做 unlearning
2. 在辅助 forget set $\hat{D}_u$ 上做 unlearning

然后通过 $\lambda_{\mathrm{BSS}}$ 来控制这两部分的权重。

也就是说，BS-S 不只是忘原答案，还会忘模型自己当前最可能生成的整条改写答案。

### 字母解释

- $L_{\mathrm{BSS}}$：BS-S 的总损失
- $\lambda_{\mathrm{BSS}}$：sequence-level bootstrapping 的混合系数
- $L(\theta; D_u)$：原始 forget set 上的 unlearning loss
- $L(\theta; \hat{D}_u)$：辅助 belief sequence forget set 上的 unlearning loss
- $D_u$：原始 forget set
- $\hat{D}_u$：模型生成的辅助 forget set
- $L$：这里可以是任意 unlearning loss，比如 GA、BS-T 等
- $\theta$：当前模型参数

### 这个公式的用途

这个公式的作用是解决 **sequence-level 的逃逸问题**。  
也就是：模型可能已经不输出原句了，但还是会输出整条语义很接近的改写句。BS-S 就是专门堵这个问题的。

---

## 9. 第四节所有关键公式的逻辑链条

第四节的方法链条可以按顺序理解成下面几步：

### 第一步：找到局部高概率 belief 区域

使用公式：

$$
H_k^{(i)} = \mathrm{Top}\text{-}k(\pi_\theta(\cdot \mid x_u, y_u^{<i}))
$$

作用：识别第 $i$ 个位置上最危险的高概率 token 区域。

### 第二步：构造 token-level 的 soft target

使用公式：

$$
t_u^i = \lambda_{\mathrm{BST}} \, \mathrm{sg}(\pi_\theta(\cdot \mid x_u, y_u^{<i})_{H_k^{(i)}}) + (1 - \lambda_{\mathrm{BST}}) e_{y_u^i}
$$

作用：把原始 target token 和高概率 belief token 混合起来。

### 第三步：在 token level 执行 belief-aware unlearning

使用公式：

$$
L_{\mathrm{BST}}(\theta; D_u) = \mathbb{E}_{D_u} \left[ \sum_{i=1}^{|y_u|} \langle t_u^i, \log \pi_\theta(\cdot \mid x_u, y_u^{<i}) \rangle \right]
$$

作用：同时压制 target token 和高概率 belief token。

### 第四步：采样 sequence-level 的高置信回答

使用公式：

$$
\hat{y}_u^{(j)} \sim \pi_\theta(\cdot \mid x_u)
$$

作用：找出模型当前最可能输出的完整改写答案。

### 第五步：构造辅助 forget set

使用公式：

$$
\hat{D}_u = \{ (x_u, \hat{y}_u^{(j)}) \}_{j=1}^{N}
$$

作用：把这些高置信改写回答加入遗忘数据。

### 第六步：在 sequence level 一起遗忘

使用公式：

$$
L_{\mathrm{BSS}} = (1 - \lambda_{\mathrm{BSS}}) L(\theta; D_u) + \lambda_{\mathrm{BSS}} L(\theta; \hat{D}_u)
$$

作用：同时删除原始答案和模型自己最可能生成的整条改写答案。

---

## 10. 第四节的总总结

第四节提出的核心方法是：

- 传统 unlearning 只压原始 target，容易导致概率质量逃到高概率改写表达上；
- 作者因此提出 bootstrapping-based unlearning；
- 在 token level，用 BS-T 同时压制 target token 和 top-k belief token；
- 在 sequence level，用 BS-S 进一步压制模型自己生成的高置信整句；
- 这样就能从局部 token 邻域和全局 sequence 邻域两个层面，同时对抗 squeezing effect。

一句话总结就是：

> 第四节的方法不是只让模型“不说原句”，而是让模型连“最可能换着说出来的那些句子”也一起忘掉。



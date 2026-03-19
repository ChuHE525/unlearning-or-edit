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











## 1. 局部 belief 的 top-k 集合

公式：

$$
H_k^{(i)} = \operatorname{Top}\text{-}k \big( \pi_\theta(\cdot \mid x_u, y_u^{<i}) \big)
$$

字母含义：

- $H_k^{(i)}$：第 $i$ 个位置上的 top-$k$ 高概率 token 集合
- $k$：取前多少个高概率 token
- $i$：当前 token 位置
- $\pi_\theta(\cdot \mid x_u, y_u^{<i})$：模型在给定输入和前缀条件下，对下一个 token 的条件概率分布
- $\theta$：当前模型参数
- $x_u$：forget prompt
- $y_u^{<i}$：目标序列在第 $i$ 个位置之前的前缀
- $\operatorname{Top}\text{-}k(\cdot)$：从概率分布中选出概率最高的前 $k$ 个 token

关键解释：

这个公式定义了当前位置的局部高概率区域。后续 token-level 的 bootstrapping 就是围绕这片高概率区域展开，而不是只盯住原始 target token。

---

## 2. 全局 belief 的序列采样

公式：

$$
\hat{y}_u \sim \pi_\theta(\cdot \mid x_u)
$$

字母含义：

- $\hat{y}_u$：模型基于 $x_u$ 生成的一条高置信完整序列
- $\pi_\theta(\cdot \mid x_u)$：模型在输入 $x_u$ 下，对所有可能输出序列的分布
- $\theta$：当前模型参数
- $x_u$：forget prompt
- $\sim$：表示“从该分布中采样得到”

关键解释：

这个公式定义了 sequence-level 的高置信回答。这些回答通常就是模型最可能生成的改写答案，后续会被加入辅助 forget set。

---

## 3. 定义 restricted belief distribution

为了让 GitHub 正常显示，把“限制在 top-k 集合上并重新归一化的分布”单独定义成：

$$
q_u^{(i)} = \operatorname{Renorm}_{H_k^{(i)}} \big( \pi_\theta(\cdot \mid x_u, y_u^{<i}) \big)
$$

字母含义：

- $q_u^{(i)}$：第 $i$ 个位置上，限制在 $H_k^{(i)}$ 内并重新归一化后的 belief 分布
- $\operatorname{Renorm}_{H_k^{(i)}}(\cdot)$：表示只保留 $H_k^{(i)}$ 中的 token，并把概率重新归一化
- $H_k^{(i)}$：第 $i$ 个位置上的 top-$k$ 高概率 token 集合
- $\pi_\theta(\cdot \mid x_u, y_u^{<i})$：原始 token 概率分布
- $\theta$：当前模型参数
- $x_u$：forget prompt
- $y_u^{<i}$：当前位置之前的前缀

关键解释：

这一步只是把论文里“restricted to top-k”的那部分单独记成一个新符号，数学含义不变，但 GitHub 更容易正常渲染。

---

## 4. BS-T 的 soft target

公式：

$$
t_u^{(i)} = \lambda_{\mathrm{BST}} \, \operatorname{sg}(q_u^{(i)}) + \big(1 - \lambda_{\mathrm{BST}}\big) e_{y_u^{(i)}}
$$

字母含义：

- $t_u^{(i)}$：第 $i$ 个位置上的 soft unlearning target
- $\lambda_{\mathrm{BST}}$：BS-T 的混合系数，用来控制 belief 分布所占权重
- $\operatorname{sg}(\cdot)$：stop-gradient，表示这一项只作为目标使用，不让梯度反向传回去
- $q_u^{(i)}$：限制在 top-$k$ 区域上的 belief 分布
- $e_{y_u^{(i)}}$：目标 token $y_u^{(i)}$ 的 one-hot 向量
- $y_u^{(i)}$：forget target 序列在第 $i$ 个位置上的 token
- $i$：当前 token 位置

关键解释：

这是 token-level bootstrapping 的核心。它把两部分混合成一个 soft target：

1. 原始目标 token  
2. 模型当前最可能逃去的高概率 token 分布

这样做的目的，是把 forgetting pressure 从“压一个点”扩展成“压一片局部高概率区域”。

---

## 5. BS-T 的 token-level loss

公式：

$$
L_{\mathrm{BST}}(\theta; D_u)
=
\mathbb{E}_{D_u}
\left[
\sum_{i=1}^{|y_u|}
\left\langle
t_u^{(i)},
\log \pi_\theta(\cdot \mid x_u, y_u^{<i})
\right\rangle
\right]
$$

字母含义：

- $L_{\mathrm{BST}}(\theta; D_u)$：BS-T 在 forget set 上的总损失
- $\theta$：当前模型参数
- $D_u$：forget set
- $\mathbb{E}_{D_u}[\cdot]$：对 forget set 中所有样本求平均
- $\sum_{i=1}^{|y_u|}$：对目标序列中每个 token 位置求和
- $|y_u|$：forget target 序列的长度
- $t_u^{(i)}$：第 $i$ 个位置上的 soft target
- $\log \pi_\theta(\cdot \mid x_u, y_u^{<i})$：当前位置整个词表上的 log 概率分布
- $\langle a, b \rangle$：向量内积
- $x_u$：forget prompt
- $y_u^{<i}$：当前位置之前的前缀

关键解释：

这个损失函数在每个位置上都会同时压制：

- 原始 target token
- 高概率 belief token

所以它实现的是 token-level 的 belief-aware unlearning。

---

## 6. BS-S 的辅助 forget set

公式：

$$
\hat{D}_u = \{ (x_u, \hat{y}_u^{(j)}) \}_{j=1}^{N}
$$

以及：

$$
\hat{y}_u^{(j)} \sim \pi_\theta(\cdot \mid x_u)
$$

字母含义：

- $\hat{D}_u$：辅助 forget set
- $(x_u, \hat{y}_u^{(j)})$：第 $j$ 条辅助 forget 样本
- $x_u$：forget prompt
- $\hat{y}_u^{(j)}$：模型采样得到的第 $j$ 条高置信回答
- $j$：第几条采样序列
- $N$：每个 forget prompt 采样出的 belief sequence 数量
- $\pi_\theta(\cdot \mid x_u)$：当前模型在 prompt $x_u$ 下的输出分布
- $\theta$：当前模型参数

关键解释：

这一步的作用是构造 sequence-level 的 bootstrapped forget data。除了原始 forget answer，还把模型自己最可能生成的完整改写回答一并加入遗忘数据。

---

## 7. BS-S 的总目标

公式：

$$
L_{\mathrm{BSS}}
=
\big(1 - \lambda_{\mathrm{BSS}}\big) L(\theta; D_u)
+
\lambda_{\mathrm{BSS}} L(\theta; \hat{D}_u)
$$

如果写成优化目标，就是：

$$
\min_\theta L_{\mathrm{BSS}}
$$

字母含义：

- $L_{\mathrm{BSS}}$：BS-S 的总损失
- $\lambda_{\mathrm{BSS}}$：sequence-level bootstrapping 的混合系数
- $L(\theta; D_u)$：原始 forget set 上的 unlearning loss
- $L(\theta; \hat{D}_u)$：辅助 belief sequence forget set 上的 unlearning loss
- $D_u$：原始 forget set
- $\hat{D}_u$：模型生成的辅助 forget set
- $L$：这里可以是任意 unlearning loss，比如 GA、BS-T 等
- $\theta$：当前模型参数

关键解释：

这个目标函数同时在两类数据上做遗忘：

1. 原始 forget data  
2. 模型自己生成的高置信改写 answers

它解决的是 sequence-level 的逃逸问题，也就是模型虽然不输出原句，但仍可能输出整条语义接近的改写句。

---

## 8. 公式之间的逻辑关系

### 第一步：定位局部高概率区域

$$
H_k^{(i)} = \operatorname{Top}\text{-}k \big( \pi_\theta(\cdot \mid x_u, y_u^{<i}) \big)
$$

作用：找到当前位置最危险的高概率 token 邻域。

### 第二步：构造 restricted belief distribution

$$
q_u^{(i)} = \operatorname{Renorm}_{H_k^{(i)}} \big( \pi_\theta(\cdot \mid x_u, y_u^{<i}) \big)
$$

作用：把局部高概率区域中的 token 分布单独提取出来。

### 第三步：构造 token-level soft target

$$
t_u^{(i)} = \lambda_{\mathrm{BST}} \, \operatorname{sg}(q_u^{(i)}) + \big(1 - \lambda_{\mathrm{BST}}\big) e_{y_u^{(i)}}
$$

作用：把原始 target token 和高概率 belief token 混合。

### 第四步：定义 token-level unlearning loss

$$
L_{\mathrm{BST}}(\theta; D_u)
=
\mathbb{E}_{D_u}
\left[
\sum_{i=1}^{|y_u|}
\left\langle
t_u^{(i)},
\log \pi_\theta(\cdot \mid x_u, y_u^{<i})
\right\rangle
\right]
$$

作用：同时压制 target token 和局部 belief token。

### 第五步：采样高置信整句

$$
\hat{y}_u^{(j)} \sim \pi_\theta(\cdot \mid x_u)
$$

作用：找出模型最可能输出的完整改写答案。

### 第六步：构造辅助 forget set

$$
\hat{D}_u = \{ (x_u, \hat{y}_u^{(j)}) \}_{j=1}^{N}
$$

作用：把这些高置信改写回答加入遗忘数据。

### 第七步：定义 sequence-level 总目标

$$
L_{\mathrm{BSS}}
=
\big(1 - \lambda_{\mathrm{BSS}}\big) L(\theta; D_u)
+
\lambda_{\mathrm{BSS}} L(\theta; \hat{D}_u)
$$

作用：同时删除原始答案和模型自己最可能生成的整条改写答案。

---

## 9. 总结

第四节的方法核心是：

- token level：通过 BS-T 压制原始 target token 以及其周围的高概率 belief token
- sequence level：通过 BS-S 压制原始 target sequence 以及模型自己最可能生成的高置信改写 sequence

最终目标不是只让模型“不输出原句”，而是让模型连“最可能换着说出来的那些答案”也一起忘掉。

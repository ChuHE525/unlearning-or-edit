# LLM UNLEARNING WITH LLM BELIEFS

* 这篇文章指出传统 unlearning 会把概率质量挤到语义相近的改写答案上，造成伪遗忘；为了解决这个问题，他们引入 model beliefs，并同时压制 target response 和 beliefs，从而更彻底地忘掉目标内容。
  现有 LLM unlearning 方法虽然能压低目标答案概率，但常常只造成“伪遗忘”，因为概率质量会被挤到语义相近的高概率改写上。为了解决这个问题，作者提出利用模型自身的高置信生成（model beliefs）作为辅助遗忘目标，构建 bootstrapping unlearning 框架。
  
* 已有 unlearning 方法只压低“原目标答案”还不够，因为概率质量会被挤到语义相近的高概率替代表达上，这就是 squeezing effect（挤压效应）；所以作者提出要把模型自己当前“最相信”的内容（model beliefs）也一起拿来忘掉。基于这个想法，他们设计了两个方法：BS-T（token-level） 和 BS-S（sequence-level）。 
更具体地说，先区分了两种“belief”：在 token 级别，模型在第 i个位置上的条件分布 πθ(⋅∣xu,y^u<i) 表示它的局部 belief；在 sequence 级别，从 πθ(⋅∣xu)采样出来、且平均对数似然较高的整个回答 y^u​ 表示它的全局 belief。BS-T 针对前者，BS-S 针对后者。


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

model beliefs

也就是模型当前最相信、最可能生成的 token 或 sequence。

提出一个：

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




### 旧方法只压原始 target，会让概率质量逃到模型自己本来就更相信的高概率改写区域；所以新方法不只要压原答案，还要压模型自己的高置信 beliefs，把它具体化成两个方法：BS-T 和 BS-S。

## 1. 局部 belief 的 top-k 集合

### 公式

<img width="342" height="63" alt="image" src="https://github.com/user-attachments/assets/46046d7e-41a0-4b5f-94c0-98942aa67cb1" />


### 字母含义

- \(H_k^{(i)}\)：第 \(i\) 个位置上的 top-k 高概率 token 集合  
- \(k\)：取前多少个高概率 token  
- \(i\)：当前 token 位置  
- \(\pi_\theta(\cdot \mid x_u, y_u^{<i})\)：模型在给定输入和前缀条件下，对下一个 token 的条件概率分布  
- 𝜃：当前模型参数  
- \(x_u\)：forget prompt  
- \(y_u^{<i}\)：目标序列在第 \(i\) 个位置之前的前缀  

### 关键解释

这个公式定义了当前位置的局部高概率区域。后续 token-level 的 bootstrapping，就是围绕这片高概率区域展开，而不是只盯住原始 target token。

把原始 target token 压下去时，概率质量最可能流向的，就是这批 top-k 高概率 token
---

## 2. 全局 belief 的序列采样

### 公式

```math
\hat{y}_u \sim \pi_\theta(\cdot \mid x_u)
```

### 字母含义

- \(\hat{y}_u\)：模型基于 \(x_u\) 生成的一条高置信完整序列  
- \(\pi_\theta(\cdot \mid x_u)\)：模型在输入 \(x_u\) 下，对所有可能输出序列的分布  
- θ：当前模型参数  
- \(x_u\)：forget prompt  
- ∼：表示“从该分布中采样得到”  

### 关键解释

这个公式定义了 sequence-level 的高置信回答。这些回答通常就是模型最可能生成的改写答案，后续会被加入辅助 forget set，不只是“下一个词”会逃，整条回答也会逃。模型很可能不再说原句，但会说一条高置信的完整改写句。

- token 级别有高概率邻域 𝐻𝑘(𝑖)

- sequence 级别有高置信整句 𝑦^u

- 要真正 unlearn，就不能只压 label token，而要把这两种 beliefs 一起纳入目标

---



## 3. BS-T 的 soft target
- BS-T 的目标是解决 token-level 的 squeezing effect

### 公式

```math
t_u^{(i)} =
\lambda_{\mathrm{BST}} \, \mathrm{sg}(q_u^{(i)})
+
(1 - \lambda_{\mathrm{BST}}) \, e_{y_u^{(i)}}
```

### 字母含义

<img width="830" height="407" alt="image" src="https://github.com/user-attachments/assets/df0fac70-e9ff-4576-a979-fd924057de18" />


### 关键解释
- “把 target token 和模型最想逃去的那些 top-k 高概率 token，一起压低。”

也就是说，遗忘信号不再只打在一个点上，而是扩展到了 target 附近那片语义相近、高概率的区域。
这正是用来对抗 squeezing effect 的。

这里 stop-gradient 也很关键。它的作用是把当前模型分布当成一个“固定 teacher target”，避免模型一边生成 belief、一边又直接对 belief 本身求梯度，导致训练不稳定。

- 这是 token-level bootstrapping 的核心。它把两部分混合成一个 soft target：

1. 原始目标 token  
2. 模型当前最可能逃去的高概率 token 分布  

这样做的目的，是把 forgetting pressure 从“压一个点”扩展成“压一片局部高概率区域”。

---

## 4. BS-T 的 token-level loss

### 公式

<img width="608" height="108" alt="image" src="https://github.com/user-attachments/assets/5318e217-c503-4254-891a-2c7b281583f0" />


### 字母含义

- \(L_{\mathrm{BST}}(\theta; D_u)\)：BS-T 在 forget set 上的总损失  
- \(\theta\)：当前模型参数  
- \(D_u\)：forget set  
- \(\mathbb{E}_{(x_u, y_u)\sim D_u}\)：对 forget set 中所有样本求平均  
- \(\sum_{i=1}^{|y_u|}\)：对目标序列中每个 token 位置求和  
- \(|y_u|\)：forget target 序列的长度  
- \(t_u^{(i)}\)：第 \(i\) 个位置上的 soft target  
- \(\log \pi_\theta(\cdot \mid x_u, y_u^{<i})\)：当前位置整个词表上的 log 概率分布  
- \(\langle a, b \rangle\)：向量内积  
- \(x_u\)：forget prompt  
- \(y_u^{<i}\)：当前位置之前的前缀  

### 关键解释
- 本质是一个 soft-label 的 token-level 目标,把 token-level 的 forgetting 从“单点压制”变成“局部邻域压制。
这个损失函数在每个位置上都会同时压制：

- 原始 target token  
- 高概率 belief token  

所以它实现的是 token-level 的 belief-aware unlearning,BS-T 会把 forgetting signal 分散到原始 target 和它的 top-k alternatives 上，从而在 token level 直接对抗 squeezing effect

---

## 5. BS-S 的辅助 forget set

### 公式

```math
\hat{y}_u^{(j)} \sim \pi_\theta(\cdot \mid x_u), \quad j = 1, \dots, N
```

```math
\hat{D}_u = \big( (x_u, \hat{y}_u^{(1)}), \dots, (x_u, \hat{y}_u^{(N)}) \big)
```

### 字母含义

- \(\hat{D}_u\)：辅助 forget set  
- \((x_u, \hat{y}_u^{(j)})\)：第 \(j\) 条辅助 forget 样本  
- \(x_u\)：forget prompt  
- \(\hat{y}_u^{(j)}\)：模型采样得到的第 \(j\) 条高置信回答  
- \(j\)：第几条采样序列  
- \(N\)：每个 forget prompt 采样出的 belief sequence 数量  
- \(\pi_\theta(\cdot \mid x_u)\)：当前模型在 prompt \(x_u\) 下的输出分布  
- \(\theta\)：当前模型参数  

### 关键解释

这一步的作用是构造 sequence-level 的 bootstrapped forget data。除了原始 forget answer，还把模型自己最可能生成的完整改写回答一并加入遗忘数据。

---

## 6. BS-S 的总目标

### 公式

```math
L_{\mathrm{BSS}}
=
(1 - \lambda_{\mathrm{BSS}}) \, L(\theta; D_u)
+
\lambda_{\mathrm{BSS}} \, L(\theta; \hat{D}_u)
```

```math
\min_{\theta} L_{\mathrm{BSS}}
```

### 字母含义

- \(L_{\mathrm{BSS}}\)：BS-S 的总损失  
- \(\lambda_{\mathrm{BSS}}\)：sequence-level bootstrapping 的混合系数  
- \(L(\theta; D_u)\)：原始 forget set 上的 unlearning loss  
- \(L(\theta; \hat{D}_u)\)：辅助 belief sequence forget set 上的 unlearning loss  
- \(D_u\)：原始 forget set  
- \(\hat{D}_u\)：模型生成的辅助 forget set  
- \(L\)：这里可以是任意 unlearning loss，比如 GA、BS-T 等  
- \(\theta\)：当前模型参数  

### 关键解释

这个目标函数同时在两类数据上做遗忘：

1. 原始 forget data  
2. 模型自己生成的高置信改写 answers  

它解决的是 sequence-level 的逃逸问题，也就是模型虽然不输出原句，但仍可能输出整条语义接近的改写句。

---

## 7. 公式之间的逻辑关系

### 第一步：定位局部高概率区域

<img width="372" height="65" alt="image" src="https://github.com/user-attachments/assets/b5891546-0f1a-43a7-92ed-0ec5fd1d19ed" />


**作用：** 找到当前位置最危险的高概率 token 邻域。

### 第二步：构造 restricted belief distribution

<img width="607" height="93" alt="image" src="https://github.com/user-attachments/assets/208060e9-de09-4b4f-9ed6-4e205ef22345" />



**作用：** 把局部高概率区域中的 token 分布单独提取出来。

### 第三步：构造 token-level soft target

```math
t_u^{(i)} =
\lambda_{\mathrm{BST}} \, \mathrm{sg}(q_u^{(i)})
+
(1 - \lambda_{\mathrm{BST}}) \, e_{y_u^{(i)}}
```

**作用：** 把原始 target token 和高概率 belief token 混合。

### 第四步：定义 token-level unlearning loss

<img width="673" height="99" alt="image" src="https://github.com/user-attachments/assets/c5c03c8f-978c-446b-9f6c-6e2030ed5400" />


**作用：** 同时压制 target token 和局部 belief token。

### 第五步：采样高置信整句

```math
\hat{y}_u^{(j)} \sim \pi_\theta(\cdot \mid x_u)
```

**作用：** 找出模型最可能输出的完整改写答案。

### 第六步：构造辅助 forget set

```math
\hat{D}_u = \big( (x_u, \hat{y}_u^{(1)}), \dots, (x_u, \hat{y}_u^{(N)}) \big)
```

**作用：** 把这些高置信改写回答加入遗忘数据。

### 第七步：定义 sequence-level 总目标

```math
L_{\mathrm{BSS}}
=
(1 - \lambda_{\mathrm{BSS}}) \, L(\theta; D_u)
+
\lambda_{\mathrm{BSS}} \, L(\theta; \hat{D}_u)
```

**作用：** 同时删除原始答案和模型自己最可能生成的整条改写答案。

---

## 8. 总结

方法核心是：

- **token level：** 通过 BS-T 压制原始 target token 以及其周围的高概率 belief token  
- **sequence level：** 通过 BS-S 压制原始 target sequence 以及模型自己最可能生成的高置信改写 sequence  

最终目标不是只让模型“不输出原句”，而是让模型连“最可能换着说出来的那些答案”也一起忘掉。










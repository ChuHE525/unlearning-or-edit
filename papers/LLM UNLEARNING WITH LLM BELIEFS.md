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




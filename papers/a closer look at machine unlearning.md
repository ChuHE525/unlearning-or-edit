# 《A Closer Look at Machine Unlearning for Large Language Models》LLM 遗忘的评估、范式与优化

- 论证了为什么现有 untargeted / targeted 都有结构性问题，再分别给出这两个对应方案：untargeted：ME+GD；targeted：IDK+AP。
  
## ME loss（Maximizing Entropy loss）
- 直接让它在 forget set 上对 next-token 预测变得高熵、接近均匀分布，也就是“不确定”。同时再用 GD 保持 retain set 上的正常性能，因此 ME+GD 就实现了“该忘的变不确定，不该忘的尽量保住”。
- ME公式为：

  <img width="656" height="102" alt="image" src="https://github.com/user-attachments/assets/962d6094-bbcb-41f5-8158-23f5eb611194" />
  
  1. 𝐷𝐹：forget set
  2. 𝜃：unlearned model 的参数
  3. (𝑥,𝑦)∼𝐷𝐹：从 forget set 中取出一个样本；x为输入的问题；y为对应的答案
  4. x′=x∘y：问题 x 和答案 y 拼在一起，目的是：不仅在答案部分计算 forget loss，也在问题部分计算 forget loss
  5. 𝑇：拼接后序列 𝑥′的 token 总长度
  6. 𝑡=1,…,𝑇：表示在这个完整序列里，逐个看第 𝑡 个 token
  7. Pt​=p(xt′​∣x<t′​;θ)：在已经看到前面 token 𝑥<𝑡的条件下，模型对第 𝑡 个 token 的预测概率分布，这是对整个词表的概率值分布；xt′：序列中的第 𝑡 个 token，x<t′：第 𝑡 个 token 前面的所有 token，p(xt′​∣x<t′​;θ)：模型给出的 next-token 概率分布
  8. U[k]:K 是 vocabulary size，也就是词表大小；整体表示定义在K大小词表上的均匀分布，均匀分布意味着每个token的概率都一样
  9. 𝐾𝐿(𝑃𝑡∥𝑈[𝐾])：是 KL divergence，衡量模型当前预测分布 𝑃𝑡与均匀分布 𝑈[𝐾]的差异，如果 𝑃𝑡很接近均匀分布，KL 就小；如果𝑃𝑡很尖锐、很偏向少数 token，KL 就大，所以最小化这个 KL，就是在逼模型：不要太确定下一个 token 是什么
  10. <img width="137" height="63" alt="image" src="https://github.com/user-attachments/assets/8f3a2c45-225a-4d2b-9eee-298629e1b3fa" />：表示对一个样本里所有 token 的 KL divergence 取平均，目的是为了不是只让某几个 token 不确定，而是希望整条 forget sample 上，模型的预测都更接近均匀分布
  11. 𝐸(𝑥,𝑦)∼𝐷𝐹[⋅]：对 forget set 上所有样本再取期望，也就是平均
      
- 整个 𝐿𝑀𝐸的含义就是：在 forget set 的所有样本上，把每个 token 的预测分布都尽量推向均匀分布。
- “最小化 KL 到均匀分布”就等价于“最大化熵”：
  
   公式为：<img width="483" height="60" alt="image" src="https://github.com/user-attachments/assets/04793d55-9ea9-4641-9a5d-d0d1d477a9d5" />

   1. 𝐻(𝑃𝑡)是分布 𝑃𝑡的熵；
   2. log𝐾是常数，因为词表大小 𝐾固定
- 因此，最小化 𝐾𝐿(𝑃𝑡∥𝑈[𝐾])等价于最大化 𝐻(𝑃𝑡)，使得模型的预测分布熵尽可能大。
- 熵可以理解成“分布的不确定性”：熵低：分布很集中，模型很自信；熵高：分布很分散，模型不确定。可以看出文章的思路为：对 forget set，不追求“答另一个具体内容”，而是让模型变得“不知道该答什么”。

### ME+GD
- 除了要解决如何在 forget set 上忘，还要求不能把 retain set 上本来会的东西也一起忘掉。
- 公式为：<img width="606" height="59" alt="image" src="https://github.com/user-attachments/assets/d342a4c7-28b8-4791-aa1a-051a836ff4a7" />
1. LME+GD​(θ)是最终总损失函数。训练时实际优化的就是它
2. α是一个 hyper-parameter，用来控制遗忘强度，α 大：更强调 forget set 上的 unlearning；α 小：更强调 retain set 上的 utility preservation，目的是控制“忘多少”和“保留多少”之间的平衡
3. ME 部分：把 forget set 上的 next-token 分布往均匀分布推
4. GD 部分：把 retain set 上的正确答案概率继续往上拉

## AP loss
- 解决targeted unlearning 容易“过度无知” 的问题，也就是模型不只在 forget set 上学会拒答，连 retain set 上本来会的问题也开始回答 “I don’t know”。
- 正则项应该同时满足两件事：降低拒答模板的概率，保持原始正确答案的概率，因此AP loss作用是：在 retain set 上防止模型把拒答行为学泛化
- 公式为：<img width="926" height="96" alt="image" src="https://github.com/user-attachments/assets/e49294bc-da16-4325-9964-372d8a0bb870" />
- 对 AP loss 的梯度分析，公式为：<img width="904" height="84" alt="image" src="https://github.com/user-attachments/assets/005fba87-03c4-4394-af45-60a85cce7bda" />
- IDK+AP，与IDK forget loss结合起来，为了在 forget set 上教模型拒答；在 retain set 上教模型别乱拒答。公式为：<img width="678" height="87" alt="image" src="https://github.com/user-attachments/assets/bf23f619-394c-4867-9000-e9c09673a018" />



## 普通语言模型微调（基于参数优化的 unlearning fine-tuning）
设模型参数是 θ，给定输入字符串 s，模型输出下一个 token 的概率分布记为 p(·|s; θ) 训练数据写成 D = {(x, y)} 其中：

x：问题 / 输入

y：答案 / 输出

- 普通微调的目标就是，最小化正确答案的预测损失：给定输入 x 和模型参数 θ 时，模型生成正确答案 y的概率越大，损失 ℓ就越小。
  
    - ℓ(y∣x;θ)=−logp(y∣x;θ)   损失 = 模型对正确答案分配的概率高低的负对数
    
    1) ℓ读作 loss（损失函数），损失越小，越倾向于正确答案
    2) 𝑦 表示 目标输出 / 正确答案 / 标签序列，就是指与输入x对应的标准答案
    3) x表示 输入，理解成 问题、提示词（prompt）或上下文
    4) θ 表示表示 模型参数 也就是 LLM 内部所有需要学习的权重；不同的 θ会对应不同的模型行为。
    5) p(y∣x;θ) 表示 在模型参数 θ下，给定输入 x时，模型生成目标输出 y 的条件概率。
    6) −log⁡是对数。这里 −log⁡p(y∣x;θ) 是机器学习里非常常见的 负对数似然（negative log-likelihood, NLL） 形式。

- 如果 p(y∣x;θ) 很大（模型很有把握地生成正确答案），那么 −log⁡p(y∣x;θ)就会很小；
如果这个p很小，损失就会很大。所以训练时最小化这个损失，本质上是在 提高正确答案的概率

- 模型看到输入 x后，越“相信”正确答案 y，损失就越小。所以训练目标就是不断调整 θ，让 p(y∣x;θ) 变大，fine-tuning 的过程就是去最小化这个 prediction loss
  
 而整个答案序列y的概率，是把每一步生成下一个 token 的概率连乘起来：

 - <img width="478" height="110" alt="image" src="https://github.com/user-attachments/assets/7426a408-8df7-4c4d-85d0-21e6e7080a11" />


   T 是输出y的token数量

   y_t 是第 t 个 token

   y<t 是第 t 个 token 之前的前缀（已经生成的token）

   ◦ 表示字符串拼接，就是把输入的x和已经生成的y<t接起来，作为下一步预测的条件

   g(s; θ) 表示模型最终生成出来的文本

   ### LLM unlearning 的目标
   给定一个原始训练集 D，其中：

需要被遗忘的数据子集叫 forget set，记作 D_F
​
剩余保留的数据叫 retain set，记作D_R = D \D_F​


机器遗忘（unlearning）的目标是：训练出一个新的模型 θu，使它同时满足两点：


1. 忘掉 D_F中的信息
即模型不再能够回忆、生成或依赖这些需要删除的内容。


2. 保留 D_R 上的能力
即模型在其余数据上的表现尽量不下降，保持原有性能和通用能力。

 #### retain set  
包含俩个部分： 
1. neighbor set：和 forget set 的数据分布很相似，但本身不是要被删除的内容

2. general knowledge：其他一般知识数据
   
- 区分 neighbor set 和 general knowledge，是为了分别衡量“局部保留能力”和“整体通用能力”，从而更准确地判断 unlearning 是否真的做到“该忘的忘掉，不该忘的保住”

#### 基于已有三个指标再新加三个指标
已有的：
1. ROUGE 只看字面重合，不看语义质量
2. Probability ：看模型对标准答案的预测概率有多高
3. Truth Ratio ：看模型更偏向正确答案，还是偏向错误答案
新增的：
1. Token Entropy (TE)：输出 token 的多样性
2. Cosine Similarity (CS)：模型在 unlearning 前后，输出语义变化有多大
3. Entailment Score (ES)：模型输出和标准答案之间的语义和事实上一不一致，统计答对的比例

   - 两个聚合指标：MU 和 FE
     1. Model Utility (MU)是在 retain set 上，把所有指标取 harmonic mean（调和平均），目的是为了只要某一个指标特别差，MU 就会被明显拉低，综合看多项指标：字面匹配，概率，对错偏好，token 多样性，语义稳定，事实正确
     2. Forget Efficacy (FE)是在 forget set 上算的，但 不包括 TE，因为 TE 不依赖 ground truth。作者先对其他指标取均值，再用 1 减去这个均值得到 FE，FE 越高，表示模型在 forget set 上，忘得越彻底

## untargeted unlearning
- 只要求模型别泄露 forget set 内容，但不指定它该怎么回答，也就是说，“不许答对”就够了，但可以答错、乱答、含糊答。
 - 对 untargeted unlearning，不用“逼近某个不确定目标”，而改用 最大化每个token的预测熵，让模型在 forget set 上保持高不确定性，从而减少幻觉风险。
 - 俩个baseline：
   1. Gradient Ascent, GA：是最直接的遗忘方法，在 forget set 上最大化 loss。
      - 公式：<img width="466" height="62" alt="image" src="https://github.com/user-attachments/assets/b57264eb-1e68-466c-afea-896c980980eb" />

  L_GA：GA 的优化目标
  
  D_F：forget set，要遗忘的数据
  
  θ：模型参数，LLM内部可训练的权重
  
  E：平均值，表示期望
  
  -：让ℓ(y∣x;θ)在这个公式中表示反向优化，让模型忘掉这些数据是对 forget set 进行“反训练”：原本想让模型更会这个答案，现在想让模型更不会这个答案让模型在 forget set 上越来越答不好
  
  (x,y)∼DF(x,y)：从 forget set D_F 中取样本对
  
  x：输入
  
  y：正确答案
  
  [ℓ(y∣x;θ)]:对每个 forget 样本，计算它的预测损失

  2. NPO 是把原答案当成“不该被偏好”的负样本，按照偏好优化的方式、相对参考模型地把它压下去。
  

## targeted unlearning
- 明确要求模型在 forget set 上给出某种指定回答，比如拒答模板：“Sorry, I don’t know.”
- AP loss 通过保留 retain set 上原答案的概率，避免模型把“拒答模板”泛化到正常问题上
- retain set 需要被模型保留的训练数据子集，是forget set的补集
- Targeted 的两个 baseline：
  1. IDK Fine-tune (IDK)
     
   把 forget set 上的问题重新标注成拒答模板，比如：“I don’t know.”   “Sorry, I don’t know.
  
  2. Direct Preference Optimization, DPO

DPO 则把 forget set 原答案当作负样本，把拒答模板当正样本，直接做 偏好优化preference optimization

### regularization（正则化）
1. GD：Grad Descent

GD 就是在 retain set 上继续做普通训练，一边在 forget set 上做“遗忘”，一边在 retain set 上做“记住”，GD 是“继续学 retain set 正确答案”。

2. KL：Kullback-Leibler Divergence

   要求 unlearned model 在 retain set 上的输出分布，不要偏离 reference model 太远，KL 是“别偏离原模型太多”

### 结果
- LLM unlearning 的难点在于评估不完整、untargeted 目标不清晰、targeted 容易过度无知；对应地，ME 与 AP 分别提供了更合理的 untargeted / targeted 解决方案。


   

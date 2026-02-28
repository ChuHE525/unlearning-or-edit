
**Decoder-only LLM = 自回归模型（autoregressive）：
当给模型一句话的时候，会不断预测下一个 token 是什么并给每一个候选一个概率，挑一个token拼上去完成续写，接着再继续预测下一个token，这个叫做生成。**

* 模型通过token id进行续写,文本先被 tokenizer（分词器） 切成 token，再变成 整数 id 序列X1 X2....XT.

* Embedding（查表）：token id → 向量
<img width="168" height="42" alt="屏幕截图 2026-02-27 173126" src="https://github.com/user-attachments/assets/1c084a86-616c-4e49-b8dd-e9d2d5efc0b8" />

 这个模型的输入不是一串数字，而是一块三维数组（张量），形状大小是BxTxd.
1.  公式中，X是指经过查表之后的输入
2.  R是指实数，整体表示的是一个实数数组。
3.  B是指Batch size(一次给模型输入多少条句子)，T是指每句话中token的数量；d是指每一个token变成的向量是多少维，每一个token被翻译为多少个数字，d越大表达能力越强，但是会更耗算力。
embedding 就像给每个 token 一个“坐标”，方便做数学运算。文字经过 tokenizer+embedding 变成 
𝑋，这是神经网络可计算的输入。

- 位置编码用于告诉模型 token 的顺序

常见方式：token embedding + position embedding

现代 LLM 常用 RoPE（作用在 Q/K，使注意力感知相对位置）

* Decoder Block（重复 N 次的积木）
   整个模型需要多层 decoder block 堆叠；每层对表示做一次更新。
   每一个block需要做：他有俩干活的，叫做：
1.    Attention（注意力）：让每个 token 去“看”左边别的 token，把重要信息拿过来，就是融合上下文信息。
2.    MLP（前馈网络）：对每个 token 自己的向量做更复杂的非线性变换（像“加工/提炼”）
   还有俩个保证安全的，让训练不会崩掉的东西，
1.  Residual（残差）：把输入加回来，防止信息/梯度断掉
2.  Norm（归一化）：把数值规模稳定住，防止越算越爆炸或太小
*     Norm → Attention → Residual
    𝐻1=𝐻+Attention(Norm(𝐻))  其中H的意思是，一堆token向量
    这个公式表示的意思是：先norm，再attention，然后residual加回去。
*     Norm → MLP → Residual
    H2=H1+MLP(Norm(H1))
    这个公式的意思是，接着上个公式的H1继续，先norm，再mlp，再residual
    分开，Attention = 跨 token 信息融合；MLP = token 内部非线性加工。
*     其中residual这一步的意思是，H new=H old+Δ ，这个Δ，表示的是子层的输出 
 Residual = 输入 + 子层输出（学增量）作用：信息不丢、梯度好传、训练稳定
*  Norm：稳定数值尺度，让深层训练更稳定。把每一个token的向量数值尺度，统一到差不多的范围。
*  总结：一层 decoder block 可以读成：

对每个 token：
先把数值调稳定（Norm），再去看左边上下文（Attention），把结果加回自己（Residual）；
再把数值调稳定（Norm），再自己深加工（MLP），把结果再加回自己（Residual）。

Attention
   attention 是在一个句子之中做指代关联和信息聚合。（就是在一句话中，我应该重点关注左边哪些token）
   他主要做的是：对于每一个位置的向量，Q（发问 我想找什么） K（我有什么信息可以匹配） V（ 提供内容 如果你关注我，我应该拿走什么信息） 
   在匹配的时候，当前位置的Xt会拿自己的qt去和每个位置的Xj的kj做相似度，共公式为：
   <img width="286" height="80" alt="屏幕截图 2026-02-28 202944" src="https://github.com/user-attachments/assets/4c11c1c9-af0f-42ca-9612-0c382cc853c3" />
   在这个公式里，score越大，

   



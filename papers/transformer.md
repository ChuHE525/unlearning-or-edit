## Transformer核心架构：

- Transformer 是一个完全依赖自注意力来计算其输入和输出表示的 transduction 模型

### 注意力关键技术组件：
- 自注意力（Self-Attention）：自注意力，也称为内部注意力，是一种将单个序列的不同位置关联起来以计算该序列表示的注意力机制
- 缩放点积注意力（Scaled Dot-Product Attention）：这是计算的核心。通过引入缩放因子 ，防止在向量维度较大时点积过大导致 Softmax 函数梯度消失的问题。
- 多头注意力（Multi-Head Attention）：这是 Transformer 的精髓。它通过多个并行运行的注意力层，让模型能够同时从不同的表示子空间关注到不同位置的信息，增强了模型的表达能力。

### 支撑模型运作的关键组件:
- 位置编码（Positional Encoding）：由于模型没有循环结构，无法感知位置。作者通过加入正弦和余弦函数的位置编码来注入词汇在序列中的相对或绝对位置信息。
- 掩码机制（Masking）：在解码器的自注意力层中，通过掩码防止模型在预测当前位置时“偷看”到未来的信息，（Q和K的点积变为负无穷），确保自回归属性。
- 残差连接（Residual Connections）与层归一化（Layer Normalization）：每个子层（如注意力层和前馈网络）都包装在残差连接中（会把这一层的输入直接加回这一层的输出），随后进行层归一化（先算出当前层输出的平均值和标准差，然后把每个输出值减去平均值，再除以标准差，这样就把输出“归一”到一个标准分布，然后再乘以一个可训练的缩放因子并加上一个位移因子），这有助于训练更深的网络

### 训练与性能优势
- 高度并行化：Transformer 允许更多的并行计算，通过自注意力机制，一次性看整个序列，各个位置可以同时计算
- 长距离依赖：在 Transformer 中，任意两个位置之间的交互步数是常数 O(1)。

### 训练技巧
- 学习率调度（Warmup）：模型使用 Adam 优化器（自动调整，会追踪每个参数过去梯度的指数加权平均，并同时根据每个参数梯度的方差来调整每个参数的步长），并结合了先线性增加、后按反平方根减小的学习率策略。
- 正则化技术：使用了**丢弃法（Dropout）和标签平滑（Label Smoothing）**来防止过拟合并提高模型准确性。

### Transformer 允许显著更多的并行计算；
- 在 Transformer 中，这一操作数量（关联任意两个输入或输出位置的信号所需的操作数量）被减少到恒定值，代价是由于对注意力加权位置进行平均而导致有效分辨率降低，通过多头注意力来抵消这种影响。

- 在 Transformer 编码器中，self - attention 是核心组件。输入经线性变换得到查询、键、值向量，通过计算注意力分数，能捕捉序列全局依赖，且可并行计算。

### Encoder & Decoder 结构

- 模型由编码器（Encoder）和解码器（Decoder）两大部分组成。编码器将输入序列转换为连续表示，解码器则负责生成输出序列，
- transformer 总体结构为 Encoder - Decoder 加堆叠层

一、Encoder（编码器）

- Encoder是Transformer的编码部分，它把输入序列x编码成z。由6层堆叠而成，每层有多头自注意力子层和前馈网络子层，通过残差连接和层归一化，统一输出维度，能捕捉序列全局信息 。

- 多头自注意力子层通过多个“头”并行计算自注意力，捕捉序列不同子空间依赖关系，再拼接投影。前馈网络子层是两层全连接网络，对每个位置向量独立做“升维 - 非线性激活 - 降维”处理.

- 残差连接是把输入直接与子层输出相加，能缓解梯度消失，让信息跨层传递；LayerNorm 则对相加后的结果中每个样本的所有特征维度归一化，使均值为 0、方差为 1，稳定训练过程，

- 公式LayerNorm(x + SubLayer(x)) 表示将子层输入x和子层输出相加后进行层归一化。统一为512是为了使各层输出维度一致，方便残差连接计算和模型后续处理.

  - 维度统一：所有子层及嵌入层的输出维度均为dmodel=512，以适配残差连接。

二、Decoder（解码器）

- 整体结构：同样由N=6个完全相同的层堆叠组成。

- 每层子层：在编码器2个子层的基础上，新增1个子层，共3个子层：    

  - 子层1：自注意力机制（经修改）；

  - 子层2：对编码器堆叠输出执行的多头注意力机制；

  - 子层3：与编码器一致的基于位置的全连接前馈网络。

- 关键设计：
        
  - 同编码器：每个子层均有残差连接+层归一化；

  - 掩码机制：修改自注意力子层，防止位置关注后续位置；

  - 预测约束：掩码操作+输出嵌入偏移1个位置，确保位置i的预测仅依赖i之前的已知输出。

 - Decoder 每层比 Encoder 多一层，Decoder 有6层，每层比Encoder多带掩码多头自注意力层，还有使它关注Encoder输出的层及FFN，各子层有残差连接和LayerNorm，自注意力层需掩码保证自回归特性(确保第 i 个位置只能依赖 i 之前的输出) 。
- 在此结构中，编码器将符号表示的输入序列（x1，……，xn）映射为连续表示的序列 z =（z1，……，zn）。给定 z 后，解码器会逐个生成符号的输出序列（y1，……，ym）。在每一步中，该模型都是自回归的，在生成下一个符号时，会将之前生成的符号作为额外输入。Transformer 采用了这种整体架构，编码器和解码器均使用堆叠的自注意力机制以及逐点的全连接层。

  

### 注意力机制可描述为：将查询向量与一组键值对向量映射为输出向量，输出通过加权求和计算得到，其中查询、键、值、输出均为向量。
####  Scaled Dot-Product Attention  

- 目的：当前位置（**Query**）应该从哪些位置（**Key/Value**）“拿信息”，每个位置拿多少。
  - **Query（Q）**：我现在想找什么信息（“问题”）
  - **Key（K）**：我能用什么特征被匹配（“标签/索引”）
  - **Value（V）**：如果你关注我，你应该拿走什么内容（“内容”）
  - **dk** :描述q和k向量有多少元素（维度）
  - 用 **Q** 和所有 **K** 算相似度 → 得到权重 → 用权重对 **V** 加权求和，得到当前位置的新表示。

  <img width="614" height="147" alt="image" src="https://github.com/user-attachments/assets/cee08dd7-dae8-4424-9860-2f10d0a8f4ed" />

- 输入Query与Key - Value，先算  
  <img width="82" height="37" alt="image" src="https://github.com/user-attachments/assets/db98f55f-0745-4f3d-9b15-3b8d9802c793" />

- 得点积相似度，除以  
<img width="63" height="63" alt="image" src="https://github.com/user-attachments/assets/af62ca02-985a-4def-9985-364efe2a0ed1" />

- 缩放，经softmax得权重，再乘Value得输出。

- 除以  
  <img width="63" height="63" alt="image" src="https://github.com/user-attachments/assets/af62ca02-985a-4def-9985-364efe2a0ed1" />

 是因当  
  <img width="36" height="46" alt="image" src="https://github.com/user-attachments/assets/4956568b-c8cf-4b84-ac27-fe27c761dae5" />

大时，点积数值幅度变大，softmax梯度小训练不稳，缩放可稳定训练。

  ####  Multi-Head Attention
   - 多头注意力 = “多头注意力”允许模型从不同的子空间、不同的位置同时提取信息。每个头独立关注不同层面的关系，不会像单头那样简单平均，能更好保留细粒度信息。最终将多个头的输出拼接，再投影回原始维度，得到丰富且多样的表示。
   - 投影成 h 份”就是给 Q、K、V 各自乘不同的可学矩阵，把同一个输入向量变成 h 个不同版本。这样每个“头”就能关注不同的特征，一起提供丰富的视角。
   - 四步做法：
     1. 把输入的token向量，同一个 token 的向量表示，会被复制给所有的头。每个头对同样的 token 输入进行不同的线性变换，分别得到自己的 Q、K、V。所以，是每个 token 的向量会被所有的头共同处理，每个头只是从不同角度去关注它。 在同一个 token，在不同头里会变成不同表示方式。
     2. 第二步就是每个头各自做一次缩放点积注意力。每个头用自己的 Q、K、V，把输入 token 的信息，经过自己的权重和注意力计算，得到一个独立的输出向量。这个向量捕捉了这个头关注的信息，随后这些头的输出会被拼接起来，用来生成最终的结果。
     3. concat，就是把所有头独立计算出来的输出向量在特征维度上拼起来，被整合到了一个长向量中。之后，我们再通过一个线性变换把这个拼接后的向量压缩回原始的维度，这样就既保留了多头信息，又保持了统一的维度结构。
     4. 拼接完后再乘一个输出矩阵 𝑊𝑂，让不同头的信息“混合”一下，形成最终输出再投影回到d_model（d_model 就是整个模型使用的隐藏层维度，也就是每个 token 表示最终被投射回的固定维度大小），
     5. 公式为：
        
        <img width="632" height="67" alt="image" src="https://github.com/user-attachments/assets/f4ecc660-a259-43e2-a5e6-485f87548145" />
    
        <img width="394" height="107" alt="image" src="https://github.com/user-attachments/assets/fb35cc7f-2ec0-414d-9d8e-b36a9765749c" />

        每一个head为：
        <img width="459" height="54" alt="image" src="https://github.com/user-attachments/assets/8ba521ed-d838-45fb-9661-a93ebd2061ff" />

        <img width="471" height="84" alt="image" src="https://github.com/user-attachments/assets/4a9509c4-bdac-4591-94d6-aee5fc146283" />
     6. 输入维度：d_model​
        每个头的维度：dk​ 或 dv​
        拼接后维度：h⋅dv​
        最后 WO​ 把它变回 d_model
        ​
#### Applications of Attention in our Model
Transformer在三个位置用到multihead attention：

编码器 - 解码器注意力（Encoder-Decoder Attention / Cross-Attention）
来源：Queries（Q）来自解码器上一层的输出，Keys（K）和 Values（V）来自编码器的最终输出（memory）。
作用：decoder 用 Q 去问 encoder 的 K/V（从输入序列中取信息），decoder 的每个位置，都可以“看”输入序列的所有位置。

编码器自注意力（Encoder Self-Attention）
来源：Q、K、V 三者都来自编码器上一层的输出。
作用：输入序列内部互相看（Q=K=V 都来自 encoder），输入句子内部各 token 互相“交流”，形成更好的输入表示。

解码器掩码自注意力（Decoder Masked Self-Attention）
来源：Q、K、V 三者都来自解码器上一层的输出，必须加上mask，每个位置只能看自己和左边（≤当前位置），不能看右边未来 token
作用：允许解码器中的每个位置只关注到它自己以及它之前的所有位置，而看不到未来的位置。
实现：通过在 softmax 输入前，将对应未来位置的注意力分数设置为 -∞（即掩码），从而在训练时保持模型的自回归特性，确保生成过程是从左到右依次进行的。
GPT 只有 decoder，masked self-attention（第三种）。​

### Position-wise Feed-Forward Networks (FFN)

 - 全连接前馈网络 FFN（也就是 MLP），它对每个位置（每个 token）单独做同样的非线性变换，只做token内部的加工，不做token之间的交流。
 - FFN (x)=max (0,xW₁+b₁) W₂+b₂
 - 令u=xW₁+b₁
   x是指一个token的向量，长度为dmodel
   W₁是指把它变为更长的向量，长度为dff
   b₁是指偏置，长度和dff一样
   括号内部的作用是把x的长度dmodel扩展到dff
 - 令h=max (0,xW₁+b₁) 这个也可以叫做ReLU激活，目的是负数变0，正数不变。
 - 最后回到整个公式可以看： W₂是要降低回到原来的维度dmodel，b₂和dmodel的维度一样。
 - FFN 公式 等价于 两次 kernel size=1×1 的卷积
 ### Embeddings and Softmax  
- Transformer 用 embedding 把 token 变成向量（维度dmodel），再用 线性层 + softmax 把 decoder 的输出变成“下一个 token 的概率”

  这句话指的是，输入和输出的token都会变成向量，每一个token的长度是dmodel，（token id 是离散符号，神经网络需要连续实数向量才能计算。
embedding 就是一张可训练的表：每个 token 对应一行向量。）

Decoder 每个位置最后会输出一个隐藏向量：ht 维度为dmodel,为了预测下一个 token，需要把它映射到词表大小V,

线性层：
<img width="277" height="54" alt="image" src="https://github.com/user-attachments/assets/103c77ad-aa7f-4c7c-a332-100f97edc2d5" />

其中W形状就是dmodel X V.

SOFTMAX:
<img width="418" height="62" alt="image" src="https://github.com/user-attachments/assets/f00f48cb-200f-44b4-98ab-4abf2a83a7c3" />
目的是把logits（每个候选token的打分）变为概率。
- Transformer 类模型中输入 Embedding、输出 Embedding 与预 Softmax 线性层权重共享
- 在做embedding时，整体要乘一个根号下的d modle（做尺度调整，让训练更稳定）
  这个E【Xt】是指：embedding 查表；用 token id Xt 取出 embedding 矩阵 E 的对应行，得到该 token 的 d 维向量

  <img width="432" height="66" alt="image" src="https://github.com/user-attachments/assets/e3a051a0-9964-484e-81b0-83efe87b768d" />

  ### Positional Encoding（位置编码）
   - embedding 和PE是同一个维度（dmodel），可以直接相加到encoder/decoder底部输入embedding上,Xt是指这个token的完整形式，位置信息+语义信息。
     
     <img width="407" height="62" alt="image" src="https://github.com/user-attachments/assets/2edd4bb4-7bf4-409d-b1a1-c8e5642113a4" />


  <img width="427" height="127" alt="image" src="https://github.com/user-attachments/assets/394dfcb3-0ad9-413c-be8f-2b3b497ee8c4" />
  


     <img width="500" height="304" alt="image" src="https://github.com/user-attachments/assets/3b61dffb-5a93-4ddc-9ea1-685573d90b92" />
     

采用固定 sin/cos：偶数维 sin，奇数维 cos，不同维度不同频率（波长从 2π 到 10000⋅2π），选择 sin/cos 因为便于模型利用相对位置，并可能外推到更长序列。




   

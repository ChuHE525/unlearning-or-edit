

## 1. Decoder-only LLM = 自回归模型（autoregressive）

**模型是一步步地预测下一个 token。你给它一段输入，它会先根据这段输入预测下一个 token，然后把这个预测出来的 token 接到输入后面，再继续预测下一个 token。这个过程不断重复，直到生成完整的文本。**  
（训练时，模型一次性看到完整的输入，通过对比预测和正确答案，并行优化参数。而微调（fine-tune）是指在预训练后，用特定任务的数据再训练，让模型更适应某些具体场景。推理就是模型生成阶段，你输入一个 prompt，它一步步续写。所以选择顺序是：先训练，然后根据需要微调，最后推理生成答案。）

---

## 2. Tokenizer 与 Token id

- 模型通过 token id 进行续写：文本先被 tokenizer（分词器）切成 token，再变成整数 id 序列 **X1 X2....XT**。

---

## 3. Embedding（查表）：token id → 向量

<img width="168" height="42" alt="屏幕截图 2026-02-27 173126" src="https://github.com/user-attachments/assets/1c084a86-616c-4e49-b8dd-e9d2d5efc0b8" />

这个模型的输入不是一串数字，而是一块三维数组（张量），形状大小是 **B×T×d**。

1. 公式中，**X** 是指经过查表之后的输入  
2. **R** 是指实数，整体表示的是一个实数数组  
3. **B** 是指 Batch size（一次给模型输入多少条句子）  
4. **T** 是指每句话中 token 的数量  
5. **d** 是指每一个 token 变成的向量是多少维：每一个 token 被翻译为多少个数字  
   - d 越大表达能力越强，但是会更耗算力  

embedding 就像给每个 token 一个“坐标”，方便做数学运算。文字经过 tokenizer+embedding 变成 **X**，这是神经网络可计算的输入。

---

## 4. 位置编码（Position Encoding）

- 位置编码用于告诉模型 token 的顺序  
- 常见方式：token embedding + position embedding  
- 现代 LLM 常用 **RoPE**（作用在 Q/K，使注意力感知相对位置）

---

## 5. Decoder Block（重复 N 次的积木）

整个模型需要多层 decoder block 堆叠；每层对表示做一次更新。每一个 block 需要做：

### 5.1 两个“干活”的模块
1. **Attention（注意力）**：让每个 token 去“看”左边别的 token，把重要信息拿过来，就是融合上下文信息  
2. **MLP（前馈网络）**：对每个 token 自己的向量做更复杂的非线性变换（像“加工/提炼”）

### 5.2 两个“保证安全”的模块（防止训练崩）
1. **Residual（残差）**：把输入加回来，防止信息/梯度断掉  
2. **Norm（归一化）**：把数值规模稳定住，防止越算越爆炸或太小  

---

## 6. Block 内部公式与含义

### 6.1 Norm → Attention → Residual
\[
H1 = H + Attention(Norm(H))
\]
其中 **H** 的意思是：一堆 token 向量。  
这个公式表示的意思是：先 norm，再 attention，然后 residual 加回去。

### 6.2 Norm → MLP → Residual
\[
H2 = H1 + MLP(Norm(H1))
\]
这个公式的意思是：接着上个公式的 H1 继续，先 norm，再 mlp，再 residual。

分开理解：**Attention = 跨 token 信息融合；MLP = token 内部非线性加工。**

---

## 7. Residual 与 Norm 的理解

### 7.1 Residual
Residual 这一步的意思是：
\[
H_{new} = H_{old} + \Delta
\]
这个 \(\Delta\) 表示的是子层的输出。  

- Residual = 输入 + 子层输出（学增量）  
- 作用：信息不丢、梯度好传、训练稳定  

### 7.2 Norm
- Norm：稳定数值尺度，让深层训练更稳定  
- 把每一个 token 的向量数值尺度，统一到差不多的范围  

---

## 8. 一层 decoder block 的总结（人话版）

对每个 token：  
先把数值调稳定（Norm），再去看左边上下文（Attention），把结果加回自己（Residual）；  
再把数值调稳定（Norm），再自己深加工（MLP），把结果再加回自己（Residual）。

---

## 9. Attention（注意力）细节

attention 是在一个句子之中做指代关联和信息聚合。  
（就是在一句话中，我应该重点关注左边哪些 token）

它主要做的是：对于每一个位置的向量，  
- **Q（发问：我想找什么）**  
- **K（我有什么信息可以匹配）**  
- **V（提供内容：如果你关注我，我应该拿走什么信息）**

### 9.1 Q / K / V：对同一个 \(h_t\) 的不同线性投影
- 第 t 个 token 当前向量叫 \(h_t\)（hidden state）  
- 用参数矩阵投影成：
\[
q_t = h_t W_Q,\quad k_t = h_t W_K,\quad v_t = h_t W_V
\]
这里 \(W_Q, W_K, W_V\) 是模型训练出来的参数。

### 9.2 相似度打分（score）
当前位置的 \(X_t\) 会拿自己的 \(q_t\) 去和每个位置的 \(X_j\) 的 \(k_j\) 做相似度，公式为：

<img width="286" height="80" alt="屏幕截图 2026-02-28 202944" src="https://github.com/user-attachments/assets/4c11c1c9-af0f-42ca-9612-0c382cc853c3" />

在这个公式里，score 越大，相关性越强，权重越高，减少偏差。

### 9.3 Causal mask（不能看未来）
- Causal mask：位置 t 只能关注 ≤t 的 token，不能看未来  
- 会加一个因果 mask：把未来位置的 score 设为负无穷  
- softmax 后这些位置权重变 0  

### 9.4 softmax：把分数转化为权重
接下来 attention 做两方面：第一步为 softmax：就是把分数转化为权重，公式：

<img width="364" height="73" alt="屏幕截图 2026-02-28 213056" src="https://github.com/user-attachments/assets/50f4bada-a764-4c02-88ab-d4a7a8860a59" />

<img width="800" height="264" alt="屏幕截图 2026-02-28 213241" src="https://github.com/user-attachments/assets/f5fa2cea-61cd-492f-8173-caead2c4347e" />

对固定的 t（也就是 score 矩阵第 t 行），做 softmax 得到一行权重。其中 \(\alpha_{t,j}\) 是这一行里的第 j 个权重。  
对于固定的 t，模型要决定：“位置 t 要把注意力分配给哪些位置 j？”  
所以它必须把一行分数变成一行权重。

softmax 的展开式：

<img width="377" height="151" alt="屏幕截图 2026-02-28 213918" src="https://github.com/user-attachments/assets/e0bf3f38-b955-40f5-9c45-ac7adc3102ab" />

<img width="856" height="164" alt="屏幕截图 2026-02-28 214057" src="https://github.com/user-attachments/assets/16abd655-b588-474f-bfcd-74337a48ef83" />

要用 exp？（e 的多少次方）

- softmax 的性质：\(\alpha_{t,j} \ge 0\)  
- 让权重都变成正数（注意力权重不能是负的）  
- 拉开差距：分数稍微大一点，exp 后会明显更大 → 更突出“更相关的位置”  

为什么要除以那个求和？
- 为了让所有权重加起来等于 1（变成“分配比例”）

### 9.5 加权求和得到输出 \(o_t\)
对每个位置 t，它会把所有位置 j 的 value 向量 \(v_j\) 按权重 \(\alpha_{t,j}\) 混合起来，得到一个新的向量 \(o_t\)。  
（\(o_t\) 就是“第 t 个 token 融合了左侧上下文后的向量表示”。）

<img width="275" height="75" alt="屏幕截图 2026-02-28 214715" src="https://github.com/user-attachments/assets/b81198a4-ac84-4439-91d5-94e96dd4614f" />

<img width="704" height="120" alt="屏幕截图 2026-02-28 214904" src="https://github.com/user-attachments/assets/93087189-668b-4be7-92c8-c8b30a7b9ee9" />

---

## 10. MLP（前馈网络）

MLP：输入向量先经过一个线性变换矩阵相乘（也就是乘以一个权重矩阵再加上偏置 bias），然后通过一个非线性激活函数（比如 ReLU 修正线性单元砍掉所有负数、GELU 高斯误差线性单元高斯分布曲线小的更小大的更大等）增加表达能力。

- Attention：跨 token 融合上下文（加权求和）  
- MLP：逐 token 的非线性变换（提炼/组合特征）  
- 结构：\(d \rightarrow d_{ff}(\approx 4d) \rightarrow d\)，中间有激活（GELU/SiLU）

---

## 11. logits 与“续写”

logits 是由模型最后一层的隐藏表示通过一个线性层（也就是矩阵乘法）得来的。具体地说，模型用最后层每个 token 的隐藏向量，乘以一个投影到词表大小的权重矩阵。这样，对于每个位置，都会得到一个对词表中每个可能下一个 token 的打分，这些打分就是 logits。之后通过 softmax，这些 logits 才被转换成概率分布。

“续写”指的是生成式模型在预测下一个 token 的任务。当你给模型一个序列，它会尝试根据前面的内容“续写”下一个合理的 token。所以在推理（生成）时，我们特别关注最后一个位置的 logits，因为它直接对应下一个要预测的 token。

---

## 12. KV Cache（推理加速）

KV cache 让每步生成不需要重复算历史 token 的 K/V，大幅减少重复计算。

KV Cache 是一种推理时加速的机制。在生成文本时，模型每一步都会计算当前 token 的 Query（Q）、Key（K）和 Value（V）。但之前生成的 token 的 K 和 V 是不会变的。KV Cache 就是把历史 token 的 K 和 V 缓存起来，这样每生成一个新 token，只需要算新 token 的 K、V 和 Q，再与已缓存的 K、V 结合即可。这样就避免重复计算历史部分，让生成更快。

---

## 13. 训练 vs 推理

### 13.1 训练（teacher forcing）
- 有完整句子 → 一次算完所有位置（并行）  
- 靠 causal mask 防止偷看未来  

### 13.2 推理（生成）
- 没有答案 → 只能一个 token 一个 token 生成（串行）  
- 常用 KV cache 加速（缓存历史 K/V）  

- **训练并行（teacher forcing），推理串行（自回归生成）。**

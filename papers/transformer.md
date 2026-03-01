**提出了一种名为 Transformer 的新型网络架构，旨在解决序列转导（sequence transduction）问题，如机器翻译**

核心架构：
 提出了一种完全基于注意力机制（Attention mechanisms）的架构，彻底放弃了传统的循环和卷积结构。
 
  Transformer 允许显著更多的并行计算；
  
  在 Transformer 中，任意两个位置之间的操作步数是常数，缩短了路径长度。
  
  关键技术组件：
  self-attention：模型使用自注意力机制来计算序列的内部表示，能够关联单个序列中不同位置的信息。
  
  Multi-Head Attention：允许模型同时关注来自不同表示子空间的信息。
  
  Scaled Dot-Product Attention：通过缩放因子防止梯度消失
  
  Positional Encoding：由于没有循环结构，模型通过向输入嵌入添加正弦和余弦函数来注入序列的位置信息
  

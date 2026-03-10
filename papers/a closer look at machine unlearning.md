# 大型语言模型的非目标性遗忘和目标性遗忘说明
## untargeted unlearning
 模型别再正确输出 forget set 的内容
 - ME 最大化预测熵
 - forget set 遗忘数集
 - 对 untargeted unlearning，不用“逼近某个不确定目标”，而改用 最大化每个token的预测熵，让模型在 forget set 上保持高不确定性，从而减少幻觉风险。
## targeted unlearning
可能忘得太狠，把正常知识也一起抹掉
- AP loss 通过保留 retain set 上原答案的概率，避免模型把“拒答模板”泛化到正常问题上
- retain set 需要被模型保留的训练数据子集，是forget set的补集


设模型参数是 θ，给定输入字符串 s，模型输出下一个 token 的概率分布记为 p(·|s; θ) 训练数据写成 D = {(x, y)} 其中：

x：问题 / 输入

y：答案 / 输出

普通微调的目标就是，最小化正确答案的预测损失：
    - ℓ(y∣x;θ)=−logp(y∣x;θ)
    
    1) ℓ读作 loss（损失函数）
    2) 𝑦 表示 目标输出 / 正确答案 / 标签序列
    3) x表示 输入
 而整个答案序列的概率，是把每个 token 的条件概率连乘起来：

 - <img width="478" height="110" alt="image" src="https://github.com/user-attachments/assets/7426a408-8df7-4c4d-85d0-21e6e7080a11" />


   T 是答案序列长度

   y_t 是第 t 个 token

   y<t 是第 t 个 token 之前的前缀

   ◦ 表示字符串拼接

   g(s; θ) 表示模型最终生成出来的文本
   

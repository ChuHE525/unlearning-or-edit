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

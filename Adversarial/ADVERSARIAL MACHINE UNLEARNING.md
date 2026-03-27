# ADVERSARIAL MACHINE UNLEARNING

Unlearner 先改模型，Auditor 再用最强的 membership inference 去查；unlearner 的目标就是既保住 retain/test 性能，又让 auditor 查不出来。
 
“对抗式机器遗忘”框架：把遗忘者和审计者建成一个双层优化问题，遗忘者既要保留模型效用，又要让成员推理攻击无法区分 forget set 与普通测试样本；为此，
 直接从原始模型 $\theta_o$ 出发，得到一个“被遗忘后的模型” $\theta_u$。

如何判断 $\theta_u$ 是否真的忘掉了 $D_f$？

答：让一个审计者用成员推理攻击来区分遗忘集 $D_f$ 和测试集。如果区分不出来，说明遗忘集在模型中的痕迹已经很弱，遗忘更可信。

{``遗忘成功'' = ``攻击者无法从模型输出中看出这些样本曾被训练过''}

**MIA** 

找forget set 的痕迹还在不在模型里
如果 forget 样本和 test 样本的输出分布还明显不同，攻击者就能区分；如果区分不了，就说明 forget 的痕迹被抹平了。

## Auditing Set 的定义

对于 unlearned model $\theta_u$，定义其 auditing set 为

<img width="534" height="111" alt="image" src="https://github.com/user-attachments/assets/df391ce5-9d56-465b-a80f-6a0d9d954d39" />

其中

$$
s_j^f = S_{\theta_u}(x_j^f), 
\qquad
s_j^{te} = S_{\theta_u}(x_j^{te}).
$$

这里，$\tilde{D}_{\theta_u}$ 表示针对模型 $\theta_u$ 构造的审计数据集，$S_{\theta_u}(x)$ 表示模型 $\theta_u$ 对输入 $x$ 的输出，$s_j^f$ 表示 forget 样本的输出，$s_j^{te}$ 表示 test 样本的输出。标签 $1$ 表示 member 类（对应 forget 样本），标签 $0$ 表示 non-member 类（对应 test 样本）。其中，模型输出 $s$ 可以是单个标量（如 cross-entropy loss），也可以是整条类别概率向量。

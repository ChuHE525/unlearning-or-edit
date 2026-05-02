---
title: ' Identifying Unlearned Data in LLMs via Membership Inference Attacks.md'

---



# Identifying Unlearned Data in LLMs via Membership Inference Attacks

## 任务

给定一个已经做过 unlearning 的模型 $M_u$ 和一组候选问题 $Q$，判断其中哪一个问题 $q_u$ 曾经被模型刻意 unlearn 过。攻击者能不能从模型留下的 **loss / gradient / LoRA 参数痕迹** 中识别出被忘掉的问题。

## 方法

攻击者拿到 unlearned model $M_u$ 和候选问题集合 $Q$，然后对每个候选问题 $q$ 计算一个分数 $\text{Score}_{M_u}(q)$。分数越高，表示该问题越可能是被 unlearn 过的目标问题。最后，攻击者选择分数最高的问题作为预测结果。

## 公式

### 1. Unlearning 目标

$$
M(q_i) \neq M_u(q_i)
$$

$$
M(q_j) = M_u(q_j),
\quad
\forall (q_j, a_j) \in F_t \setminus \{(q_i, a_i)\}
$$

### 符号解释

- $M$：原始模型
- $M_u$：unlearned model
- $q_i$：被要求忘掉的目标问题
- $a_i$：目标问题对应的答案
- $q_j$：同一个 topic 下其他没有被忘掉的问题
- $F_t$：topic $t$ 下的所有 question-answer pairs

这个公式表示：目标问题 $q_i$ 要被忘掉，所以 $M(q_i)$ 和 $M_u(q_i)$ 应该不同；但同一个 topic 下其他问题 $q_j$ 要被保留，所以 $M(q_j)$ 和 $M_u(q_j)$ 应该一致。

---

### 2. FUMA 攻击打分

$$q_u \in Q
$$

$$q^*=
\arg\max_{q \in Q}
\text{Score}_{M_u}(q)
$$

### 符号解释

- $Q$：候选问题集合
- $q_u$：真正被 unlearn 的问题
- $q^*$：攻击者预测的被 unlearn 问题
- $\text{Score}_{M_u}(q)$：攻击方法给候选问题 $q$ 计算的分数
- $\arg\max$：选择分数最高的那个问题

这个公式表示：候选集合 $Q$ 里有一个真正被忘掉的问题 $q_u$。攻击者对每个候选问题计算分数，并把分数最高的问题 $q^*$ 当作预测结果。如果 $q^* = q_u$，说明攻击成功。

---

### 3. Recall@k

$$
\text{Recall@k}=
\begin{cases}
1, & q_u \in \text{Top-k ranked candidates} \\
0, & q_u \notin \text{Top-k ranked candidates}
\end{cases}
$$

### 符号解释

- $\text{Recall@k}$：真正被忘掉的问题是否出现在前 $k$ 个预测结果中
- $q_u$：真正被 unlearn 的问题
- $\text{Top-k ranked candidates}$：按攻击分数排序后的前 $k$ 个候选问题

这个公式表示：如果真正被忘掉的问题 $q_u$ 出现在攻击排序的前 $k$ 名里，Recall@k 就是 1；否则就是 0。

---

### 4. Margin

$$\text{margin}_{(M_u, q_u)}=\frac{\text{Score}_{M_u}(q_u)-
\max_{q_i \in Q, q_i \neq q_u}
\text{Score}_{M_u}(q_i)
}{
\max_{q_i \in Q, q_i \neq q_u}
\text{Score}_{M_u}(q_i)
}
$$

### 符号解释

- $\text{Score}_{M_u}(q_u)$：真正被忘掉问题的分数
- $\max_{q_i \in Q, q_i \neq q_u}\text{Score}_{M_u}(q_i)$：除真实目标外，其他候选问题中的最高分
- $\text{margin}_{(M_u,q_u)}$：真实目标和最强干扰项之间的分数差距

这个公式表示：真实被忘掉的问题比分数第二高的候选问题高多少。margin 越大，说明攻击越有信心；如果 margin 小于 0，说明真实目标没有排第一，攻击失败。

## 总结

FUMA 的核心是：把模型是否留下遗忘痕迹转化成一个 membership inference attack。攻击者通过给候选问题打分，判断哪个问题最像被 unlearn 过。如果真正的 forget question 被排在前面，就说明这个 unlearning 方法仍然留下了可被识别的痕迹。
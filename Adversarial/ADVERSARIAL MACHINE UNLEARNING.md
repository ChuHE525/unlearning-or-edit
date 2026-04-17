# Adversarial Machine Unlearning 

## 方法 
- Unlearner 先修改模型，Auditor 再用最强的 membership inference attack 检查 forget set 是否还留下痕迹。Unlearner 的目标是：既保住 retain/test 性能，又让 Auditor 区分不出 forget 样本和 test 样本。



## 公式
### Auditing Set：
- 把 forget 样本和 test 样本经过 unlearned model 后的输出拿出来，分别标成 1 和 0，训练 auditor 判断这些输出来自 forget 还是 test。
对于 unlearned model $\theta_u$，构造审计数据集: $\tilde{D}_{\theta_u} = \{(s_j^{f}, 1), (s_j^{te}, 0)\}_{j=1}^{q}$





$$
s_j^f = S_{\theta_u}(x_j^f), 
\qquad
s_j^{te} = S_{\theta_u}(x_j^{te})
$$

### 符号说明

- $\widetilde{D}_{\theta_u}$：针对 unlearned model 构造的审计数据集。
- $S_{\theta_u}(x)$：模型 $\theta_u$ 对输入 $x$ 的输出。
- $s_j^f$：forget 样本经过模型后的输出。
- $s_j^{te}$：test 样本经过模型后的输出。
- 标签 $1$：forget 样本，代表 member-like。
- 标签 $0$：test 样本，代表 non-member-like。
- 标 1 和 0 是为了训练 auditor 做二分类：1 表示 forget 样本输出，0 表示 test 样本输出；如果 auditor 能分出来，说明模型还留下 forget 痕迹，如果分不出来，说明遗忘更干净。



###  Auditor 的最优攻击：
- Auditor 会从攻击器空间 Ha中选择在审计训练集上表现最好的攻击器 θa，然后用它在审计验证集上计算攻击效用 Ua

Auditor 的效用函数为：

$$
U_a(\theta_a,\theta_u)=
M(\widetilde{D}_{val}^{\theta_u};\theta_a)
$$

其中，Auditor 不是随便选攻击器，而是选择训练集上表现最好的攻击器：

$$
\theta_a \in \mathcal{B}_{\theta_u}=
\mathop{\arg\max}_{\theta_a' \in \mathcal{H}_a}
M(\widetilde{D}_{tr}^{\theta_u};\theta_a')
$$

### 符号说明

- $\theta_a$：Auditor 的攻击器参数。
- $\theta_u$：Unlearner 得到的模型参数。
- $M(\cdot;\theta_a)$：攻击器表现指标。
- $\widetilde{D}_{tr}^{\theta_u}$：审计训练集。
- $\widetilde{D}_{val}^{\theta_u}$：审计验证集。
- $\mathcal{H}_a$：Auditor 可选攻击器空间。
- $\mathcal{B}_{\theta_u}$：给定 $\theta_u$ 后 Auditor 的最佳响应集合。



### Unlearner 的目标函数
- 第一项管模型性能，第二项管遗忘是否干净。

$$
C_u(\theta_u,\theta_a)=
L(D_r;\theta_u)
+
\alpha M(\widetilde{D}_{val}^{\theta_u};\theta_a)
$$

### 两项含义

第一项：

$$
L(D_r;\theta_u)
$$

表示 retain set 上的损失，用来保证模型不要把该保留的知识也忘掉。

第二项：

$$
\alpha M(\widetilde{D}_{val}^{\theta_u};\theta_a)
$$

表示 Auditor 的攻击效果。Unlearner 希望它越小越好，也就是让 Auditor 查不出 forget 痕迹。

### 符号说明

- $C_u$：Unlearner 的总成本函数。
- $D_r$：retain set。
- $\alpha$：权衡系数。
- $\alpha$ 越大，越重视让 Auditor 失败。
- $\alpha$ 越小，越重视保留模型性能。

 



### 双层优化 / Stackelberg Game

整体目标写成：

$$
\mathop{\arg\min}_{\theta_u \in \mathcal{H}_u}
L(D_r;\theta_u)
+
\alpha M(\widetilde{D}_{val}^{\theta_u};\theta_a)
\quad
\text{s.t.}
\quad
\theta_a \in \mathcal{B}_{\theta_u}
$$

### 含义

这是一个先手—后手博弈：

- **Unlearner 先手**：选择如何修改模型 $\theta_u$。
- **Auditor 后手**：看到 $\theta_u$ 后，选择最优攻击器 $\theta_a$。



### 对 Unlearner 目标求梯度

$$
\frac{\partial C_u}{\partial \theta_u}=
\frac{\partial L(D_r;\theta_u)}{\partial \theta_u}
+
\frac{\partial M(\widetilde{D}_{val}^{\theta_u};\theta_a)}{\partial \theta_a}
\cdot
\frac{\partial \theta_a}{\partial \widetilde{D}_{tr}^{\theta_u}}
\cdot
\frac{\partial \widetilde{D}_{tr}^{\theta_u}}{\partial \theta_u}
$$

### 含义

第一项：

$$
\frac{\partial L(D_r;\theta_u)}{\partial \theta_u}
$$

是普通 retain loss 的梯度，用来保持模型性能。

第二项表示一条链式影响：

$$
\theta_u
\longrightarrow
\widetilde{D}_{tr}^{\theta_u}
\longrightarrow
\theta_a
\longrightarrow
M
$$

---

### 用 KKT 条件表示 Auditor 最优解

Auditor 的最优攻击器通常没有显式解，不能直接写成：

$$
\theta_a = g(\widetilde{D}_{tr}^{\theta_u})
$$

但如果下层攻击器优化是凸问题，比如线性 SVM，那么最优解满足 KKT 条件，可以写成隐式方程：

$$
f(\widetilde{D}_{tr}^{\theta_u},\theta_a)=0
$$

### 符号说明

- $f$：由 KKT 最优性条件组成的方程组。
- $\widetilde{D}_{tr}^{\theta_u}$：审计训练集。
- $\theta_a$：Auditor 的最优攻击器参数。




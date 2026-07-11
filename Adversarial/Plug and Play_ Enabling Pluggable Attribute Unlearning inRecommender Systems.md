# Plug and Play: Enabling Pluggable Attribute Unlearning in Recommender Systems

## 方法

- 通过最大化类别编码率 $R^c$ 来破坏用户嵌入中与性别、年龄等属性相关的分布模式，同时使总体编码率 $R$ 维持在合理值 $b$ 附近，从而有效删除敏感属性信息，并保留推荐所需的用户兴趣信息。

## 公式

### （1）总体编码长度

计算全部嵌入向量包含的信息量，用编码长度衡量嵌入分布是集中还是分散。

$$L(\mathbf{Z}, \epsilon)=
\frac{m+d}{2}
\log \det
\left(
\mathbf{I}+
\frac{d}{m\epsilon^2}
\mathbf{Z}\mathbf{Z}^{\top}
\right)
$$

### （2）总体编码率

计算每个样本平均需要的编码量。

$$
R(\mathbf{Z},\epsilon)=
\frac{1}{2}
\log\det
\left(
\mathbf{I}+
\frac{d}{m\epsilon^2}
\mathbf{Z}\mathbf{Z}^{\top}
\right)
$$

### （3）类别编码率

按照隐私属性划分类别后计算类别内部编码率。

$$
R^{c}(\mathbf{Z}, \epsilon \mid \Pi)=
\sum_{j=1}^{k}
\frac{\operatorname{tr}(\Pi_j)}{2m}
\log\det
\left(
\mathbf{I}+
\frac{d}
{\operatorname{tr}(\Pi_j)\epsilon^2}
\mathbf{Z}\Pi_j\mathbf{Z}^{\top}
\right)
$$

### （4）最大化类别编码率

破坏同一属性类别的共同规律，同时让不同类别发生重叠。

$$
J(\hat{\mathbf{U}}, \Pi)=
R^c(\hat{\mathbf{U}}, \epsilon \mid \Pi)-
R(\hat{\mathbf{U}}, \epsilon)
$$

### （5）带约束优化目标

限制嵌入大小并保证类别分配有效。

$$
\begin{aligned}
\max_{\hat{\mathbf{U}}, \Pi}
\quad &
J(\hat{\mathbf{U}}, \Pi)
\\
\text{s.t.}\quad &
\|\hat{\mathbf{U}}_j\|_F^2 = m_j
\\
&
\Pi \in \Omega
\end{aligned}
$$

### （6）属性遗忘目标

删除敏感属性，同时避免推荐信息被过度删除。

$$
\begin{aligned}
\max_{\hat{\mathbf{U}}, \Pi}
\quad &
R^c(\hat{\mathbf{U}}, \epsilon \mid \Pi)-
\lambda
\left|
R(\hat{\mathbf{U}}, \epsilon)-b
\right|
\\
\text{s.t.}\quad &
\|\hat{\mathbf{U}}_j\|_F^2 = m_j
\\
&
\Pi \in \Omega
\end{aligned}
$$

### （7）类别分配矩阵梯度

计算类别分配矩阵的更新方向。

$$
\nabla_{\Pi_j}
R^c(\hat{\mathbf{U}},\epsilon \mid \Pi)=
\frac{\operatorname{tr}(\Pi_j)}{2m}
\hat{\mathbf{U}}^{\top}
\left(
\mathbf{I}
+
\frac{d}
{\operatorname{tr}(\Pi_j)\epsilon^2}
\hat{\mathbf{U}}\Pi_j\hat{\mathbf{U}}^{\top}
\right)^{-1}
\hat{\mathbf{U}}
$$

### （8）总体编码率梯度

计算总体编码率对用户嵌入的梯度。

$$
\nabla_{\hat{\mathbf{U}}}
R(\hat{\mathbf{U}},\epsilon)=
A\hat{\mathbf{U}}
\left(
\mathbf{I}+
\hat{\mathbf{U}}^{\top}
A\hat{\mathbf{U}}
\right)^{-1}
$$

### （9）类别编码率梯度

计算类别编码率对用户嵌入的梯度。

$$
\nabla_{\hat{\mathbf{U}}}
R^c(\hat{\mathbf{U}},\epsilon \mid \Pi)=
\sum_{j=1}^{k}
A\Pi_j\hat{\mathbf{U}}
\operatorname{tr}(\Pi_j)
\left(
\mathbf{I}+
\frac{d}
{\operatorname{tr}(\Pi_j)\epsilon^2}
\hat{\mathbf{U}}
\Pi_j
\hat{\mathbf{U}}^{\top}
\right)^{-1}
$$

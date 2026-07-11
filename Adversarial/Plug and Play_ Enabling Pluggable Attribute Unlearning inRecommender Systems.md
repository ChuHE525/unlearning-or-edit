---
title: 'Plug and Play: Enabling Pluggable Attribute Unlearning in Recommender Systems'
---

# Plug and Play: Enabling Pluggable Attribute Unlearning in Recommender Systems

## 方法

- 通过最大化类别编码率 $R^c$ 来破坏用户嵌入中与性别、年龄等属性相关的分布模式，同时使总体编码率 $R$ 维持在合理值 $b$ 附近，从而有效删除敏感属性信息，并保留推荐所需的用户兴趣信息。

## 公式

- 计算全部嵌入向量要有多少信息，用编码长度来衡量一组嵌入是集中还是分散

$$
L(\mathbf{Z}, \epsilon)=
\frac{m+d}{2}
\log \det
\left(
\mathbf{I}+
\frac{d}{m\epsilon^2}
\mathbf{Z}\mathbf{Z}^{\top}
\right)
\tag{1}
$$

1. Z 是指

- 计算每个样本平均需要的编码量，表示每个用户嵌入是集中还是分散

$$
R(\mathbf{Z},\epsilon)
=
\frac{1}{2}
\log\det
\left(
\mathbf{I}
+
\frac{d}{m\epsilon^2}
\mathbf{Z}\mathbf{Z}^{\top}
\right)
\tag{2}
$$

- 把用户按照隐私属性分类，再分别计算每个类别内部的编码率

$$
R^{c}(\mathbf{Z}, \epsilon \mid \Pi)
=
\sum_{j=1}^{k}
\frac{\operatorname{tr}(\Pi_j)}{2m}
\log \det
\left(
\mathbf{I}
+
\frac{d}
{\operatorname{tr}(\Pi_j)\epsilon^2}
\mathbf{Z}\Pi_j\mathbf{Z}^{\top}
\right)
\tag{3}
$$

- 破坏同一属性类别的共同规律，同时让不同的属性类别相互堆叠

$$
\max_{\hat{\mathbf{U}}, \Pi}
J(\hat{\mathbf{U}}, \Pi)
=
R^c(\hat{\mathbf{U}}, \epsilon \mid \Pi)
-
R(\hat{\mathbf{U}}, \epsilon)
\tag{4}
$$

- 限制向量大小，并保证类别分配有效

$$
\begin{aligned}
\max_{\hat{\mathbf{U}}, \Pi}
\quad &
J(\hat{\mathbf{U}}, \Pi)
=
R^c(\hat{\mathbf{U}}, \epsilon \mid \Pi)
-
R(\hat{\mathbf{U}}, \epsilon)
\\
\text{s.t.}\quad &
\|\hat{\mathbf{U}}_j\|_F^2 = m_j,
\\
&
\Pi \in \Omega.
\end{aligned}
\tag{5}
$$

- 删除敏感属性，同时避免推荐信息过度删除

$$
\begin{aligned}
\max_{\hat{\mathbf{U}}, \Pi}
\quad &
\hat{J}(\hat{\mathbf{U}}, \Pi)
=
R^c(\hat{\mathbf{U}}, \epsilon \mid \Pi)
-
\lambda
\left|
R(\hat{\mathbf{U}}, \epsilon)-b
\right|
\\
\text{s.t.}\quad &
\|\hat{\mathbf{U}}_j\|_F^2 = m_j,
\\
&
\Pi \in \Omega.
\end{aligned}
\tag{6}
$$

- 计算类别分配矩阵该往哪个方向更新

$$
\nabla_{\Pi_j}
R^c(\hat{\mathbf{U}},\epsilon \mid \Pi)
=
\frac{\operatorname{tr}(\Pi_j)}{2m}
\hat{\mathbf{U}}^{\top}
\left(
\mathbf{I}
+
\frac{d}
{\operatorname{tr}(\Pi_j)\epsilon^2}
\hat{\mathbf{U}}
\Pi_j
\hat{\mathbf{U}}^{\top}
\right)^{-1}
\hat{\mathbf{U}}
\tag{7}
$$

- 计算总体编码率 $R$ 对用户嵌入的梯度

$$
\nabla_{\hat{\mathbf{U}}}
R(\hat{\mathbf{U}},\epsilon)
=
A\hat{\mathbf{U}}
\left(
\mathbf{I}
+
\hat{\mathbf{U}}^{\top}
A\hat{\mathbf{U}}
\right)^{-1}
\tag{8}
$$

- 计算类别编码率 $R^c$ 对用户嵌入的梯度

$$
\nabla_{\hat{\mathbf{U}}}
R^c(\hat{\mathbf{U}},\epsilon \mid \Pi)
=
\sum_{j=1}^{k}
A\Pi_j\hat{\mathbf{U}}
\operatorname{tr}(\Pi_j)
\left(
\mathbf{I}
+
\frac{d}
{\operatorname{tr}(\Pi_j)\epsilon^2}
\hat{\mathbf{U}}
\Pi_j
\hat{\mathbf{U}}^{\top}
\right)^{-1}
$$

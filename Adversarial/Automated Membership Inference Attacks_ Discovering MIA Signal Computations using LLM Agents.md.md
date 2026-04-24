---
title: 'Automated Membership Inference Attacks: Discovering MIA Signal Computations using LLM Agents.md'

---

# Automated Membership Inference Attacks: Discovering MIA Signal Computations using LLM Agents


## 任务

**自动设计一个更强的 membership inference attack (MIA) 信号函数**：给定样本和模型输出，输出一个 membership score，用来判断该样本是否出现在训练集中。

## 方法

让不同 LLM agents 分工合作：**Explorer** 负责想新攻击思路，**Exploiter** 负责在已有强方法上继续优化，**Programmer** 把设计写成代码，**Executor** 跑实验，**Analyzer** 总结结果并把经验存回数据库，形成“探索—实现—评估—反馈”的闭环。



## 公式

## 1. MIA 信号函数定义

$$
f : O \times X \to \mathbb{R}
$$





### 符号解释

* $f$：membership inference signal function，即攻击打分函数
* $O$：模型输出空间，例如 predicted class、confidence score、logits、生成文本等
* $X$：输入样本空间
* $\mathbb{R}$：实数空间，表示输出是一个连续分数
* $f(o, x)$：给定模型输出 $o$ 和样本 $x$ 后计算出的 membership score
- AutoMIA 要找的对象是一个**打分规则**。它读入模型输出和输入样本，输出一个实数分数；分数越高，样本越像训练集成员。

## 2. 最优攻击函数搜索目标

$$
f^* = \arg\max_{f \in F} J(f, D^{train}_{MIA})
$$


### 符号解释

* $f^*$：最优的 MIA 信号函数
* $\arg\max$：取使目标最大化的那个解
* $f$：任意一个候选信号函数
* $F$：候选 MIA 信号函数构成的设计空间
* $J(\cdot)$：评价函数，如 AUC、低 FPR 下的 TPR
* $D^{train}_{MIA}$：用于设计攻击的训练集，包含 member 和 non-member 样本
- 把设计 MIA写成一个**搜索最优信号函数**的问题：在候选函数集合 $F$ 里，找到在 MIA 训练集上表现最好的那个函数 $f^*$。也就是说，AutoMIA 优化的不是目标模型参数，而是**攻击规则本身**。 
---

## 3. 单次实验记录的数据库表示

$$
s = (id, d, c, r) \in DB
$$



  

### 符号解释

* $s$：一次实验尝试
* $id$：实验编号
* $d$：设计信息
* $c$：代码实现
* $r$：运行结果
* $DB$：实验数据库
* $\in$：属于
- 这个式子是AutoMIA 的记忆单元。系统不是一次次盲试，而是把每次实验都存进数据库，供后续 agent 检索、比较和学习。


## 4. 设计信息的结构

$$
d = (idea, design, parent_id)
$$

 

### 符号解释

* $d$：一次设计描述
* $idea$：高层想法，核心思路
* $design$：更具体的设计说明
* $parent_id$：父设计编号，表示该设计是从哪个旧设计演化来的

- 数据库里存的不只是代码，还要存高层想法和设计说明。这说明 AutoMIA 不是直接在代码层面盲目进化，而是先在**自然语言设计层**上进化攻击思路，再把它实现成代码。

## 5. 结果信息的结构

$$
r = (status, metrics, analysis)
$$



### 符号解释

* $r$：实验结果
* $status$：运行状态，例如成功、失败、超时
* $metrics$：实验指标，如 AUC
* $analysis$：对本次结果的分析总结

- 说明一次尝试的输出不只是“分数高低”，还包括运行状态和分析总结。也就是说，系统不仅会跑实验，还会把实验结果整理成后续可利用的经验。 

## 6. 用户配置

$$
C = (codebase, spec, params)
$$



### 符号解释

* $C$：用户给定的配置
* $codebase$：实验底座，负责模型加载、数据加载、推理和评估
* $spec$：函数 $f$ 的规格说明，规定输入输出结构和上下文
* $params$：系统参数，如尝试次数、超时时间、探索/利用安排

- AutoMIA 并不是固定针对一种场景，而是由用户先定义任务配置，再在这个配置下自动搜索攻击。也就是说，不同的 threat model、模型访问权限和数据模态，都能通过配置 $C$ 注入系统。 

## 7. Explorer 生成初始设计

$$
d^{(0)} = LLM_{gen}(R, spec)
$$



### 符号解释

* $d^{(0)}$：第 0 轮的初始设计
* $LLM_{gen}$：负责生成新设计的 LLM 代理 
* $R$：从数据库里取出来给它参考的一些已有实验
* $spec$：任务规格说明，该按什么要求设计方法
* 上标 $(0)$：第 0 次迭代，最开始版本

- 让生成代理参考数据库里的旧实验 R 和当前任务要求 spec，先提出一个新的攻击设计初稿

## 8. Explorer 的新颖性判断

$$
(action, suggestion) = LLM_{judge}(d^{(t)}, E)
$$





### 符号解释

* $action$：判断动作或结果，如通过/不通过
* $suggestion$：若不够新颖，给出的修改建议
* $LLM_{judge}$：负责判断新颖性的 LLM 子代理
* $d^{(t)}$：第 $t$ 轮当前设计
* $E$：与当前设计相关的已有设计集合
* $t$：当前迭代轮数

- Explorer 提出一个设计后，不会立刻采用，而是先检查它是否足够新。如果和已有设计太像，系统就不会停下，而是要求继续修改。这一步保证了搜索过程真的在**扩展设计空间**，而不是重复已有思路。 

## 9. Explorer 的设计精修

$$
d^{(t+1)} = LLM_{refine}(d^{(t)}, suggestion)
$$





### 符号解释

* $d^{(t+1)}$：下一轮 refined 后的新设计
* $LLM_{refine}$：负责精修设计的 LLM 子代理
* $d^{(t)}$：当前设计
* $suggestion$：新颖性判断阶段给出的建议
* $t+1$：下一轮迭代

- 如果当前设计不够新，系统不会直接放弃，而是根据上一轮的建议继续改。这样 Explorer 形成了一个生成—判断—精修的循环，使新设计既新颖又可行。 

## 10. Exploiter 的高分实验集合

$$
T = s_1, \ldots, s_K
$$





### 符号解释

* $T$：top-$K$ 高性能实验的集合
* $s_1, \ldots, s_K$：集合中的各个实验
* $K$：保留的高性能实验个数

- Exploiter 不负责从零发明，而是从当前数据库里最强的 top-$K$ 设计出发继续优化。这个集合 $T$ 就是 Expoliter 的候选父代池。 

## 11. Exploiter 选择父设计的概率

$$
P(d_{parent} = s_i) =
\frac{|AUC(s_i)-0.5|}
{\sum_{j=1}^{K}|AUC(s_j)-0.5|}
$$





### 符号解释

- $P(d_{parent}=s_i)$：方法 $s_i$ 被选中作为父设计的概率 
- $d_{parent}$：这次被拿来继续优化的旧设计 
- $s_i$：父代池里的第 $i$ 个候选实验 
- $AUC(s_i)$：这个实验的攻击效果分数 
- $0.5$：随机猜测的基线 AUC 
- $|AUC(s_i)-0.5|$：这个方法比随机猜强了多少 
- $\sum_{j=1}^{K}$：把 top-$K$ 候选方法的权重全部加起来 
- $K$：父代池里候选高分实验的数量 

- Exploiter 更偏向选择那些**明显优于随机猜**的方法来继续优化。因为 $AUC=0.5$ 代表接近随机，所以离 0.5 越远，说明这个设计越有攻击信息量，越值得继续挖。 

## 12. Exploiter 生成子设计

$$
d_{child} = LLM_{exploiter}(d_{parent}, S_{anc}, S_{sib}, S_{rel}, spec)
$$





### 符号解释

* $d_{child}$：新生成的子设计
* $LLM_{exploiter}$：负责局部优化的 LLM 代理
* $d_{parent}$：父设计
* $S_{anc}$：ancestor chain，祖先链
* $S_{sib}$：sibling set，兄弟设计集合
* $S_{rel}$：语义相关设计集合
* $spec$：任务规格说明

- 体现了 Exploiter 的局部精修思想：它不是盲目小改，而是综合参考父设计、祖先设计、兄弟设计和语义相近设计，再产生新的子设计。因此，它的优化不是随机扰动，而是**带历史上下文的定向改进**。 

## 13. Programmer 生成代码

$$
c.program = A_{programmer}(d, spec)
$$



 

### 符号解释

* $c.program$：生成出的程序代码
* $A_{programmer}$：Programmer agent
* $d$：设计说明
* $spec$：函数规格说明

- 前面所有设计都是自然语言层面的思路，这一步才真正把它落到实现层。Programmer 负责把设计翻译成可执行代码，让攻击思路能被实验验证。

## 14. Executor 执行实验

$$
r = A_{executor}(c.program, C.codebase, T_{max})
$$





### 符号解释

* $r$：运行结果
* $A_{executor}$：Executor agent
* $c.program$：上一阶段生成的代码
* $C.codebase$：用户给定的实验底座
* $T_{max}$：单次尝试的最大运行时间

- 真正把代码跑起来并拿到实验结果。如果代码报错，系统会回传错误继续修复；如果运行成功，就得到本次攻击设计的性能指标。
## 15. Analyzer 结果分析

$$
r.analysis = A_{analyzer}(r.metrics, d)
$$





### 符号解释

* $r.analysis$：结果分析文本
* $A_{analyzer}$：Analyzer agent
* $r.metrics$：本次实验得到的指标
* $d$：当前实验对应的设计
- Analyzer 的作用是把实验指标转成可复用经验。它根据本次设计和指标总结原因，再把分析和结果一起存回数据库，帮助后续设计更快收敛。 

# 总结

重点不是某一个具体的 MIA signal 细节，而是 **AutoMIA 这套自动搜索框架的建模**：先把 MIA 写成寻找最优打分函数的问题，再通过 Explorer、Exploiter、Programmer、Executor、Analyzer 和数据库构成一个带记忆的闭环搜索系统，最终自动发现比人工 baseline 更强的攻击信号。 

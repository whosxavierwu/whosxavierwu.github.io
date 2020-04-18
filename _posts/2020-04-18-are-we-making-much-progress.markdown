---
layout: post
title:  "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches"
date:   2020-04-18 00:00:00 +0800
categories: recommender
---

# 1、概述

这篇 RecSys 2019 的 Best Paper，从标题看来就很强，有种“在座各位都是……”的感觉。这两天通读下来，个人认为论文的贡献主要还是在于对学界敲响警钟，而不在于 new idea。

总的来说，作者从2018年的顶会里挑选了18篇（RecSys:7, KDD:4, WWW:4, SIGIR:3）深度学习Top-N推荐模型的文章，发现其中只有7篇（RecSys:1, KDD:3, WWW:2, SIGIR:1）的结果能被不那么难的进行复现，而这7篇之中有6篇是往往能被相对简单的算法超越。剩下的一篇确实能显著的超越baseline，但并不总能超越非神经网络的线性排序算法。

毕竟通篇强调 Reproducibility，作者当然有将实验代码公开到 GitHub 上： [https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation) ，有兴趣的同学可以进一步研究。

# 2、Baseline

作为参考，论文选择了以下7个算法作为 Baseline 算法。

## 2.1 TopPopular

顾名思义，直接取热门商品种的Top-N。其中热门程度通过显性或隐性的打分数量。

## 2.2 ItemKNN

传统的基于kNN与物品间相似度的协同过滤算法。

用$\vec{r_i}, \vec{r_j} \in R^{\|U\|}$分别表示物品$i, j$的打分向量，向量维度$\|U\|$表示用户总数；则物品间相似度可通过余弦相似度进行计算：

$$
s_{ij} = \frac{\vec{r_i}\vec{r_j}}{||\vec{r_i}||*||\vec{r_j}|| + h}
$$

其中，打分向量可选用 TF-IDF 或者 BM25 进行加权；相似度也可不用余弦相似度，而直接用向量内积。

得到物品间相似度以后，根据用户所浏览过的物品找到相似的物品即可。

## 2.3 UserKNN

传统的基于kNN与用户间相似度的协同过滤算法。

整体流程与 ItemKNN 相似。用$\vec{r_i}, \vec{r_j} \in R^{\|I\|}$分别表示用户$i, j$的打分向量，向量维度$\|I\|$表示物品总数；则用户间相似度可通过余弦相似度进行计算：

$$
s_{ij} = \frac{\vec{r_i}\vec{r_j}}{||\vec{r_i}||*||\vec{r_j}|| + h}
$$

同样的，打分向量可选用 TF-IDF 或者 BM25 进行加权；相似度也可不用余弦相似度，而直接用向量内积。

得到用户间相似度以后，根据用户找到与其相似的用户，推送相似用户所浏览过的物品即可。

## 2.4 ItemKNN-CBF

与 ItemKNN 基本一致，只是不是简单的使用的物品的打分向量，而是采用基于物品content计算物品的特征向量。

用$\vec{f_i}, \vec{f_j} \in R^{\|F\|}$分别表示物品$i, j$的特征向量，向量维度$\|F\|$表示特征总数；则物品间相似度可通过余弦相似度进行计算：

$$
s_{ij} = \frac{\vec{f_i}\vec{f_j}}{||\vec{f_i}||*||\vec{f_j}|| + h}
$$

## 2.5 ItemKNN-CFCBF

结合 ItemKNN 与 ItemKN-CBF ，将$\vec{r_i}$与$\vec{f_i}$简单拼接成$\vec{v_i}=[\vec{r_i}, w\vec{f_i}]$。后面的步骤一样，不再赘述。

## 2.6 $P^3\alpha$

基于图上random-walk的思想。

用$r_{ui}$表示用户$u$对物品$i$的打分，$N_u$表示用户$u$打分的总数，$N_i$表示物品$i$打分的总数。物品相似度计算如下：

$$
\begin{align}
s_{ij} &= \sum_{u}p_{ju}*p_{ui} \\
&= \sum_{u}(\frac{r_{uj}}{N_j})^\alpha*(\frac{r_{ui}}{N_u})^\alpha
\end{align}
$$

后面的步骤则与 ItemKNN 的一样。

## 2.7 $RP^3\beta$

在 $P^3\alpha$ 的基础上进行改进。假设物品的热门度为$h_{i}$。（存疑？）则物品相似度改为：

$$
\begin{align}
s_{ij} = (\sum_{u}(\frac{r_{uj}}{N_j})^\alpha*(\frac{r_{ui}}{N_u})^\alpha)/(q_i^\beta*q_j^\beta)
\end{align}
$$

# 3、实验

对于所有的Baseline算法，统一利用 Scikit-Optimize 通过 Bayesian Search 自动找到最优参数。$k$取5~800，$h$取0~1000，$\alpha, \beta$取0~2。

## 3.1 Collaborative Memory Networks (CMN)

> Travis Ebesu, Bin Shen, and Yi Fang. 2018. Collaborative Memory Network for Recommendation Systems. In Proceedings SIGIR ’18. 515–524.

![CMN vs baseline]({{ site.url }}/assets/cmn_vs_baseline.png)

## 3.2 Metapath based Context for RECommendation (MCRec)

> Binbin Hu, Chuan Shi,Wayne Xin Zhao, and Philip S Yu. 2018. Leveraging metapath based context for top-n recommendation with a neural co-attention model. In Proceedings KDD ’18. 1531–1540.

![MCRec vs baseline]({{ site.url }}/assets/mcrec_vs_baseline.png)

## 3.3 Collaborative Variational Autoencoder (CVAE)

> Xiaopeng Li and James She. 2017. Collaborative variational autoencoder for recommender systems. In Proceedings KDD ’17. 305–314.

![CVAE vs baseline]({{ site.url }}/assets/cvae_vs_baseline.png)

## 3.4 Collaborative Deep Learning (CDL)

> HaoWang, NaiyanWang, and Dit-Yan Yeung. 2015. Collaborative deep learning for recommender systems. In Proceedings KDD ’15. 1235–1244.

![CDL vs baseline]({{ site.url }}/assets/cdl_vs_baseline.png)

## 3.5 Neural Collaborative Filtering (NCF)

> Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In Proceedings WWW ’17. 173–182.

![NCF vs baseline]({{ site.url }}/assets/ncf_vs_baseline.png)

## 3.6 Spectral Collaborative Filtering (SpectralCF)

> Lei Zheng, Chun-Ta Lu, Fei Jiang, Jiawei Zhang, and Philip S. Yu. 2018. Spectral Collaborative Filtering. In Proceedings RecSys ’18. 311–319.

![SpectralCF vs baseline]({{ site.url }}/assets/SpectralCF_vs_baseline.png)

## 3.7 Variational Autoencoders for Collaborative Filtering (Mult-VAE)

> Dawen Liang, Rahul G Krishnan, Matthew D Hoffman, and Tony Jebara. 2018. Variational Autoencoders for Collaborative Filtering. In Proceedings WWW ’18. 689–698.

![Mult-VAE vs baseline]({{ site.url }}/assets/multvae_vs_baseline.png)

![Mult-VAE-add vs baseline]({{ site.url }}/assets/multvae_vs_baseline-1.png)

# 结语

这里先对论文进行简单的总结与摘要。文中提到的七篇论文的对比实验……光看数据也没有什么收获，得等后面具体的看七篇论文后再展开来说了。

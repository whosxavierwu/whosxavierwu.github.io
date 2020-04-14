---
layout: post
title:  "Deep Neural Network for YouTube Recommendation System"
date:   2020-04-13 00:00:00 +0800
categories: recommender
---

# 概述

这篇论文是对YouTube中基于DNN的推荐系统的整体描述。本文旨在对论文进行总结。

整个系统主要分 Candidate Generation 和 Ranking，也就是召回和排序两部分。召回模块从数百万的视频集合中挑选出数百个候选视频；排序模块则是从数百个中挑选出几十个视频，并排序后推送给用户。

下图是整体框架：

![整体框架]({{ site.url }}/assets/youtube-dnn-whole.jpg)

# 一、召回

![Candidate Generation]({{ site.url }}/assets/youtube-dnn-candidate-generate.jpg)

## 1. 问题定义

YouTube将召回问题转化为一个多分类问题去处理，建模以预测：在$t$时刻发生的某次视频观看事件$w_t$中，具体观看的是视频集合$V$中的哪个视频。

假设用$U$表示这次事件中的用户，用$C$表示上下文，用$u$表示对$U$、$C$一起进行embedding后的特征向量，用$v_i$表示对视频$i$进行embedding后的特征向量。则我们需要预测的分类到视频$i$的概率可以形式化如下：

$$
P(w_t=i|U,C)=\frac{e^{v_iu}}{\sum_{j\in V}e^{v_ju}}
$$

通过这样的问题转化以后，理想情况下，我们能预测到在当前用户、当前情景下，每个视频被观看的概率。取概率最高的前M个视频即可作为召回模块的输出。

## 2. 数据准备

在准备数据样本时，需要注意：

1. 数据源方面，采用所有YouTube视频的观看事件（如嵌在其他网站的视频等），而不仅仅是YouTube主站上的。
2. 对每个用户带来的训练样本数进行了限制，从而避免高活跃用户对模型的过度影响。
3. 注意避免样本数据中掺入未来信息，模型的输入应该始终只有打标签以前的数据。

![训练数据筛选]({{ site.url }}/assets/youtube-dnn-dataset.jpg)

## 3. 特征处理

特征方面，抽象来看，主要涉及用户属性、用户行为与事件时间特征三大块。作者在论文中给出了不同特征组合的效果对比：

![Features]({{ site.url }}/assets/youtube-dnn-feature-select.jpg)

### 3.1 用户属性特征

用户属性特征在论文中只是简单的一笔带过，包括：用户处理的地理位置、设备、性别、登录状态、年龄。直觉来看，这类特征应该对新用户的推荐效果有着重要影响。

尽管没有细讲，从最后对不同特征组合的实验来看，却似乎带有很大的提升："All Features"相对于"Watches, Searches & Example Age"有显著提升。

### 3.2 用户行为特征

对用户行为的特征挖掘，主要从用户的视频观看历史（"watch vector"）与搜索历史（"search vector"）着手。

#### "watch vector"

从用户的视频观看历史中挖掘特征主要分两步：
1. 通过单独的模型预训练好每个视频的embedding。
2. 取出用户历史（*All or Top-k?*）观看的视频的embedding取均值，作为 "watch vectors"。

具体而言，是如何做embedding的呢？文中只是简单的提了一下：

> Inspired by continuous bag of words language models, we learn high dimensional embeddings for each video in a fixed vocabulary

从我的理解来看，应该是将每个用户历史观看的视频ID序列，看作一个“句子”，所有用户的“句子”汇聚成一个语料集合；进而参考Word2Vec中基于CBOW的训练方式来做训练，从而获得每个视频的embedding。

作者还提到了一句：

> Importantly, the embeddings are learned jointly with all other model parameters through normal gradient descent backpropagation updates 

这个我不太理解。

#### "search vector"

从用户的搜索历史中挖掘特征的步骤，与前面相似：
1. 将每个query分词成unigrams跟bigrams，而token又是被embedding好的，
2. 汇总所有的这些embedding求均值，作为 "search vector"

> Search history is treated similarly to watch history - each query is tokenized into unigrams and bigrams and each token is embedded. Once averaged, the user’s tokenized, embedded queries represent a summarized dense search history

从作者的描述来看，应该就是基于用户的搜索预料来训练Word2Vec模型，从而得到embedding向量。

### 3.3 事件时间特征

"Example Age" 是个较为特殊的特征。引入这个特征，是因为作者观察到，用户更偏好新产的视频。

> we feed the age of the training example as a feature during training. At serving time, this feature is set to zero (or slightly negative) to reflect that the model is making predictions at the very end of the training window.

论文在一张插图的描述中提到：

> the example age is expressed as $t_{max} - t_N$ where $t_{max}$ is the maximum observed time in the training data.

$t_N$指的是样本打标签的时间，也就是当前的事件的时间戳，这个好理解。

虽然说得比较模糊，但结合前面的描述：在serving时，该特征被置为零。所以$t_{max}$应该是指全体训练样本中的最大观测时间。

至于具体是用秒？分钟？小时？还是天？则没有提及，考虑到不同量纲之间可以通过线性变换来相互切换，所以这个问题的影响不大。

作者通过统计分析表明，模型在加入了"Example Age"之后，能比较好的捕捉到视频上传时间的影响。

![Example Age]({{ site.url }}/assets/youtube-dnn-example-age.jpg)

那么问题来了，为什么不直接用"Days Since Upload"来做特征呢？

## 4. 模型训练与线上服务

### 4.1 训练技巧： Negative Sampling

一般情况下，基于 SoftMax 的 Cross-Entropy Loss 形式如下：

$$
logit(i)=\frac{exp(w_{i}x)}{\sum^{M}_{j}{exp({w_{j}x})}}
$$

$$
loss=-log(logit(i))=-(w_ix)+log(\sum^{M}_{j}{exp(w_jx)})
$$

可以看到，当类别数$M$多达数百万的时候，损失函数的后半部分$ log(\sum^{M}_{j}exp(w_jx)) $的计算量将会特别大。

而 Negative Sampling 的思路则是，通过采样指定$K$个类别，从而把计算量从$O(M) \to O(K)$控制了下来。作者在论文中指出，一般$K$取数千。

这里有几个细节：

1. $K$是否把类别$i$包含在内？
2. 具体如何进行随机采样？均匀采样？
3. 是每个训练样本都做一次采样？还是每个batch做一次采样？
4. 每次负采样、训练时，并不会更新$K$个被选中的类别以外的类别权重。那么如果存在某个类别的样本数量相对较大，会不会对模型效果有影响？

### 4.2 线上服务

![Candidate Generation Serving]({{ site.url }}/assets/youtube-dnn-recall-serving.jpg)

模型框架图中的这个细节，是我一开始没有留意到的。

当时只是想当然的认为，在做serving时，每次用户来到时，跑一遍模型预测，然后取出概率值Top N的视频来召回。而从YouTube的框架图来看，实际做serving时是以下步骤：

1. 从最后一层ReLU层获取用户向量$\vec{u}$（256维）；
2. 从SoftMax层获取视频向量$\vec{v_j}$；
3. 通过最近邻搜索来找到近似的Top N视频。

以上的简要描述可能仍然不好理解。

我们知道，在ReLU和SoftMax两层之间存在一个大小为$(256, V)$的权重矩阵$\vec{W}$，$V$表示视频总数；$\vec{W}$通过训练学习到。

来看常规的feedward流程：

1. 计算至最后的ReLU层得到$\vec{u}$；
2. 进行矩阵乘法$\vec{z}=\vec{u}^T\vec{W}$；
3. 进行指数运算$exp(\vec{z})$；
4. 归一化$\vec{y}=exp(\vec{z})/\|\|exp(\vec{z})\|\|_1$；
5. 按$y_j$进行倒序取Top-N作为召回结果；

观察到，由于指数运算具有单调性，且在进行召回时只关注模型输出的相对值，而不关注绝对值；我们发现3、4两步可以省略掉，直接在计算出${\vec{z}}$之后，取$z_j$的值来作为排序的依据即可。

由于视频数量巨大，$\vec{z}=\vec{u}^T\vec{W}$这一步仍然存在高昂的计算成本。为了提升效率，在完成了模型训练之后，可以提前把$\vec{W}$拆成一个个列向量$\vec{v_j}$。

线上serving时，计算出用户向量$\vec{u}$之后，下一步就变成了寻找与$\vec{u}$内积最大的N个列向量${\vec{v_j}}$的问题。而这可以转化为最近邻搜索问题（作者引用论文：[An investigation of practical approximate nearest neighbor](http://www.cs.cmu.edu/~agray/approxnn.pdf)）。

# 二、排序

![Ranking]({{ site.url }}/assets/youtube-dnn-ranking.jpg)

## 1. 问题定义

YouTube的推荐系统中，将排序问题转化为一个对视频观看时长的预测问题。

## 2. 数据

## 3. 特征

### 3.1. "video embedding"

### 3.2. "language embedding"

### 3.3. "time since last watch"

### 3.4. "number of previous impressions"


# 参考

1. [论文地址](https://dl.acm.org/doi/10.1145/2959100.2959190)
2. [王喆-1](https://zhuanlan.zhihu.com/p/52169807)
3. [王喆-十个工程问题](https://zhuanlan.zhihu.com/p/52504407)
4. [王喆-模型Serving](https://zhuanlan.zhihu.com/p/61827629)
5. [工程再现](https://zhuanlan.zhihu.com/p/38638747)

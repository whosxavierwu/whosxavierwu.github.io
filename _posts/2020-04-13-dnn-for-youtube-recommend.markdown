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

一个显而易见的问题是：当类别数量多达数百万时，如何使模型仍然能有效地进行学习？YouTube所选择的解决方案是 Negative Sampling，这点留待数据与特征之后，再进行展开。

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

> Inspired by continuous bag of words language models [14], we learn high dimensional embeddings for each video in a fixed vocabulary

从用户的视频观看历史中挖掘特征主要分两步：
1. 通过单独的模型预训练好每个视频的embedding。
2. 取出用户历史（*All or Top-k?*）观看的视频的embedding取均值，作为 "watch vectors"。

> Importantly, the embeddings are learned jointly with all other model parameters through normal gradient descent backpropagation updates 

*具体而言，是如何做embedding的？*

#### "search vector"

> Search history is treated similarly to watch history - each query is tokenized into unigrams and bigrams and each token is embedded. Once averaged, the user’s tokenized, embedded queries represent a summarized dense search history

从用户的搜索历史中挖掘特征的步骤，与前面相似：
1. 将每个query分词成unigrams跟bigrams，而token又是被embedding好的，
2. 汇总所有的这些embedding求均值，作为 "search vector"

*具体而言，是如何做embedding的？*

### 3.3 事件时间特征

"Example Age" 是个较为特殊的特征。引入这个特征，是因为作者观察到，用户更偏好新产的视频。

> we feed the age of the training example as a feature during training. At serving time, this feature is set to zero (or slightly negative) to reflect that the model is making predictions at the very end of the training window.

论文在一张插图的描述中提到：

> the example age is expressed as $t_{max} - t_N$ where $t_{max}$ is the maximum observed time in the training data.

$t_N$指的是样本打标签的时间，也就是当前的事件的时间戳，这个好理解。

但说得模糊的是，$t_{max}$到底指的是全体训练样本中的最大观测时间？还是当前样本事件发生之前的最大观测时间？

结合前面所说的，在serving时，该特征被置为零，我更倾向于是前者。

另一个细节就是，时间距离用的是秒？分钟？小时？还是天？

作者通过一张图来说明 "Example Age" 的有效性：

![Example Age]({{ site.url }}/assets/youtube-dnn-example-age.jpg)

## Negative Sampling

*TODO：具体细节？*

## 4. 模型上线

"video vectors"具体含义：
需要单独的embed video vector，还是延用最下方的embedded video watches里面的已经embed好的结果？

online serving 为什么不直接用模型进行预测？而是采用 nearest neighbor search ?

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

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

假设用$U$表示这次事件中的用户、用$C$表示上下文，用$u$表示对$U$、$C$一起进行embedding后的特征向量，用$v_i$表示对视频$i$进行embedding后的特征向量。则我们需要预测的分类到视频$i$的概率可以形式化如下：

$$
P(w_t=i|U,C)=\frac{e^{v_iu}}{\sum_{j\in V}e^{v_ju}}
$$

通过这样的问题转化以后，理想情况下，我们能预测到在当前用户、当前情景下，每个视频被观看的概率。取概率最高的前M个视频即可作为召回模块的输出。

随之而来的一个问题是：当类别数量多达数百万时，如何使模型仍然能有效地进行学习？YouTube所选择的解决方案是 Negative Sampling。

## Negative Sampling

？

## 2. 数据准备

在准备数据样本时，需要注意：

1. 应采用所有YouTube视频的观看事件（如嵌在其他网站的），而不仅仅是YouTube主站上的。
2. 采用 Negative Sampling，论文中提到，一般会采样出数千负样本。
3. 应该限制每个用户带来的训练样本的数量。
4. 注意避免样本的时间穿越问题，核心是只能用打标之前的数据来做feature。

![训练数据筛选]({{ site.url }}/assets/youtube-dnn-dataset.jpg)

## 3. 特征处理

主要包括以下一些特征：

### 3.1. "watch vector"，用户最近观看过的视频；

在固定好历史watch的长度，比如过去20次浏览的video，可能存在部分用户所有的历史浏览video数量都不足20次，在average的时候，是应该除以固定长度（比如上述例子中的20）还是选择除以用户真实的浏览video数量？

**embedding是如何做的？**

### 3.2. "search vector"，用户最近的搜索；

同样有上面的问题

### 3.3. "geographic embedding"，用户所处的地理位置、所使用的设备等；



### 3.4. "Example Age"，样本日期



### 3.5. 其他如性别等。

![Example Age]({{ site.url }}/assets/youtube-dnn-example-age.jpg)

作者在论文中给出了不同特征组合的效果对比：

![Features]({{ site.url }}/assets/youtube-dnn-feature-select.jpg)

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

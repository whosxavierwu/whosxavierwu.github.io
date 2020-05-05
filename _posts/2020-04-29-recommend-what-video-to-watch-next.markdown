---
layout: post
title:  "Recommending What Video to Watch Next: A Multitask Ranking System"
date:   2020-04-29 22:55:00 +0800
categories: recommender
---

# 概述

在该论文中，YouTube提出了一种双塔结构的排序模型，新颖的点主要在于：
1. 模型是基于多目标进行训练的；
2. 借鉴广告CTR中的思想，引入了用于消除位置偏差的浅层网络；
3. 引入了MMoE，带门控的专家网络；

需要注意的一点是，该论文的应用场景与YouTube之前那篇经典论文["Deep Neural Network for YouTube Recommendation System"](https://whosxavierwu.github.io/recommender/2020/04/12/dnn-for-youtube-recommend.html)不同，之前的推荐场景是在首页推荐频道中，而本文的推荐场景是发生在，用户正在看或看完一个视频以后，紧接着YouTube进行推荐。

![Watch Next]({{ site.url }}/assets/youtube-watch_next.png)

# 一、召回

召回部分，文中说得非常笼统，用了一小段话就给带过了，这里只能逐句逐句来进行解析了：

> Our video recommendation system uses multiple candidate generation algorithms, each of which captures one aspect of similarity between query video and candidate video. 

如果没有留意到业务应用场景的话，可能会对"query video"这个词感到懵逼，毕竟一般的推荐场景中不会有"query"这种东西。

这里的推荐是在用户看某个视频的时候就在旁边给他出一个推荐清单。由于正在观看的视频极有可能是用户当前感兴趣的内容，我们认为与该视频相关的视频也更有可能被点击观看，所以在推荐时需要考虑"watching video"，也就是"query video"。按这个思路，我们还能想到，其实可以用基于物品的协同过滤算法来作为baseline。

通过不同算法来捕捉"query video"和"candidate video"在不同方面的相似度。具体有哪些方面呢？

> For example, one algorithm generates candidates by matching topics of query video. 

> Another algorithm retrieves candidate videos based on how often the video has been watched together with the query video. 

作者举例说了两种，也是比较容易想到的：视频在主题上的相关性；两个视频间共现频率；

简单发散一下，我能想到的还有：
+ 两个视频的UP主之间的相关性；
+ 两个视频在封面、标题上的相关性；
+ 两个视频在时长、观看人数上的相关性；
+ ……

> We construct a sequence model similar to [10] for generating personalized candidate given user history. 

借鉴了论文"Deep Neural Network for YouTube Recommendation System"（[论文地址](https://dl.acm.org/doi/10.1145/2959100.2959190)、[博客地址](https://whosxavierwu.github.io/recommender/2020/04/12/dnn-for-youtube-recommend.html)）的做法，来基于用户观看历史构建序列模型（类似于Word2Vec），进而生成候选视频。

> We also use techniques mentioned in [25] to generate context-aware high recall relevant candidates.

使用了"Efficient Training on Very Large Corpora via Gramian Estimation"（[论文地址](https://arxiv.org/abs/1807.07187)）中的技术来生成与上下文相关的高召回候选集合。

# 二、排序

## 1. 问题定义

前面说到，本文是对多目标进行建模，所以对问题、优化目标的定义是和通常情况下的不太一样。

通常情况下，我们的模型训练目标是点击CTR或视频观看时长，抽象来看，是对用户参与程度的反映。YouTube的该论文认为，除了参与度"engagement objectives"以外，推荐系统还应该考虑用户的满意度，也就是"satisfaction objectives"。满意度可以从用户对所推荐的视频点击like或者dismiss的操作来体现。

## 2. 数据准备

## 3. 特征处理

## 4. 模型训练

### 4.1 整体结构

### 4.2 MMoE

### 4.3 位置消偏

# 结语

# 参考

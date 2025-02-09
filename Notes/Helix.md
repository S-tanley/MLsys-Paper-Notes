## Abstract

high

throughput, low-latency large language model (LLM) serving on heterogeneous GPU clusters。

我觉得这个还真的是，挺有意思的，因为我之前就一直有在想，通常来讲大家用的集群都是比较统一的，但是用这种比较统一的集群很显然就是会比较贵，如果随便有什么GPU，就加到cluster里，然后还能用的话，听起来就很平民。

 a *max-flow* problem for a directed, weighted graph

那这个就是很经典的优化问题了。

很自然，他们就用到了mixed integer linear programming (MILP) algorithm，很巧这学期学了linear programming或者说是linear optimization。

前面那个图又让我想到tensorflow这个框架，就tensorflow就是出一个计算图，这样的话可以做很多优化，就所谓的 optimize model placement and request scheduling，不知道这两者有没有什么相同之处。

## **1 Introduction**

模型太大，部署太贵。

跟我之前想的少有出入，但是和新想法是一样的。他这里想的是cloud厂商有很多heterogeneous GPU在各个data center里。我想的是个人算力了，就是个人算力肯定是乱七八糟的，但也是分布式系统，而且如果能整合的话我觉得应该会很便宜了。

introduction里有很多background，然后还有很多related work。他这里说的是之前的工作主要还是focus on training而不是inference。

之前说过的pipeline model parallelism，就是会有这个算力分配问题，如果你用heterogeneous的system，甚至来讲可能更适合。

GPU and network heterogeneity

解决模型并行问题，解决request scheduling

## **2 Background**

Most of today’s LLMs adopt a decoder-only Transformer

architecture

介绍Transformer，模型怎么输出。

memory demand，要存中间states，存在kv-cache里。很多方法用来减少kv-cache。




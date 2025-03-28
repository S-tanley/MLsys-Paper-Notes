---
title: Reading Notes for SmartMoE
date: 2025-03-27
categories: Reading Paper
tags: MLSys
---

# Summary

## Abstract & Introduction & Background and Motivation

Deep neural network（DNN）现在越来越大，除了dense model，就是比较传统的model之外，越来越多的人开始关注sparsely activated model。针对dense model，之前有很多auto-parallelization的方法，但是这些方法对sparsely activated model，比如说MoE架构的模型就没那么好用了。所以他们主要做的就是实现对sparsely activated model做自动并行的分布式训练的方法。

Intro就先说一下来龙去脉，就众所周知，scaling law目前对DNN一直没有失效，所以各家基本上就是一直往上堆参数。但模型变大了就练不动了，所以就要找efficient method去做训练。一种就是从系统上，你去想办法去做并行训练，这样相同的计算资源，相同的时间，你能训练更大模型。要不然就是你对模型做改进，比如说MoE，虽然参数量很大，但因为每次训练的时候都只用activate一部分参数，所以就训练开销会降下来。

他们想说的就是没人把这两个结合起来，或者说有人在做MoE的时候，顺便针对他们的MoE有做这种并行训练的优化，但是没有太多去做通用一些的，automatic的并行方法给MoE。作者就说现有的模型都用是specific expert parallelism，但是这个方法又会影响training efficiency，是有一些方法来降低这个expert parallelism的cost，但是他们都“require special system expertise”，其实就还是再讲通用性的问题。然后又讲了自动化的问题，就说现有的automatic parallelization training systems都是为dense model服务的。这些基本都是在说research gap，就当前这个领域的空缺，或者也是一个为什么作者他们要做这个的原因。

<img src="./SmartMoE/CleanShot 2025-03-27 at 14.51.39.png" alt="CleanShot 2025-03-27 at 14.51.39" style="zoom:67%;" />

这里放了个图，大概展示了一下dense model和MoE model的区别。

MoE有这么一个比较独特的结构，肯定就有一些独特的property。这里作者强调的是“dynamic and imbalanced property”。其实也很好理解，每次不一样的数据，gate选的expert就不一样，就很明显dynamic。然后肯定有的experts用的多，有的experts用的少，这就是imbalanced。而且这些都是和每次的数据直接相关的，所以他们强调了一个概念：**data-sensitive**。

既然MoE有data- sensitive的性质，那么当前的parallelization

approaches肯定就有不好的地方。这里说了两个limitation：

* **Limited Optimization Space**

  这个说的是，data- sensitive可以让我们做更多的优化，但是现在的并没有考虑这么多可以做优化的地方，因为他们是base dense model的。

* **Large Searching Overhead**

  MoE因为data- sensitive，所以每次各个expert的workload都不一样，所以其实每次的execution plan应该也都不一样，所以我们希望可以每次都换到一个合适的execution plan。但是现在方法，用一些规划方法，算plan的时间太久了，很明显就没法满足这种比较高的换plan的方法。

他们的方法就是我先算一堆比较好的plan，叫做**static pool**，然后又用了一个很快的算法去实时从这个pool里选plan。同时，他们还实现了一种机制，我也不知道该不该叫它一种机制，叫做awareness of workload。反正就是可以知道每个experts的workload，这样根据这个我们就可以选那些experts放到一个机器上，或者我们应该怎么并行。

在background里还介绍了各种并行，什么data parallelism（DP）， pipeline model parallelism（PP）， tensor model parallelism（TP）， expert parallelism（EP）。反正关于并行这里，就名称不统一，基本上就是横着切模型（PP），竖着切模型（TP），或者复制很多歌模型（DP）。EP是“a combination of Data Parallelism and Tensor Model Parallelism specialized for the MoE scenario”。这里也给了图，也算清楚吧。

<img src="./SmartMoE/CleanShot 2025-03-27 at 16.07.46.png" alt="CleanShot 2025-03-27 at 16.07.46" style="zoom:67%;" />

主要还是图c，MoE part竖着切的，所谓的TP，但是Multi-Head那里又是复制了很多块，所以可以做DP。其实很合理的，因为你想MoE每次只用一个expert，但是前面的attention操作每一次都要做。所以前面每次都要做的我们用DP，后面E个expert只选一个的我们用TP。如果运气好，我们E个data做数据并行，到gate的时候正好assign到E个不同的experts，十分完美。

当然我们基本都用**Hybrid Parallelism**，就把不同的parallelism拼起来，你是怎么拼的，就是一个 **parallel execution plan**。也有一些automatic parallelization，但是time- consuming。

他们总结了关于提出一个automatic parallelization training system的难点：

* **Space of Hybrid Parallelism**

  一个sytem不一定可以用所有的并行技术，你要知道这个space有多大，而且要尽量让他大。

* **Performance Modeling**

  怎么知道你选出来的plan好还是差。

* **Searching Algorithm**

  怎么从space里快速的选。

<img src="./SmartMoE/CleanShot 2025-03-27 at 16.30.16.png" alt="CleanShot 2025-03-27 at 16.30.16" style="zoom:67%;" />

Figure 3我感觉就是非常清楚的说了为什么要change plan，为什么要workload aware performance modeling。

## Overview & Enlarged Space for Hybrid Parallelism

"Beyond prior works that generate optimal execution plans based on *model* architecture and *hardware* specification, we take the **workload** into account for data-sensitive models."

<img src="./SmartMoE/CleanShot 2025-03-28 at 15.37.41.png" alt="CleanShot 2025-03-28 at 15.37.41" style="zoom:67%;" />

这里分阶段讲了一下Two-Stage Auto-Parallelization。

第一个阶段是**Offline Pool Construction**。就是构建一个plan **pool**，都是一些比较好的execution plans。需要注意的就是他专门选的是那种相互之间比较好换的，要不然肯定不可行。

第二个阶段就是**Online Adaptive Parallelization**。这个时候就是已经开始训练了，就是用一个轻量级的算法去实时的找合适的execution plan。

这里有个概念感觉要提一下，**expert slot**，就是对于每一个expert（FFN）是用的什么并行。

<img src="./SmartMoE/CleanShot 2025-03-28 at 16.08.47.png" alt="CleanShot 2025-03-28 at 16.08.47" style="zoom:70%;" />

还有一个concrete example，但其实我有点没看懂。

<img src="./SmartMoE/CleanShot 2025-03-28 at 17.26.05.png" alt="CleanShot 2025-03-28 at 17.26.05" style="zoom:67%;" />

还说了一个**expert placement**，我感觉这个和Helix[2]做的差不多，只不过Helix主要是在说inference，这个主要说的是训练而已。

## Offline Pool Construction &  Online Adaptive Parallelization



## Evaluation & Related Work & Conclusion



# Thoughts

## About their method

他这个方法我感觉主要应该还是侧重在这个automatic这一块，就是想要提出一个比较通用的方法，他在文章里我感觉也是一直在暗暗强调。

他们方法感觉也就是他们自己说的那个contribution：

1. 更多的并行组合方法，所谓的“enlarge the combination space of hybrid parallelism”。
2. Offline构建一个execution pool，online的时候用一个light-weight算法能实时从pool里选好的。
3. 实现2需要“awareness of workload”，还需要一个light- weight算法。

基本上就是这些。

## About MoE Framework

有关MoE架构那里，有一点点跟我之前的理解稍有出入。他的MoE网络每一次经过gate，只会assign到**一个**最合适的expert。我之前的理解是可以随便组合expert，尤其是这个explicitly说了，一个FFN就是一个expert，也就是说expert之间应该是没有communication的。我之前的理解有点像backbone，backbone就是随机冻住一部分参数，就也是为了能够去训练比较大的网络。然后MoE的就是网络只有第一层是全连接，后面的不是全连接，但是我们gate去选expert其实就是选FFN的第一层的node。但不得不说现在看完他这种独立的说法，我觉得我以前的理解确实是有点误区，毕竟就是如果不是全连接的话，你怎么连接又是个问题。我现在就每个FFN就是个expert，然后都一样都是全连接网络，相互独立的，然后每次选一个最合适的，确实就是在实际训练的时候可行性更高。尤其是你要memory efficient，你有的时候要offload到别的地方去储存，然后在用的时候再load回来，确实需要一个个expert独立。

不过我现在就在想，每一个FFN反正就是都是全连接嘛，最后我把一些week connection直接drop掉呢。不知道distill是不是这么干的。

## About Expert Parallelism

就是如果我们有这个gata了，其实就是说我们大概知道每个expert在一个workload里被用几次，那有没有可能我们把用的次数多的expert复制几个，就他们在做一个小的DP，这样是不是当用几个data同时assign到同一个expert的时候不用等了。或者至少inference的时候我们可以这么搞。

他这个Figure 5，我也是有点小问题，每一层的experts一定要一样数量，呀这个layer的意思是一个expert里面的层数还是experts的层数，有点不清楚，但是我还是比较倾向于一个expert（FFN）的层数

## Confusion & Guess

这篇文章里比较重要的一个执行单元是“worker”，但其实我也没太懂他这一个worker到底是什么，我猜可能是一个GPU或者一个node。如果这样的话反正应该也要有一个assign任务的节点了。



# Reference

[1]: SmartMoE
[2]: Helix


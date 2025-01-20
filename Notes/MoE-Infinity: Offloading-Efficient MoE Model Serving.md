# MoE-Infinity: Offloading-Efficient MoE Model Serving[^1]

## Content

**Abstract**:

Confused about one important concept, mixture-of-experts (MoE) models and sparse mixture-of-experts (MoE) models.

Watched a video[^2], MoE itself is a sparse model, which means only parts of the model are involved in the calculation each time. This leads to higher efficiency.

Mainly about efficiency, cost, and latency. The MoE-Infinity is more efficient, cheaper, and has lower latency.

**1. Introduction**:

The big network contains small networks. It just reminds me of the multitask models. 

Current MoE models need massive GPU resources. Offloading is a good way to overcome it: only load experts to the GPU memory when needed.

There are many models that applied offloading but their performance is not good: high latency.

The aim of this paper is to optimize offloading efficiency. Some observations and work need to be done.

Contributions.

**2. Background and Motivation**:

explain the MoE model and some evaluation standards.

High latency, elaborate what it has said in the introduction.

**3. Exploiting Expert Activation Awareness**:

Propose three approaches to address the three issues mentioned above.

Two open concerns, tracing activated experts and expert prefetching and caching. Request level of tracing, and dynamically adjust its prefetching and caching priorities.

**4. Request-Level Expert Activation Tracing**

Confused

EAM stores what experts are activated, we store EAMs into a fixed size EAMC, then calculate the distance between the current EAM with the EAMs in the EAMC, then find the most similar and do more accurate prefetch and cache.

One question is if you already know the current EAM, why don't you just fetch things according to this. Why do you still need to find a most similar one to replace? One reasonable explanation is that we have a very big EAM, record in every layer what expert will be activated, and we only use first a few layers' dates to calculate the similarity then we can predict the future activated experts of the current EAM.

One future work is to find an efficient cluster algorithm to calculate the similarity.

**5. Activation-Aware Prefetching & Caching**

Detail for prefetching and caching. 

? NUMA nodes, DMA operations, kv-cache

Future work:  multi-server support

**6. Practical Concerns**

Idk why this is a paragraph.

**7. Evaluation**

experienment settings.

Future work: how to enable better caching for the

initial iterations

Figures are important.

? Tail latency, CDF graph

**8. Conclusions**



## Notes

其实就是解决效率问题的，他的主要提升途径是提升prefetching和caching。写了非常多evaluation。

[^1]: Leyang Xue, Yao Fu, Zhan Lu, Luo Mai, and Mahesh Marina. 2024. MoE-Infinity: Offloading-Efficient MoE Model Serving. Retrieved September 9, 2024 from [http://arxiv.org/abs/2401.14361](https://urldefense.com/v3/__http:/arxiv.org/abs/2401.14361__;!!Mak6IKo!JGsmunK4816F9qTIzFh0QAl_Tgr55CX5aR2IjjIRrsDaJvPzsI_qdQNouSpg-XfJEGex1dhRSBnoetD1NKmblw$)
[^2]: https://www.bilibili.com/video/BV1Ep421m7cx/?spm_id_from=333.337.search-card.all.click&vd_source=c78e36117913d8047d79be34ab257bec

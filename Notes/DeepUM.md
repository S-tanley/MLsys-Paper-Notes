# DeepUM: Tensor Migration and Prefetching in Unified Memory[^1]

## Content

**Terms**:

DNNs: Deep neural networks

**ABSTRACT**:

DeepUM that exploits CUDA Unified Memory (UM) to allow GPU memory oversubscription for DNNs

 a new correlation **prefetching** technique to hide the page migration overhead

**1. INTRODUCTION**

DNNs are very big and need tremendous computation resources.

Fortunately, Researchers can fine-tune those large models with limited resources.

However, the SOTA models are too big to even fine-tune.

They focus on GPU memory swapping to solve the memory capacity problem of DNNs.

Two ways: UM and pure GPU memory with swapping-in/swapping-out

memory objects

? memory fragmentation issues

It prefetches pages based on the information in the correlation tables by predicting which kernel will execute next.

two optimization techniques: page pre-eviction and page invalidation



**8. CONCLUSIONS**

No code modification is needed.

performance is good.

## Notes





[^1]: Jaehoon Jung, Jinpyo Kim, and Jaejin Lee. 2023. DeepUM: Tensor Migration and Prefetching in Unified Memory. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, January 27, 2023. ACM, Vancouver BC Canada, 207–221. [https://doi.org/10.1145/3575693.3575736](https://urldefense.com/v3/__https:/doi.org/10.1145/3575693.3575736__;!!Mak6IKo!JGsmunK4816F9qTIzFh0QAl_Tgr55CX5aR2IjjIRrsDaJvPzsI_qdQNouSpg-XfJEGex1dhRSBnoetClEEIKig$)
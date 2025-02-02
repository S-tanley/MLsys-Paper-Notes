# Task-Specific Expert Pruning for Sparse Mixture-of-Experts[^1]

## Content

**Abstract**: 

MoE is hard to be deployed on cloud or mobile environment.

resource-limited downstream tasks

propose a general method to

progressively drop the non-professional experts for the target downstream task

**1. Introduction**

The contribution of each expert activated may be different. 

convert large sparse MoE models into small single-expert dense model by task-specific expert pruning method.

Having a threshold, and just deleting experts which do not meet the threshold. In the end, we want one expert/layer.

**6. Conclusion**:

this new fine-tuning paradigm could preserve most benefits of the pre-trained MoE models and much better than the densely pre-trained counterparts

## Notes









[^1]: Tianyu Chen, Shaohan Huang, Yuan Xie, Binxing Jiao, Daxin Jiang, Haoyi Zhou, Jianxin Li, and Furu Wei. 2022. Task-Specific Expert Pruning for Sparse Mixture-of-Experts. [https://doi.org/10.48550/arXiv.2206.00277](https://urldefense.com/v3/__https:/doi.org/10.48550/arXiv.2206.00277__;!!Mak6IKo!JGsmunK4816F9qTIzFh0QAl_Tgr55CX5aR2IjjIRrsDaJvPzsI_qdQNouSpg-XfJEGex1dhRSBnoetCSU053XA$)
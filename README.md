# Implementation of classical FL algorithms

---

## Availabel Dataset

- CIFAR-10
- CIFAR-100
- SVHN

### Available data partition type

- IID
- Dirichlet Distribution (\alpha)
- Pathlogical distribution 

> ## Implemented Algorithms

As Fine-tuning the aggregated model at server side is an effective way to reduce the influence of Non-IID data, the implementation supports the comparison between w/ ft and w/o ft by setting different args.ft_data_per_cls.

**FedAvg** — [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html) *AISTATS 2017*

  
> ***Regularization-based tFL***

- **FedProx** — [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) *MLsys 2020*

- **FedDyn** — [Federated Learning Based on Dynamic Regularization](https://openreview.net/forum?id=B7v4QMR6Z9w) *ICLR 2021*

- **SCAFFOLD** - [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a.html) *ICML 2020*


> ***Knowledge-distillation-based tFL***
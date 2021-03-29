# Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding

This repository contains the code release for:

**Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding.**  
David Klindt*, Lukas Schott*, Yash Sharma*, Ivan Ustyuzhaninov, Wieland Brendel, Matthias Bethge&dagger;, Dylan Paiton&dagger;
https://arxiv.org/abs/2007.10930

An example latent traversal using our learned model: <br/>
![Sample traversal](https://github.com/bethgelab/slow_disentanglement/blob/master/latent_factors.gif?raw=true)


**Abstract:** We construct an unsupervised learning model that achieves nonlinear disentanglement of underlying factors of variation in naturalistic videos. Previous work suggests that representations can be disentangled if all but a few factors in the environment stay constant at any point in time. As a result, algorithms proposed for this problem have only been tested on carefully constructed datasets with this exact property, leaving it unclear whether they will transfer to natural scenes. Here we provide evidence that objects in segmented natural movies undergo transitions that are typically small in magnitude with occasional large jumps, which is characteristic of a temporally sparse distribution. We leverage this finding and present SlowVAE, a model for unsupervised representation learning that uses a sparse prior on temporally adjacent observations to disentangle generative factors without any assumptions on the number of changing factors. We provide a proof of identifiability and show that the model reliably learns disentangled representations on several established benchmark datasets, often surpassing the current state-of-the-art. We additionally demonstrate transferability towards video datasets with natural dynamics, Natural Sprites and KITTI Masks, which we contribute as benchmarks for guiding disentanglement research towards more natural data domains.

### Cite
If you make use of this code in your own work, please cite our paper:
```
@inproceedings{klindt2021towards,
title={Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding},
author={David A. Klindt and Lukas Schott and Yash Sharma and Ivan Ustyuzhaninov and Wieland Brendel and Matthias Bethge and Dylan Paiton},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=EbIDjBynYJ8}
}
```

### Datasets
Our work also contributes two new datasets. <br/>
The Natural Sprites dataset can be downloaded here: https://zenodo.org/record/3948069 <br/>
The KITTI Masks dataset can be downloaded here: https://zenodo.org/record/3931823


### Acknowledgements

The repository is based on the following [Beta-VAE reproduction](https://github.com/1Konny/Beta-VAE). The MCC metric was adopted from the [Time-Contrastive Learning release](https://github.com/hirosm/TCL).

### Contact

- Maintainers: [David Klindt](https://github.com/david-klindt) & [Lukas Schott](https://github.com/lukas-schott) & [Yash Sharma](https://github.com/ysharma1126)

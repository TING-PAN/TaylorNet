# Taylor saves for later: disentanglement  for video prediction using Taylor representation
Official implementation for Neurocomputing paper:  Taylor saves for later: disentanglement  for video prediction using Taylor representation, by Ting Pan, Zhuqing Jiang, Jianan Han, Shiping Wen, Aidong Men, Haiying Wang.

# Requirements
* Python 3.6.12
* torch 1.6.0
* torchvision 0.7.0
* cuda 10.1

# Datasets
Please  move them to the ./dataset.

**mmnist**: [test](http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy)
            [train](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)

**TaxiBJ**: [TaxiBJ](https://pan.baidu.com/s/1Ttc2T1mFD_HyEUJu1dSYQQ)    **password**: lkck

**Human3.6**: https://github.com/Yunbo426/MIM/tree/master/data

# Pre-trained Models
Please  move them to the ./weights.

[Pre-trained Models for mmnist, TaxiBJ and Human3.6](https://pan.baidu.com/s/1356mLd6lW7AI_cmqXaDM2g) 

**password**: iedf

# Abstract
Video prediction is a challenging task with wide application prospects in meteorology and robot systems. Existing works fail to trade off short-term and long-term prediction performances and extract robust latent dynamics laws in video frames. We propose a two-branch seq-to-seq deep model to disentangle the Taylor feature and the residual feature in video frames by a novel recurrent prediction module (TaylorCell) and residual module, based on a novel principle for feature separation. TaylorCell can expand the video frames' high-dimensional features into the finite Taylor series to describe the latent laws. In TaylorCell, we propose the Taylor prediction unit (TPU) and the memory correction unit (MCU). TPU employs the first input frame's derivative information to predict the future frames, avoiding error accumulation. MCU distills all past frames' information to correct the predicted Taylor feature from TPU. Correspondingly, the residual module extracts the residual feature complementary to the Taylor feature. Due to the characteristic of the Taylor series, our model works better on datasets with short-range spatial dependencies and stable dynamics. On three generalist datasets (Moving MNIST, TaxiBJ, Human 3.6), our model reaches and outperforms the state-of-the-art model in the short-term and long-term forecast, respectively. Ablation experiments demonstrate the contributions of each module in our model.

# Citation
Please cite the following paper if you find this repository useful.

    @article{PAN2022166,
            title = {Taylor saves for later: Disentanglement for video prediction using Taylor representation},
            journal = {Neurocomputing},
            volume = {472},
            pages = {166-174},
            year = {2022},
            issn = {0925-2312},
            doi = {https://doi.org/10.1016/j.neucom.2021.11.021},
            url = {https://www.sciencedirect.com/science/article/pii/S0925231221016957},
            author = {Ting Pan and Zhuqing Jiang and Jianan Han and Shiping Wen and Aidong Men and Haiying Wang},
            }


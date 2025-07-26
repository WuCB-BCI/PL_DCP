# PL-DCP: A Pairwise Learning framework with Domain and Class Prototypes for EEG emotion recognition under unseen target conditions 
*   A Pytorch implementation of our under reviewed paper "PL-DCP: A Pairwise Learning framework with Domain and Class Prototypes for EEG emotion recognition under unseen target conditions".
# Installation
*   Python 3.8
*   Pytorch 2.0.0
*   NVIDIA CUDA 11.8
*   NVIDIA CUDNN 8700
*   Numpy 1.24.3
*   Scikit-learn 0.22.1
*   scipy 1.5.2 
*   GPU NVIDA GeForce RTX 3090
# Databases
*   [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html ""), [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html ""), [SEED-V](https://bcmi.sjtu.edu.cn/~seed/seed-v.html "") 
# Training
*   Data Process Module: DataProcess.py
*   Model framework definition file: DemoModel.py
*   Pairwise Learning module definition file : Pairwise_Learning.py
*   Datakit Module: utils.py
*   Pipeline of the PL_DCP : unseen_test.py
# Usage
*   After modify setting (path, etc), just run the main function in the unseen_test.py.

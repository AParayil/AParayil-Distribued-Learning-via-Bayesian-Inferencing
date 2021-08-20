# AParayil-Distributed-Learning-via-Bayesian-Inferencing
This repository includes implementation of our Neurips work ["Decentralized Langevin Dynamics for Bayesian Learning"](https://proceedings.neurips.cc/paper/2020/file/b8043b9b976639acb17b035ab8963f18-Paper.pdf) along with some of the additional visualizations included in the paper. The work looks at federated learning from probabilistic perspective and approaches the problem as inference of global posterior. 
For an overview on the state-of-the-art of federated optimization, please refer: [A Field Guide to Federated Optimization](https://arxiv.org/pdf/2107.06917.pdf)

Requirements: 
-------------
python 3.x, numpy, matplotlib, pytorch (>=1.0), torchvision, scikit-learn



Main file to run experiments: 
-----------------------------

Toy Data (Parameter estimation for Gaussian mixture) - GMM Centralized ULA - toy_data_sgld.py, Decentralized ULA - toy_data_dist_sgld.py

Logistic Regression (Using UCI a9a dataset): Centralized ULA - a9a_sgld.py, Decentralized ULA - a9a_dist_sgld.py

Image Classification( MNIST) SGD - mnist_sgd_svhn_pred_scores.py, Centralized ULA - mnist_sgld_svhn_pred_scores.py, Decentralized ULA - mnist_dist_sgld_svhn_pred_scores.py

All content in this repository is licensed under the MIT license.


Official implementation submitted for the Neurips submission can be found at https://github.com/pkgurram/d-ula




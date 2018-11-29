# Learning to Reason with Third-Order Tensor Products
This repository contains the code accompanying the paper [*Learning to Reason with Third-Order Tensor Products*](https://papers.nips.cc/paper/8203-learning-to-reason-with-third-order-tensor-products) published at NeurIPS, 2018. It encompasses our implementation of the Tensor Product Representation Recurrent Neural Network (TPR-RNN) applied to the bAbI tasks with SOTA results. A download script for a pretrained model is provided.

# Requirements
- Python 3
- Tensorflow==1.11.0
- Seaborn==0.9.0
- Sklearn==0.0

Make sure to upgrade pip before installing the requirements.
```bash
pip install --upgrade pip
pip install -r requirements.txt
sh download_data_and_model.sh
```

# Usage
Run the pre-trained model.
```bash
python3 evaluation.py
```

Train from scratch. (Look at the train.py files for details)
```bash
python3 train.py
```

Generate small hierarchically clustered similarity matrices of a random set of sentences using different internal representations and the cosine similarity.
```bash
python3 cluster_analysis.py
```

# Results
With this code we achieved the following error (rounded to two decimal places) when trained on all bAbI tasks simultaneously. In the appendix of the paper we provide a breakdown per task.

task|run-0|run-1|run-2|run-3|run-4|run-5|run-6|run-7|best|mean|std
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
all|1.50|1.69|1.13|1.04|0.78|0.96|1.20|2.40|0.78|1.34|0.52

# Citation
```
@inproceedings{schlag2018tprrnn,
  title={Learning to Reason with Third Order Tensor Products},
  author={Schlag, Imanol and Schmidhuber, J{\"u}rgen},
  booktitle={Advances in Neural Information Processing Systems},
  pages={10002--10013},
  year={2018}
}
```

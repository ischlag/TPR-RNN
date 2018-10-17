# Learning to Reason with Third-Order Tensor Products
This repository contains the code accompanying the paper Learning to Reason with Third-Order Tensor Products. The paper introduces Tensor Product Representation Recurrent Neural Network (TPR-RNN) applied to the bAbI tasks and achieves SOTA.

# Requirements
- python 3
- tensorflow==1.11.0
- seaborn==0.9.0
- sklearn==0.0

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
todo

# Citation
todo

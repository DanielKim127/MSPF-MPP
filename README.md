# MSPF-MPP: A Multi-view Self-supervised Pretraining Framework for Molecular Property Prediction


## 1. Abstract
Accurate molecular property prediction is fundamental to computational drug discovery and molecular design. However, existing methods often rely on a single molecular view such as SMILES or 2D molecular graphs, or require 3D conformations which are costly and unreliable. This paper proposes a multi-view self-supervised pretraining framework for molecular property prediction. We first pretrain a shared graph isomorphism network (GIN) on large-scale unlabeled molecular graphs through a node-correspondence task between original and perturbed graphs, and use the resulting atom embeddings, termed graph-matching positional encodings, to initialize the learnable positional channel of the 2D encoder as a data-driven and perturbation-robust structural prior. Subsequently, we use the pretrained weights of ChemBERTa2 to process the 1D SMILES, while self-supervised pretraining in this work is conducted on the 2D branch and the 2D-3D cross-view setting through AttrMask and cross-view alignment with molecule-level and substructure-level contrastive learning as well as variational representation reconstruction. In the downstream task, fingerprint, 1D, and 2D molecular representations are combined for prediction, and a consistency regularization term under noisy and missing-view perturbations is introduced to improve robustness to view degradation. Experiments on public molecular property prediction benchmarks show that our proposed method outperforms several single and multi-view baselines in terms of overall performance and generalization ability, validating the synergistic gains among graph-matching positional encoding, 2D-3D self-supervised pretraining, and multi-view fusion regularization (MVFR).


<img width="1159" height="823" alt="image" src="https://github.com/user-attachments/assets/c319d659-a6c2-4940-9a2e-a3783e9ef971" />


## 2. Installation
### (1) create virtual environment
$ conda create --name MSPF-MPP python=3.10

$ conda activate MSPF-MPP

### (2) install requirements
$ python -m pip install -r requirements.txt

### (3) clone the source code of MSPF-MPP
$ git clone https://github.com/DanielKim127/MSPF-MPP.git

$ cd MSPF-MPP


## 3. Dataset and Weight
PCQM4Mv2 can be downloaded from OGB-LSC: [https://ogb.stanford.edu/docs/lsc/pcqm4mv2/](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/). MoleculeNet datasets are available at [https://moleculenet.org/](https://moleculenet.org/). The ChemBERTa2 pretrained weights are available at [https://huggingface.co/DeepChem/ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM).   


## 4. Pretrained Weights
The repository includes two pretrained checkpoints: the GMPE graph-matching weights at checkpoints/gmpe/gmpe_pcqm4m/best.ckpt and the 2D-3D pretraining weights at runs/pretrain_2d3d_pcqm4m/best.pt.


## 5. Graph-matching Training and Generation
$ python tools/pretrain_align.py --dataset PCQM4M

$ python tools/encode_gmpe.py --csv_path <dataset_path> --ckpt checkpoints/gmpe/gmpe_pcqm4m/best.ckpt



## 6. 2D-3D Pretraining
$ python pretrain_2d3d.py --config configs/pretrain_2d3d.yaml



## 7. Fine-tuning
$ python finetuning.py --config configs/finetune_multiview.yaml --pretrain_ckpt runs/pretrain_2d3d_retrained/best.pt

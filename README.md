<div align="center">
    <p>
    <img src="./src/mamba_logo.png" alt="DepMamba Logo" style="height: 200px;">
    </p>
     <p>
    Official PyTorch code for training and inference pipeline for <br>
    <b><em>DepMamba: Progressive Fusion Mamba for Multimodal Depression Detection</em></b>
    </p>
    <p>
    </p>
    <a href="https://github.com/Jiaxin-Ye/DepMamba"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/Jiaxin-Ye/DepMamba"><img src="https://img.shields.io/badge/Python-3.8+-orange" alt="version"></a>
    <a href="https://github.com/Jiaxin-Ye/DepMamba"><img src="https://img.shields.io/badge/PyTorch-1.13+-brightgreen" alt="python"></a>
    <a href="https://github.com/Jiaxin-Ye/DepMamba"><img src="https://img.shields.io/badge/License-MIT-red.svg" alt="mit"></a>
</div>


### 📕Introduction

In this paper, we propose a **T**emporal-aware b**I**-direction **M**ulti-scale Network, termed **TIM-Net**, which is a novel temporal emotional modeling approach to learn multi-scale contextual affective representations from various time scales. 

## 📖 Usage:


### 1. Clone Repository

```bash
git clone https://github.com/Jiaxin-Ye/DepMamba.git
```

### 2. Requirements

Our code is based on Python 3.8 and CUDA 11.7. There are a few dependencies to run the code. The major libraries including Mamba and PyTorch are listed as follows:

```bash
conda create -n DepMamba -c conda-forge python=3.8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install packaging
git clone https://github.com/Dao-AILab/causal-conv1d.git 
cd causal-conv1d 
git checkout v1.1.3 
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
cd ..
git clone https://github.com/state-spaces/mamba.git
cd ./mamba
git checkout v1.1.3
MAMBA_FORCE_BUILD=TRUE pip install .
pip install -r requirement.txt
```

If you want to record training log, you need to login your own `wandb` account. 

```bash
$ wandb login
```

Change these lines in `main.py` to your own account.

```python
wandb.init(
    project="DepMamba", entity="<your-wandb-id>", config=args, name=wandb_run_name,
)
```

### 3. Prepare Datasets

We use the [D-Vlog](https://doi.org/10.1609/aaai.v36i11.21483) and [LMVD](https://arxiv.org/abs/2407.00024) dataset, proposed in this paper. For the D-Vlog dataset, please fill in the form at the bottom of the [dataset website](https://sites.google.com/view/jeewoo-yoon/dataset), and send a request email to the [author](mailto:yoonjeewoo@gmail.com). For the LMVD dataset, please download features on the released [Baidu Netdisk website](https://pan.baidu.com/s/1gviwLfbFcRSaARP5oT9yZQ?pwd=tvwa) or [figshare](https://figshare.com/articles/dataset/LMVD/25698351). 

Following D-Vlog's setup, the dataset is split into train, validation and test sets with a 7:1:2 ratio. For the LMVD without official splitting, we randomly split the LMVD with a 8:1:1 ratio and the specific division is stored in `./datasets/lmvd_labels.csv'.

### 5. Training and Testing

#### Training

```bash
$ python main.py --train True --train_gender both --test_gender both --epochs 120 --batch_size 16 --learning_rate 1e-4 --model DepMamba --dataset dvlog --gpu 0

$ python main.py --train True --train_gender both --test_gender both --epochs 120 --batch_size 16 --learning_rate 1e-4 --model DepMamba --dataset lmvd --gpu 0
```

#### Testing

If you want to test your model on 10-fold cross-validation manner with `X' random seed, you can run the following commands:

```bash
$ python main.py --mode test --data CASIA  --test_path ./Test_Models/CASIA_32 --split_fold 10 --random_seed 32
$ python main.py --mode test --data EMODB  --test_path ./Test_Models/EMODB_46 --split_fold 10 --random_seed 46
$ python main.py --mode test --data EMOVO  --test_path ./Test_Models/EMOVO_1 --split_fold 10 --random_seed 1
$ python main.py --mode test --data IEMOCAP  --test_path ./Test_Models/IEMOCAP_16 --split_fold 10 --random_seed 16
$ python main.py --mode test --data RAVDE  --test_path ./Test_Models/RAVDE_46 --split_fold 10 --random_seed 46
$ python main.py --mode test --data SAVEE  --test_path ./Test_Models/SAVEE_44 --split_fold 10 --random_seed 44
```

You can download our model files from our shared [link]( https://pan.baidu.com/s/1EtjhuljeHwvIjYG8hYtMXQ?pwd=HDF5) to `Test_Models` folder. 


## 📖 Citation

- If you find this project useful for your research, please cite [our paper](https://arxiv.org/abs/2211.08233):

```bibtex
@inproceedings{yedepmamba,
  title={Temporal Modeling Matters: A Novel Temporal Emotional Modeling Approach for Speech Emotion Recognition},
  author = {Ye, Jiaxin and Wen, Xincheng and Wei, Yujie and Xu, Yong and Liu, Kunhong and Shan, Hongming},
  booktitle = {ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, June 4-10, 2023},
  pages={1--5},
  year = {2023}
}
```

## 🙌🏻 Acknowledgement

- We acknowledge the wonderful work of [Mamba](https://arxiv.org/abs/2312.00752) and [Vision Mamba](https://arxiv.org/abs/2401.09417). 
- We borrow their implementation of [Mamba](https://github.com/state-spaces/mamba) and [bidirectional Mamba](https://github.com/hustvl/Vim). 
- We acknowledge [AllenYolk](https://github.com/AllenYolk/depression-detection) and [ConMamba](https://github.com/xi-j/Mamba-ASR).
- The training piplines are adapted from [SpeechBrain](https://speechbrain.github.io).

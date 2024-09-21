# RNADiffFold: Generative RNA Secondary Structure Prediction using Discrete Diffusion Models
![fig_1](/figures/fig1_overview.png)

RNA molecules are essential macromolecules that perform diverse biological functions in living beings. Precise prediction of RNA secondary structures is instrumental in deciphering their complex three-dimensional architecture and functionality. Traditional methodologies for RNA structure prediction, including energy-based and learning-based approaches, often depict RNA secondary structures from a static perspective and rely on stringent a priori constraints. Inspired by the success of diffusion models, in this work, we introduce RNADiffFold, an innovative generative prediction approach of RNA secondary structures based on multinomial diffusion. We reconceptualize the prediction of contact maps as akin to pixel-wise segmentation and accordingly train a denoising model to refine the contact maps starting from a noise-infused state progressively. We also devise a potent conditioning mechanism that harnesses features extracted from RNA sequences to steer the model toward generating an accurate secondary structure. These features encompass one-hot encoded sequences, probabilistic maps generated from a pre-trained scoring network, and embeddings and attention maps derived from RNA-FM. Experimental results on both within- and cross-family datasets demonstrate RNADiffFold's competitive performance compared with current state-of-the-art methods. Additionally, RNADiffFold has shown a notable proficiency in capturing the dynamic aspects of RNA structures, a claim corroborated by its performance on datasets comprising multiple conformations.

## Prerequisites

* python >= 3.8
* torch >= 2.0.1 with cudnn >= 11.8

:star: **Note**:
1. Before using the `requirements.yml` file, please update the prefix path in the last line to match your own system's path.
2. Use the following command to create the environment. 
```
conda env create -f requiremnets.yml
```
3. Activate the environment.
```
conda activate RNADiffFold
```

## Pre-trained Models and using data
Pre-trained models are available in the [checkpoint](https://drive.google.com/drive/folders/1jt6G-O15I0Sn6kbplLZhftbv8DLZeyR_?usp=sharing). The training and evaluation data are stored in the [data](https://drive.google.com/drive/folders/1NwClE1Df56LfZ2wGkvr2LnPpGnHH6CGU?usp=sharing), with all data preprocessed for computational efficiency. 

## Usage

### Training
We provide the data used for training and evaluating RNADiffFold. Please download the data and place it in the `./data` directory. If you wish to train the model with your own data, please preprocess it using the scripts available in the `./preprocess_data` directory.

We utilize the pretrained weights of the [Ufold](https://github.com/uci-cbcl/UFold) and [RNA-FM](https://github.com/ml4bio/RNA-FM) to condition the model. If you wish to train the RNADiffFold model from scratch, please download the conditioner pretrained weights from [checkpoint](https://drive.google.com/drive/folders/1jt6G-O15I0Sn6kbplLZhftbv8DLZeyR_?usp=sharing) and place them in the `./ckpt/cond_ckpt`.

Then, run the following command to train the model:

```bash
python train.py --device cuda:0
                --diffusion_dim 8
                --diffusion_steps 20
                --cond_dim 8
                --dataset all
                --batch_size 1
                --dp_rate 0.1
                --lr 0.0001
                --warmup 5
                --seed 2023
                --log_wandb True
                --epochs 400
                --eval_every 20
                -u_conditioner_ckpt ufold_train_alldata.pt
```
### Evaluating 
We provide the test script for user to evaluate the prediction result using the following command:

```bash
python evaluation/eval.py
```

The predict results for each sequence will be stored in the `./evaluation/results` directory.

### Predicting
We provide the predict script for user to predict the secondary structure of the RNA sequence. Users can put the RNA sequence data in `./prediction/predict_data` in fasta format. Then, run the following command to predict the secondary structure:

```bash
python prediction/predict.py
```
The predict results for each sequence will be stored in the `./prediction/'predict_results/ct_files'` directory.s

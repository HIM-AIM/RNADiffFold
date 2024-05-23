# RNADiffFold: Generative RNA Secondary Structure Prediction using Discrete Diffusion Models

## 1.Please use the .yml file to create your environment
'''
conda env create -f requiremnets.yml
'''

## 2. During training, the hyperparameters are set as follows:

'''
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
'''
## 3. The model checkpoints and the training and testing data can be downloaded in: [https://drive.google.com/drive/folders/1jt6G-O15I0Sn6kbplLZhftbv8DLZeyR_?usp=drive_link]

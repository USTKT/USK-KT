# UST-KT
<img src="https://github.com/USTKT/USK-KT/blob/main/Picture/model.jpg?raw=true" width="700">
## Installation
Create conda envirment.

```
conda create --name=ustkt python=3.7.5
conda activate ustkt
```

```
pip install -U pykt-toolkit -i  https://pypi.python.org/simple 

```

# Dataset
we use datasets including :

Bridge2006 (https://pslcdatashop.web.cmu.edu/KDDCup/)

Algebra2005 (https://pslcdatashop.web.cmu.edu/KDDCup/)

NIPS34 (https://eedi.com/projects/neurips-education-challenge)



## Data Preparation

```
cd train
python data_preprocess.py --dataset_name=bridge2algebra2006
```



## Run Your Model

We provide the Hyper Parameter we use for training, you could run USTKT as follows command 

```
CUDA_VISIBLE_DEVICES=0 python wandb_ustkt_train.py --fold=0 --emb_type=qid --loss3=0.5 --d_ff=64   --nheads=4 --dropout=0.1 --loss2=0.5 --final_fc_dim2=256 --loss1=0.5 --d_model=256 --num_attn_heads=4 --num_layers=2 --seed=42    --final_fc_dim=512 --n_blocks=4 --start=50 --learning_rate=0.0001  --dataset_name=bridge2algebra2006 --emb_type='stoc_qid' --atten_type='w2_hawkes'  --use_decoupling=1 --model_name=ustkt --gamma2=1

```

## Run Baseline Model
You can also use the follows command to run baseline methods.

Such as CUDA_VISIBLE_DEVICES=2 python wandb_akt_train.py --use_wandb=0 --add_uuid=0 --fold=0 --emb_type=qid --d_ff=64   --dropout=0.1   --d_model=256 --num_attn_heads=4  --seed=42   --n_blocks=4  --learning_rate=0.0001  --dataset_name=bridge2algebra2006 




## Evaluate Your Model

Now, let’s use `wandb_predict.py` to evaluate the model performance on the testing set.

```
python wandb_predict.py --save_dir=saved_model/YourModelPath
```

--save_dir is the save path of your trained model that you can find in your training log



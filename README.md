# stan_dynamic

## Preliminary
```
conda create -n dynamics python=3.8
conda activate dynamics
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
pip install torchdiffeq
conda install scikit-image pandas tqdm tensorboard
```


## test for pediatric

pediatric_mvit_v2_s_16_2_pretrained_80
python main.py --num_epochs 80 --source_data dynamic --model_name mvit_v2_s  --frames 16

dynamic_mvit_v2_s_16_2_pretrained_80
python main.py --num_epochs 80 --source_data pediatric --model_name mvit_v2_s --frames 16

convlstm
python main.py --num_epochs 100 --source_data pediatric --model_name convlstm --lr 1e-3 --lr_step_period 30 --frames 32 --batch_size 1


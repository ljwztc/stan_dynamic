# stan_dynamic

## test for pediatric

pediatric_mvit_v2_s_16_2_pretrained_80
python main.py --num_epochs 80 --source_data dynamic --model_name mvit_v2_s  --frames 16

dynamic_mvit_v2_s_16_2_pretrained_80
python main.py --num_epochs 80 --source_data pediatric --model_name mvit_v2_s --frames 16

convlstm
python main.py --num_epochs 100 --source_data pediatric --model_name convlstm --lr 1e-3 --lr_step_period 30 --frames 32 --batch_size 1


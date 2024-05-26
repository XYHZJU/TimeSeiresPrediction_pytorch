export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 0 \
  --root_path ./dataset/datagen/ \
  --data_path BTP_May.csv \
  --model_id May_EnvDLinear \
  --model EnvDLinear \
  --data custom \
  --features MS \
  --seq_len 30 \
  --label_len 0 \
  --features MS \
  --validate_step 10 \
  --patience 7 \
  --learning_rate 0.0025 \
  --pred_len 10 \
  --enc_in 37 \
  --dec_in 37 \
  --c_out 37 \
  --train_epochs 20 \
  --des 'Exp' \
  --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_2 \
#   --model DLinear \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 0 \
#   --pred_len 2 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1 \


# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_3 \
#   --model DLinear \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 0 \
#   --pred_len 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1 \


# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_4 \
#   --model DLinear \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 0 \
#   --pred_len 4 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_5 \
#   --model DLinear \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 0 \
#   --pred_len 5 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1
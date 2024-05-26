export CUDA_VISIBLE_DEVICES=0


# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_1 \
#   --model Linear \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 0 \
#   --pred_len 5 \
#   --validate_step 5 \
#   --learning_rate 0.001 \
#   --train_epochs 16
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path BTP.csv \
  --model_id BTP_100_1 \
  --model Linear \
  --data custom \
  --features MS \
  --seq_len 30 \
  --label_len 0 \
  --pred_len 10 \
  --enc_in 37 \
  --validate_step 10 \
  --learning_rate 0.001 \
  --dec_in 37 \
  --c_out 37 \
  --des 'Exp' \
  --itr 1 \


# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_3 \
#   --model Linear \
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
#   --model Linear \
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
#   --model Linear \
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
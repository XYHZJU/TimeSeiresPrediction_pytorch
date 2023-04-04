export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path BTP0.csv \
  --model_id BTP_100_1 \
  --model Decomp_BiLSTM \
  --data custom \
  --features MS \
  --seq_len 10 \
  --label_len 0 \
  --pred_len 10 \
  --validate_step 10 \
  --rnn_layers 3 \
  --patience 7 \
  --factor 3 \
  --train_epochs 16 \
  --decomp_num 3 \
  --patience 7 \
  --moving_avg 25 \
  --d_model 74 \
  --enc_in 37 \
  --dec_in 37 \
  --c_out 37 \
  --des 'Exp' \
  --itr 1
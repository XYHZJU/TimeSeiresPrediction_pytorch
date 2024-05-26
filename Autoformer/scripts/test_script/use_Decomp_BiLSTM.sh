export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path BTP.csv \
  --model_id EMD_BiLSTM_test \
  --model Decomp_BiLSTM \
  --data custom \
  --features S \
  --batch_size 64 \
  --seq_len 10 \
  --label_len 0 \
  --pred_len 10 \
  --validate_step 10 \
  --rnn_layers 1 \
  --patience 7 \
  --factor 3 \
  --train_epochs 16 \
  --decomp_num 3 \
  --patience 7 \
  --moving_avg 25 \
  --d_model 74 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1
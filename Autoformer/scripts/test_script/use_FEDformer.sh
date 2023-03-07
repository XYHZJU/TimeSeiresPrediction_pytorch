export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path BTP.csv \
  --model_id BTP_100_1 \
  --model FEDformer \
  --data custom \
  --features MS \
  --seq_len 10 \
  --label_len 10 \
  --pred_len 10 \
  --modes 32 \
  --version Wavelets \
  --L 1 \
  --validate_step 10 \
  --learning_rate 0.0005 \
  --weight_decay 1 \
  --d_ff 256 \
  --d_model 64 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --freq t \
  --moving_avg 25 \
  --patience 7 \
  --enc_in 37 \
  --dec_in 37 \
  --c_out 37 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 16


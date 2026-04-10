CUDA_VISIBLE_DEVICES=4 python main.py \
--exp_name best_model \
--model_name ThumosModel \
--num_epochs 500 \
--detection_inf_step 30 \
--soft_nms \
--data_path /data/lizl/THUMOS14
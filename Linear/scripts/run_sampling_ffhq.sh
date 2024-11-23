# super_resolution_config.yaml inpainting_random_config.yaml inpainting_box_config.yaml gaussian_deblur_config.yaml motion_deblur_config.yaml
python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --gpu=0 \
    --save_dir=results_ffhq;

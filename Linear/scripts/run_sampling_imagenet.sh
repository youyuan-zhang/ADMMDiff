# super_resolution_config.yaml inpainting_random_config.yaml inpainting_box_config.yaml gaussian_deblur_config.yaml motion_deblur_config.yaml
python3 sample_condition.py \
    --model_config=configs/imagenet_model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/inpainting_box_config.yaml \
    --gpu=0 \
    --save_dir=results_compare_imagenet;

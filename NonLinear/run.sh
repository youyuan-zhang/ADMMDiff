# segmentation map guidance
python main.py -s parse_ddim_admm --doc celeba_hq --timesteps 200 --seed 255 --rho_scale 0.2 --ref_path ./images/00255.jpg --batch_size 1
# sketch guidance
python main.py -s sketch_ddim_admm --doc celeba_hq --timesteps 200 --seed 1234 --rho_scale 20.0 --ref_path ./images/00500.jpg --batch_size 1
# text guidance
python main.py -s clip_ddim_admm --doc celeba_hq --timesteps 200 --seed 1234 --prompt "young girl with short hair" --batch_size 1
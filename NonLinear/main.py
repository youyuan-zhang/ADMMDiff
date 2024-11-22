import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
# import torch.utils.tensorboard as tb

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--output", type=str, default="output", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "--ni",
        type=bool,
        default=True,
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100, help="number of steps involved"
    )
    parser.add_argument(
        "--model_type", type=str, default="face", help=" face | imagenet "
    )
    parser.add_argument(
        "--batch_size", type=int, default=10
    )
    parser.add_argument(
        "--class_num", type=int, default=10
    )




    parser.add_argument(
        "-s",
        "--sample_strategy",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--mu", type=float, default=1.0, help=""
    )

    parser.add_argument(
        "--rho_scale", type=float, default=0.1
    )
    parser.add_argument(
        "--prompt", type=str, default="black"
    )
    parser.add_argument(
        "--stop", type=int, default=100
    )
    parser.add_argument(
        "--ref_path", type=str, default=None
    )
    parser.add_argument(
        "--ref_path2", type=str, default=None
    )
    parser.add_argument(
        "--scale_weight", type=float, default=None
    )
    parser.add_argument(
        "--rt", type=int, default=1
    )
    

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    args.image_folder = args.output
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args


def main():
    args = parse_args_and_config()

    runner = Diffusion(args)
    runner.sample(args.sample_strategy)

    return 0


if __name__ == "__main__":
    sys.exit(main())

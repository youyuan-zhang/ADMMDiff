
# Decoupling Training-Free Guided Diffusion by ADMM
  

## Abstract
In this paper, we consider the conditional generation problem by guiding off-the-shelf unconditional diffusion models with differentiable loss functions in a plug-and-play fashion. We propose a novel framework that distinctly decouples these two components and develop a new algorithm based on the Alternating Direction Method of Multipliers (ADMM) to adaptively balance these components. 

![cover-img](./_assets_/teanser.png)

## Set environment
```
conda env create -f environment.yml
conda activate admm
```
## Run
Please refer to the folders ./NonLinear and ./Linear.
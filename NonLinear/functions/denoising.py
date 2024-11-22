import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os

from .clip.base_clip import CLIPEncoder
from .face_parsing.model import FaceParseTool
from .anime2sketch.model import FaceSketchTool 
from .landmark.model import FaceLandMarkTool
from .arcface.model import IDLoss

import numpy as np
import cv2

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

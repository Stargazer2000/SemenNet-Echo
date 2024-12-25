import torch
import numpy as np
import timm
import glob


for checkpoint in glob.glob("D:/Tea3/BiSeNet_3/BiSeNet/FinalCheckpoints/*.pth"):
    model=checkpoint
    print(checkpoint)


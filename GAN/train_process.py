"""
Import necessary libraries to create a generative adversarial network
The code is mainly developed using the PyTorch library
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from gan_models import discriminator, generator
import numpy as np
import matplotlib.pyplot as plt

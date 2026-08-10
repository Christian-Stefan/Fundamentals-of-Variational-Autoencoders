import ctypes as ct
import random as rdm
import torch.nn as nn
import torch
import pandas as pd
import matplotlib.pyplot as PLT
from torchinfo import summary
from utils.Code_for_students.MNIST_dataloader import create_dataloaders
from utils.Code_for_students.main_template import x_noisy_example, x_clean_example
import ast
import torch.nn.functional as F
import argparse
from torch.optim import (
    SGD,RAdam, Adam
)
from torch.nn import (
    BCELoss, MSELoss
)
from torchvision.transforms import v2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tqdm
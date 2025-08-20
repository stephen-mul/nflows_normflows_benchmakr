import torch
from lightning import LightningModule
from torch import nn
from torch import optim

import normflows as nf

class SimpleFlow(LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers):
        

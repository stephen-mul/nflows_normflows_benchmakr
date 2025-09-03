import torch
import matplotlib.pyplot as plt

from lightning import LightningModule
from torch import optim, linspace, meshgrid

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

 
class SimpleFlow(LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.save_hyperparameters()
        
        # Define the base distribution
        self.base_distribution = StandardNormal(shape=[input_dim])
        
        # Define the transforms
        transforms = []
        for _ in range(num_layers):
            transforms.append(MaskedAffineAutoregressiveTransform(
                features=input_dim,
                hidden_features=hidden_dim,
                num_blocks=2))
            transforms.append(ReversePermutation(features=input_dim))
        
        self.transform = CompositeTransform(transforms)
        
        # Define the flow
        self.flow = Flow(
            self.transform,
            self.base_distribution,
        )

    def forward(self, x):
        return self.flow.log_prob(x)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        log_prob = self.forward(x)
        loss = -log_prob.mean()
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def inference(self, outtput_dir=None):
        xline = linspace(-1.5, 2.5, 100)
        yline = linspace(-0.75, 1.25, 100)
        xgrid, ygrid = meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
        xyinput = xyinput.to(self.device)
        with torch.no_grad():
            zgrid = self.flow.log_prob(xyinput).exp().reshape(100, 100)
        
        plt.contourf(xgrid.numpy(),
                     ygrid.numpy(),
                     zgrid.numpy(),
                     )
        plt.title('Learned Distribution')
        plt.colorbar()
        if outtput_dir:
            plt.savefig(f"{outtput_dir}/nflow_learned_distribution.png")
        plt.close()




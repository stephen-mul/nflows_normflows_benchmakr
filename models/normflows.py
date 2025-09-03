import torch
import matplotlib.pyplot as plt
import normflows as nf
from lightning import LightningModule
from torch import optim, linspace, meshgrid
from normflows import flows

from models.transforms import ReversePermutation

class SimpleFlow(LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.save_hyperparameters()

        # Define base distribution
        self.base_distribution = nf.distributions.base.DiagGaussian(2)

        # Define the transforms
        transforms = []
        for _ in range(num_layers):
            transforms.append(
                flows.affine.autoregressive.MaskedAffineAutoregressive(
                    features=input_dim,
                    hidden_features=hidden_dim,
                    num_blocks=2,
                )
            )
            transforms.append(
                ReversePermutation(features=input_dim)
                )
            
        self.flow = nf.NormalizingFlow(
            self.base_distribution,
            transforms,
        )

    def forward(self, x):
        return self.flow.forward_kld(x)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.forward(x)
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
        plt.show()
        if outtput_dir:
            plt.savefig(f"{outtput_dir}/normflow_learned_distribution.png")
        plt.close()

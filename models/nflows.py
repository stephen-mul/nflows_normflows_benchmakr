from lightning import LightningModule
from torch import optim

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



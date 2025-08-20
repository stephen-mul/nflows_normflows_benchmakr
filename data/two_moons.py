import numpy as np
import os
import matplotlib.pyplot as plt
from lightning import LightningDataModule
from lightning.fabric.utilities.rank_zero import rank_zero_only
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_moons

class TwoMoonsDataset(Dataset):
    def __init__(self, n_samples=1000, noise=0.1):
        self.n_samples = n_samples
        self.noise = noise
        self.data, self.labels = make_moons(n_samples=self.n_samples, noise=self.noise)
        self.data = np.float32(self.data)
        self.labels = np.int64(self.labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

class TwoMoonsDataModule(LightningDataModule):
    def __init__(self, 
                 n_samples=1000, 
                 noise=0.1, 
                 batch_size=32,
                 plotting_dir=None,
                 ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.noise = noise
        self.batch_size = batch_size
        self.plotting_dir = plotting_dir
        self._plotted = False

    def setup(self, stage=None):
        # Create dataset if needed
        if not hasattr(self, 'dataset'):
            self.dataset = TwoMoonsDataset(
                n_samples=self.n_samples, 
                noise=self.noise,
                )
        if self.plotting_dir:
            os.makedirs(self.plotting_dir, exist_ok=True)
            self._plot_data()
            self._plotted = True

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
    @rank_zero_only
    def _plot_data(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.dataset.data[:, 0], 
            self.dataset.data[:, 1], 
            c=self.dataset.labels, 
            cmap='viridis', 
            s=10
            )
        plt.title('Two Moons Dataset')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(os.path.join(self.plotting_dir, 'initial_two_moons_plot.png'))
        plt.close()
    




from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_moons
import numpy as np

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
    def __init__(self, n_samples=1000, noise=0.1, batch_size=32):
        super().__init__()
        self.n_samples = n_samples
        self.noise = noise
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = TwoMoonsDataset(n_samples=self.n_samples, noise=self.noise)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False) 
    




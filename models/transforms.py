# Custom implementation of ReversePermutation flow for normflows
# Should replicate behaviour of nflows ReversePermutation

import torch
from normflows.flows import Flow
    

class Permutation(Flow):
    """
    Applies a fixed permutation of features along a given dimension.
    """

    def __init__(self, permutation: torch.Tensor, dim: int = 1):
        super().__init__()

        if permutation.ndim != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("dim must be a non-negative integer.")

        self.dim = dim
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))

    def forward(self, z, context=None):
        z = z.index_select(self.dim, self.permutation.to(z.device))
        log_det = torch.zeros(z.size(0), device=z.device)
        return z, log_det

    def inverse(self, z, context=None):
        z = z.index_select(self.dim, self.inverse_permutation.to(z.device))
        log_det = torch.zeros(z.size(0), device=z.device)
        return z, log_det
    
class ReversePermutation(Permutation):
    """
    Reverses the order of features along a given dimension.
    """

    def __init__(self, features: int, dim: int = 1):
        if not isinstance(features, int) or features <= 0:
            raise ValueError("Number of features must be a positive integer.")
        permutation = torch.arange(features - 1, -1, -1)
        super().__init__(permutation, dim)
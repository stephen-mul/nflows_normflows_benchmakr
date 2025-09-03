import torch

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print("device:", torch.cuda.get_device_name(i))
    print("capability:", torch.cuda.get_device_capability(i))  # e.g., (7, 5) for T4

import torch


def softmax(tensor: torch.tensor) -> torch.tensor:
    return torch.ops.cudabox.softmax(tensor)

import torch 
from torch import nn

class KLLoss(nn.Module):
    """
    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """
    def __init__(
        self,
        size: int,
        padding_idx: int
        normalize_length: bool = False):

        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="none", log_target = True)
        self.padding_idx = padding_idx
        self.size = size
        self.normalize_length = normalize_length


class AEDKLLoss(KLLoss):
    def forward(
        self,
        x: torch. Tensor,
        target_dist: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

    assert x.size(2) == self.size
    batch_size = x.size(0)

    x = x.view(-1, self.size)
    target_dist = target_dist.view(-1, self.size)

    target = target.view(-1)
    ignore = target == self.padding_idx
    total = len(target) - ignore.sum().item

    kl = self.criterion(torch.log_softmax(x, dim=1), target_dist)
    denom = total if self.normalize_length else batch_size
    return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


class CTCKLLoss(KLLoss):
    def forward(
        self,
        x: torch.Tensor,
        target_dist: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:

    assert x.size(2) = self.size
    batch_size = x.size(0)
    seq_len = x.size(1)
    
    x = x.view(-1, self.size)
    target_dist = target_dist.view(-1, self.size)

    mask = mask.view(-1)
    total = batch_size * seq_len - mask.sum().item()

    kl = self.criterion(torch.log_softmax(x, dim=1), target_dist)
    denom = total if self.normalize_length else batch_size
    return kl.masked_fill(mask.unsqueeze(1), 0).sum() / denom
    




import torch
import torch.nn as nn
from info_nce import InfoNCE


class MSBertLoss(nn.Module):
    def __init__(self, temperature: float, ignore_index: int, reduction: str = "mean") -> None:
        super(MSBertLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction
        )
        self.info_nce = InfoNCE(temperature, reduction)

    def forward(
        self, logits_lm1: torch.Tensor, mask_token1: torch.Tensor,
        logits_lm2: torch.Tensor, mask_token2: torch.Tensor,
        pool1: torch.Tensor, pool2: torch.Tensor
    ):
        l1 = self.cross_entropy(
            logits_lm1.view(-1, logits_lm1.shape[-1]),
            mask_token1.view(-1)
        )
        l2 = self.cross_entropy(
            logits_lm2.view(-1, logits_lm2.shape[-1]),
            mask_token2.view(-1)
        )
        l3 = self.info_nce(pool1.squeeze(), pool2.squeeze())
        return l1 + l2 + l3

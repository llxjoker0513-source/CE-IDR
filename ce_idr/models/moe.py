import torch.nn as nn


class MoEFusion(nn.Module):
    """Interface for the intentionally withheld expert fusion implementation."""

    def __init__(self, ch, num_experts, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "The expert fusion implementation is not included in the CE-IDR "
            "partial release. A private implementation and configuration are required."
        )

    def forward(self, fsv, fsi, fsemv, fsemi):
        """Expected private return value: fused_feature, auxiliary_loss."""
        raise NotImplementedError("Expert fusion is withheld.")

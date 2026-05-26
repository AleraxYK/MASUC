import torch
import torch.nn as nn


def feature_extractor(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Extracts features from the model up to the final fully connected layer.
    Assumes standard ResNet-like architecture where the final layer is named 'fc'.

    NOTE: kept for backward compatibility. Prefer ``forward_with_features``
    when both logits and features are needed in the same forward.
    """
    features = inputs
    for name, module in model.named_children():
        if name == 'fc':
            features = torch.flatten(features, 1)
            break
        features = module(features)
    return features


def classifier_extractor(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """
    Passes extracted features through the final classifier layer of the model.
    Assumes the final layer is named 'fc'.
    """
    return model.fc(features)


def forward_with_features(model: nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single forward pass through a ResNet-style model that returns BOTH the
    final logits and the pre-classifier (flattened, pooled) features.

    Halves the cost of the MASUC inner loop versus calling the model twice
    (once for logits, once for features).
    """
    features = inputs
    last_pre_fc = None
    for name, module in model.named_children():
        if name == 'fc':
            last_pre_fc = torch.flatten(features, 1)
            features = module(last_pre_fc)
        else:
            features = module(features)
    if last_pre_fc is None:
        last_pre_fc = torch.flatten(features, 1)
    return features, last_pre_fc


class EnergyRunningMean:
    """
    O(1) running mean of energies, replacing the O(N) torch.cat accumulator
    that was leaking memory and triggering CPU-GPU sync every batch.

    Math is identical to the original cumulative-mean implementation
    (sum / count over all energies seen so far in the current epoch),
    just computed incrementally.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        self._sum   = torch.zeros((), device=device, dtype=dtype)
        self._count = torch.zeros((), device=device, dtype=dtype)

    def update(self, energy: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self._sum   = self._sum   + energy.detach().sum()
            self._count = self._count + energy.numel()
        return self._sum / self._count.clamp(min=1.0)

    def reset(self) -> None:
        self._sum.zero_()
        self._count.zero_()

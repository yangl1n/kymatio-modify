import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kymatio'))
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D


def _count_scattering_coefficients(J, L, max_order):
    """Compute the number of scattering output channels."""
    count = 1  # order 0
    count += J * L  # order 1
    if max_order >= 2:
        count += L * L * J * (J - 1) // 2  # order 2
    return count


# ---------------------------------------------------------------------------
# Base: scattering feature extractor
# ---------------------------------------------------------------------------

class ScatteringBase(nn.Module):
    """Base class that wraps the scattering transform as a feature extractor.

    Subclasses only need to implement ``_build_classifier`` and, if the
    default flatten-then-classify flow is insufficient, override ``forward``.
    """

    def __init__(self, J, shape, in_channels, num_classes,
                 L=8, max_order=2, learnable=False):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.J = J
        self.L = L
        self.max_order = max_order
        self.shape = shape

        self.scattering = ScatteringTorch2D(
            J=J, shape=shape, L=L, max_order=max_order, learnable=learnable,
        )

        num_coeffs = _count_scattering_coefficients(J, L, max_order)
        spatial_h = shape[0] // (2 ** J)
        spatial_w = shape[1] // (2 ** J)
        self.feature_dim = in_channels * num_coeffs * spatial_h * spatial_w

        self.classifier_input_dim = self.feature_dim
        self.classifier = self._build_classifier()

    # -- interface for subclasses ------------------------------------------

    def _build_classifier(self):
        """Return an ``nn.Module`` that maps (B, feature_dim) -> (B, num_classes)."""
        raise NotImplementedError

    # -- forward -----------------------------------------------------------

    def extract_features(self, x):
        """Run scattering and flatten."""
        B = x.shape[0]
        s = self.scattering(x)
        return s.reshape(B, -1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, H, W)

        Returns
        -------
        logits : torch.Tensor of shape (B, num_classes)
        """
        features = self.extract_features(x)
        return self.classifier(features)

    # -- freeze / unfreeze helpers -----------------------------------------

    def head_parameters(self):
        """Yield classifier-side parameters (classifier)."""
        yield from self.classifier.parameters()

    def freeze_scattering(self):
        for p in self.scattering.parameters():
            p.requires_grad = False

    def unfreeze_scattering(self):
        for p in self.scattering.parameters():
            p.requires_grad = True

    def freeze_classifier(self):
        for p in self.head_parameters():
            p.requires_grad = False

    def unfreeze_classifier(self):
        for p in self.head_parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Example classifiers
# ---------------------------------------------------------------------------

class ExampleLinearClassifier(ScatteringBase):
    """Scattering features + single Linear layer."""

    def _build_classifier(self):
        d = self.classifier_input_dim
        layers = []
        layers.append(nn.Linear(d, self.num_classes))
        return nn.Sequential(*layers)


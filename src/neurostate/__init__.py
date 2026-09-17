"""NeuroState: gradient-isolated boundary learning for interpretable EEG
transformers.

Reference implementation for the IEEE MLSP 2026 paper by Yogarajan Sivakumar
and Hong Man.
"""

from .isolation import PhaseSchedule, isolate
from .losses import ACBLLoss, AttentionPriorLoss, PseudoBoundaryLoss
from .model import MultiResContrastiveNeuroState, forward_with_intermediates

__version__ = "1.0.0"

__all__ = [
    "MultiResContrastiveNeuroState",
    "forward_with_intermediates",
    "ACBLLoss",
    "PseudoBoundaryLoss",
    "AttentionPriorLoss",
    "PhaseSchedule",
    "isolate",
    "__version__",
]

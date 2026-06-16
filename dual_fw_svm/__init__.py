"""Fast reproductions of the SVM optimizers from the OPT2025 paper."""

from .binary import BinaryL2DualSVM
from .multiclass import BlockCoordinateFrankWolfeSVM, MulticlassFrankWolfeSVM

__all__ = [
    "BinaryL2DualSVM",
    "MulticlassFrankWolfeSVM",
    "BlockCoordinateFrankWolfeSVM",
]

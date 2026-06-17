"""Fast dual proximal and Frank-Wolfe SVM optimizers."""

from .binary import BinaryL2DualSVM
from .multiclass import BlockCoordinateFrankWolfeSVM, MulticlassFrankWolfeSVM

__all__ = [
    "BinaryL2DualSVM",
    "MulticlassFrankWolfeSVM",
    "BlockCoordinateFrankWolfeSVM",
]

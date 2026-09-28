"""Pure-PyTorch Mamba-3 SISO, with batched training and recurrent decoding."""

from .modules import Mamba3, Mamba3Block, Mamba3Stack
from .norm import RMSNormGated
from .ops import mamba3_siso_chunked, mamba3_siso_step

__all__ = [
    "Mamba3", "Mamba3Block", "Mamba3Stack", "RMSNormGated",
    "mamba3_siso_chunked", "mamba3_siso_step",
]

from typing import Any


class _TensorType:
    """Lightweight TensorType shim for gfn runtime annotations.

    torchtyping 0.1.4 subclasses torch internals that are incompatible with
    torch 2.4. The installed gfn package only uses TensorType in annotations,
    so returning Any preserves runtime behavior without changing torch/CUDA.
    """

    def __getitem__(self, item):
        return Any


TensorType = _TensorType()

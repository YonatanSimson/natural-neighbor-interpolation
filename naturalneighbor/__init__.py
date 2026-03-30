from .naturalneighbor import griddata

try:
    import ccudanaturalneighbor  # noqa: F401

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

__all__ = [
    "griddata",
    "CUDA_AVAILABLE",
]

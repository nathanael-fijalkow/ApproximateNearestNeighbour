from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
except Exception as e:  # pragma: no cover - makes errors clearer if torch missing
    raise RuntimeError("This package requires PyTorch. Please install torch first: pip install torch") from e


@dataclass
class SearchResult:
    indices: torch.Tensor  # shape (n_query, k) long
    scores: torch.Tensor   # shape (n_query, k) float


class Index(ABC):
    """
    Minimal index interface for cosine-similarity ANN.

    Conventions:
    - Vectors are 2D torch.float32 tensors of shape (n, d).
    - All vectors are L2-normalized internally for cosine similarity.
    - search() returns top-k by cosine similarity (higher is better).
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._xb: Optional[torch.Tensor] = None  # stored, normalized base vectors

    def add(self, xb: torch.Tensor) -> None:
        xb = _ensure_2d_float(xb)
        assert xb.size(1) == self.dim, f"expected dim={self.dim}, got {xb.size(1)}"
        xb = _normalize(xb)
        self._add_impl(xb)

    def build(self) -> None:
        self._build_impl()

    def search(self, xq: torch.Tensor, k: int) -> SearchResult:
        xq = _ensure_2d_float(xq)
        assert xq.size(1) == self.dim
        xq = _normalize(xq)
        return self._search_impl(xq, k)

    @abstractmethod
    def _add_impl(self, xb: torch.Tensor) -> None:
        ...

    def _build_impl(self) -> None:
        # optional to override
        return None

    @abstractmethod
    def _search_impl(self, xq: torch.Tensor, k: int) -> SearchResult:
        ...


# ---------- helpers ----------

def _ensure_2d_float(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dtype != torch.float32:
        x = x.float()
    return x.contiguous()


def _normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    norms = torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(eps)
    return x / norms

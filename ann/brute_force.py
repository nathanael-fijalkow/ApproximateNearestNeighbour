from __future__ import annotations
import torch
from .base import Index, SearchResult
from .utils import get_device


class BruteForceIndex(Index):
    """Exact cosine search via dense matrix multiplication."""

    def __init__(self, dim: int, device: torch.device | str | None = None):
        super().__init__(dim)
        self.device = get_device(device)
        self._xb = torch.empty(0, dim, dtype=torch.float32, device=self.device)

    def _add_impl(self, xb: torch.Tensor) -> None:
        xb = xb.to(self.device)
        if self._xb.numel() == 0:
            self._xb = xb
        else:
            self._xb = torch.cat([self._xb, xb], dim=0)

    def _search_impl(self, xq: torch.Tensor, k: int) -> SearchResult:
        assert self._xb.numel() > 0, "index is empty; call add() first"
        xq = xq.to(self._xb.device)
        sims = xq @ self._xb.T  # cosine since both are normalized
        k = min(k, self._xb.size(0))
        scores, idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
        return SearchResult(indices=idx.cpu(), scores=scores.cpu())

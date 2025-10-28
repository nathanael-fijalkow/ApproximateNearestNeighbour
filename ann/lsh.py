from __future__ import annotations
import math
from typing import Dict, List, Set, Tuple
import torch
from .base import Index, SearchResult
from .utils import get_device


def _pack_bits(signs: torch.Tensor) -> int:
    # signs: (n_bits,) bool
    out = 0
    for i, b in enumerate(signs.tolist()):
        if b:
            out |= (1 << i)
    return out


class LSHIndex(Index):
    """
    Random hyperplane LSH for cosine similarity.

    - n_tables: number of independent hash tables
    - n_bits: number of hyperplanes (bits) per table
    - max_candidates: cap on unique candidates gathered
    """

    def __init__(self, dim: int, n_tables: int = 8, n_bits: int = 16, max_candidates: int = 2048, device: torch.device | str | None = None):
        super().__init__(dim)
        self.device = get_device(device)
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.max_candidates = max_candidates
        self._xb = torch.empty(0, dim, dtype=torch.float32, device=self.device)
        self._tables: List[Dict[int, List[int]] ] = []
        self._planes: torch.Tensor | None = None  # (n_tables, n_bits, dim)

    def _add_impl(self, xb: torch.Tensor) -> None:
        xb = xb.to(self.device)
        start = self._xb.size(0)
        if start == 0:
            self._xb = xb
        else:
            self._xb = torch.cat([self._xb, xb], dim=0)
        # If planes already exist, add to tables incrementally; else will be built in build()
        if self._planes is not None:
            self._index_range(range(start, start + xb.size(0)))

    def _build_impl(self) -> None:
        if self._xb.size(0) == 0:
            return
        g = torch.Generator(device=self.device)
        g.manual_seed(42)
        self._planes = torch.randn(self.n_tables, self.n_bits, self.dim, generator=g, device=self.device)
        self._planes = torch.nn.functional.normalize(self._planes, dim=-1)
        self._tables = [dict() for _ in range(self.n_tables)]
        self._index_range(range(0, self._xb.size(0)))

    def _index_range(self, idx_range: range) -> None:
        assert self._planes is not None
        xb = self._xb
        for i in idx_range:
            v = xb[i]
            # compute signatures for each table
            for t in range(self.n_tables):
                planes = self._planes[t]  # (n_bits, dim)
                signs = (v @ planes.T) >= 0  # (n_bits,)
                key = _pack_bits(signs)
                bucket = self._tables[t].setdefault(key, [])
                bucket.append(i)

    def _probe(self, xq: torch.Tensor) -> List[List[int]]:
        assert self._planes is not None
        results: List[List[int]] = []
        for q in xq:
            cand: Set[int] = set()
            for t in range(self.n_tables):
                planes = self._planes[t]
                signs = (q @ planes.T) >= 0
                key = _pack_bits(signs)
                ids = self._tables[t].get(key)
                if ids:
                    for i in ids:
                        cand.add(i)
                if len(cand) >= self.max_candidates:
                    break
            results.append(list(cand))
        return results

    def _search_impl(self, xq: torch.Tensor, k: int) -> SearchResult:
        assert self._xb.size(0) > 0, "index is empty; call add/build first"
        xq = xq.to(self.device)
        cands = self._probe(xq)
        all_idx = []
        all_scores = []
        for q, cand in zip(xq, cands):
            if len(cand) == 0:
                # fallback to random few to avoid empty
                cand = list(range(min(self._xb.size(0), self.max_candidates)))
            subset = self._xb[cand]
            sims = (subset @ q)
            if sims.dim() == 0:
                sims = sims.unsqueeze(0)
            kq = min(k, sims.numel())
            scores, order = torch.topk(sims, k=kq, dim=0, largest=True, sorted=True)
            idx = torch.tensor(cand, device=scores.device, dtype=torch.long)[order]
            # pad if needed
            if kq < k:
                pad = k - kq
                pad_idx = torch.full((pad,), -1, dtype=torch.long, device=idx.device)
                pad_scores = torch.full((pad,), float("nan"), dtype=scores.dtype, device=scores.device)
                idx = torch.cat([idx, pad_idx])
                scores = torch.cat([scores, pad_scores])
            all_idx.append(idx)
            all_scores.append(scores)
        return SearchResult(indices=torch.stack(all_idx).cpu(), scores=torch.stack(all_scores).cpu())

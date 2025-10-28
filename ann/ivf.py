from __future__ import annotations
from typing import List, Tuple
import torch
from .base import Index, SearchResult
from .utils import get_device


def _kmeans(x: torch.Tensor, n_clusters: int, n_iter: int = 25, seed: int = 42) -> torch.Tensor:
    """Simple KMeans on unit vectors using cosine distance (== maximize dot).
    Returns centroids of shape (n_clusters, d).
    """
    n, d = x.shape
    g = torch.Generator(device=x.device).manual_seed(seed)
    # init with random samples
    perm = torch.randperm(n, generator=g, device=x.device)
    centroids = x[perm[:n_clusters]].clone()
    for _ in range(n_iter):
        # assign to best centroid by dot
        sims = x @ centroids.T  # (n, n_clusters)
        assign = sims.argmax(dim=1)
        # recompute centroids
        for k in range(n_clusters):
            mask = assign == k
            if mask.any():
                c = x[mask].mean(dim=0)
                c = torch.nn.functional.normalize(c, dim=0)
                centroids[k] = c
            else:  # reinitialize empty cluster
                centroids[k] = x[torch.randint(0, n, (1,), generator=g, device=x.device)].squeeze(0)
    return centroids


class IVFIndex(Index):
    """
    Inverted File (IVF-Flat) with cosine similarity.

    - nlist: number of coarse centroids (lists)
    - n_probe: number of lists to probe at search time
    """

    def __init__(self, dim: int, nlist: int = 100, n_probe: int = 5, device: torch.device | str | None = None, kmeans_iter: int = 25):
        super().__init__(dim)
        self.device = get_device(device)
        self.nlist = nlist
        self.n_probe = n_probe
        self.kmeans_iter = kmeans_iter
        self._xb = torch.empty(0, dim, dtype=torch.float32, device=self.device)
        self._lists: List[List[int]] = []  # indices per list concatenated via mapping
        self._list_ids: List[List[int]] = []  # per list: vector ids
        self._centroids: torch.Tensor | None = None

    def _add_impl(self, xb: torch.Tensor) -> None:
        xb = xb.to(self.device)
        if self._xb.numel() == 0:
            self._xb = xb
        else:
            self._xb = torch.cat([self._xb, xb], dim=0)

    def _build_impl(self) -> None:
        if self._xb.size(0) == 0:
            return
        # train centroids
        self._centroids = _kmeans(self._xb, self.nlist, n_iter=self.kmeans_iter)
        # assign
        sims = self._xb @ self._centroids.T
        assign = sims.argmax(dim=1)
        self._list_ids = [[] for _ in range(self.nlist)]
        for i, lid in enumerate(assign.tolist()):
            self._list_ids[lid].append(i)

    def _search_impl(self, xq: torch.Tensor, k: int) -> SearchResult:
        assert self._xb.size(0) > 0 and self._centroids is not None
        xq = xq.to(self.device)
        cent = self._centroids
        csims = xq @ cent.T  # (nq, nlist)
        topc_sim, topc = torch.topk(csims, k=min(self.n_probe, cent.size(0)), dim=1)
        idx_out = []
        score_out = []
        for qi, lists in enumerate(topc.tolist()):
            cand_ids: List[int] = []
            for lid in lists:
                cand_ids.extend(self._list_ids[lid])
            if not cand_ids:
                cand_ids = list(range(self._xb.size(0)))  # fallback
            subset = self._xb[cand_ids]
            sims = xq[qi] @ subset.T
            kq = min(k, sims.numel())
            scores, order = torch.topk(sims, k=kq, dim=0)
            idx = torch.tensor(cand_ids, device=scores.device, dtype=torch.long)[order]
            if kq < k:
                pad = k - kq
                idx = torch.cat([idx, torch.full((pad,), -1, dtype=torch.long, device=idx.device)])
                scores = torch.cat([scores, torch.full((pad,), float("nan"), dtype=scores.dtype, device=scores.device)])
            idx_out.append(idx)
            score_out.append(scores)
        return SearchResult(indices=torch.stack(idx_out).cpu(), scores=torch.stack(score_out).cpu())

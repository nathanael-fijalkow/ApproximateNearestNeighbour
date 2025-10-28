from __future__ import annotations
from typing import List
import torch
from .base import Index, SearchResult
from .utils import get_device


def _kmeans_1d(x: torch.Tensor, k: int, n_iter: int = 25, seed: int = 42) -> torch.Tensor:
    """K-means on a subspace (d_sub,). Returns centroids of shape (k, d_sub)."""
    n, d_sub = x.shape
    g = torch.Generator(device=x.device).manual_seed(seed)
    perm = torch.randperm(n, generator=g, device=x.device)
    centroids = x[perm[:k]].clone()
    
    for _ in range(n_iter):
        # assign
        dists = torch.cdist(x, centroids, p=2)  # (n, k)
        assign = dists.argmin(dim=1)
        # recompute
        for j in range(k):
            mask = assign == j
            if mask.any():
                centroids[j] = x[mask].mean(dim=0)
            else:
                centroids[j] = x[torch.randint(0, n, (1,), generator=g, device=x.device)].squeeze(0)
    return centroids


class PQCodec:
    """
    Product Quantization encoder/decoder.
    
    - m: number of subspaces
    - nbits: bits per subspace -> 2^nbits centroids per subspace
    
    The vector dimension d must be divisible by m.
    """
    def __init__(self, d: int, m: int = 8, nbits: int = 8, kmeans_iter: int = 25):
        assert d % m == 0, f"dim {d} must be divisible by m={m}"
        self.d = d
        self.m = m
        self.nbits = nbits
        self.ksub = 1 << nbits  # 2^nbits
        self.d_sub = d // m
        self.kmeans_iter = kmeans_iter
        self.codebooks: List[torch.Tensor] | None = None  # list of (ksub, d_sub)
    
    def train(self, xb: torch.Tensor) -> None:
        """Train PQ codebooks on base vectors (n, d)."""
        n, d = xb.shape
        assert d == self.d
        device = xb.device
        self.codebooks = []
        
        for i in range(self.m):
            sub = xb[:, i * self.d_sub : (i + 1) * self.d_sub]  # (n, d_sub)
            cb = _kmeans_1d(sub, self.ksub, n_iter=self.kmeans_iter, seed=42 + i)
            self.codebooks.append(cb)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode vectors to PQ codes. x: (n, d) -> codes: (n, m) uint8."""
        assert self.codebooks is not None, "call train() first"
        n, d = x.shape
        assert d == self.d
        codes = torch.empty(n, self.m, dtype=torch.uint8, device=x.device)
        
        for i in range(self.m):
            sub = x[:, i * self.d_sub : (i + 1) * self.d_sub]
            cb = self.codebooks[i]
            dists = torch.cdist(sub, cb, p=2)  # (n, ksub)
            codes[:, i] = dists.argmin(dim=1).to(torch.uint8)
        return codes
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode PQ codes back to approximate vectors. codes: (n, m) -> x: (n, d)."""
        assert self.codebooks is not None
        n, m = codes.shape
        assert m == self.m
        device = codes.device
        x = torch.empty(n, self.d, dtype=torch.float32, device=device)
        
        for i in range(self.m):
            cb = self.codebooks[i].to(device)
            x[:, i * self.d_sub : (i + 1) * self.d_sub] = cb[codes[:, i].long()]
        return x
    
    def compute_asymmetric_table(self, xq: torch.Tensor) -> torch.Tensor:
        """
        Precompute distance table for asymmetric search.
        xq: (nq, d) -> table: (nq, m, ksub) where table[i, j, k] = dist from query i subspace j to centroid k.
        
        For cosine similarity on normalized vectors, we use negative dot product as distance.
        """
        assert self.codebooks is not None
        nq, d = xq.shape
        assert d == self.d
        device = xq.device
        table = torch.empty(nq, self.m, self.ksub, dtype=torch.float32, device=device)
        
        for i in range(self.m):
            q_sub = xq[:, i * self.d_sub : (i + 1) * self.d_sub]  # (nq, d_sub)
            cb = self.codebooks[i].to(device)  # (ksub, d_sub)
            # cosine distance = -dot (since normalized)
            table[:, i, :] = -(q_sub @ cb.T)  # (nq, ksub)
        return table
    
    def asymmetric_distance(self, codes: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """
        Compute distances from queries to encoded vectors using precomputed table.
        codes: (n, m) uint8
        table: (nq, m, ksub) float
        -> dists: (nq, n) float
        """
        nq, m, ksub = table.shape
        n, m2 = codes.shape
        assert m == m2 == self.m
        device = table.device
        codes = codes.to(device).long()
        
        dists = torch.zeros(nq, n, dtype=torch.float32, device=device)
        for i in range(self.m):
            # gather distances for subspace i
            sub_dists = table[:, i, :]  # (nq, ksub)
            code_i = codes[:, i]  # (n,)
            # broadcast: for each query, look up distance to each code
            dists += sub_dists[:, code_i]  # (nq, n)
        return dists


class PQIndex(Index):
    """
    Standalone Product Quantization index with asymmetric distance.
    
    - m: number of subspaces
    - nbits: bits per subspace
    """
    def __init__(self, dim: int, m: int = 8, nbits: int = 8, device: torch.device | str | None = None, kmeans_iter: int = 25):
        super().__init__(dim)
        self.device = get_device(device)
        self.pq = PQCodec(dim, m=m, nbits=nbits, kmeans_iter=kmeans_iter)
        self._xb = torch.empty(0, dim, dtype=torch.float32, device=self.device)
        self._codes: torch.Tensor | None = None
    
    def _add_impl(self, xb: torch.Tensor) -> None:
        xb = xb.to(self.device)
        if self._xb.numel() == 0:
            self._xb = xb
        else:
            self._xb = torch.cat([self._xb, xb], dim=0)
    
    def _build_impl(self) -> None:
        if self._xb.size(0) == 0:
            return
        # train codebooks
        self.pq.train(self._xb)
        # encode
        self._codes = self.pq.encode(self._xb)
    
    def _search_impl(self, xq: torch.Tensor, k: int) -> SearchResult:
        assert self._codes is not None, "call build() first"
        xq = xq.to(self.device)
        # asymmetric distance
        table = self.pq.compute_asymmetric_table(xq)
        dists = self.pq.asymmetric_distance(self._codes, table)
        # convert distance to similarity (negate)
        sims = -dists
        kq = min(k, self._codes.size(0))
        scores, idx = torch.topk(sims, k=kq, dim=1, largest=True, sorted=True)
        
        if kq < k:
            pad = k - kq
            idx = torch.cat([idx, torch.full((idx.size(0), pad), -1, dtype=torch.long, device=idx.device)], dim=1)
            scores = torch.cat([scores, torch.full((scores.size(0), pad), float("nan"), dtype=scores.dtype, device=scores.device)], dim=1)
        
        return SearchResult(indices=idx.cpu(), scores=scores.cpu())

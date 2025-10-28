from __future__ import annotations
import heapq
import random
from typing import List, Tuple, Set
import torch
from .base import Index, SearchResult
from .utils import get_device


class HNSWIndex(Index):
    """
    Hierarchical Navigable Small World (HNSW) graph index.
    
    Multi-layer graph structure where:
    - Layer 0 contains all points
    - Higher layers contain exponentially fewer points
    - Search starts from top layer and zooms in
    
    Parameters:
    - M: max neighbors per node (typically 16-48)
    - M0: max neighbors at layer 0 (typically 2*M)
    - ef_construction: candidate list size during building
    - ef_search: candidate list size during query
    - ml: layer selection multiplier (1/ln(ml) is the decay factor)
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        M0: int | None = None,
        ef_construction: int = 200,
        ef_search: int = 50,
        ml: float = 1.0 / 0.693,  # ~1/ln(2)
        device: torch.device | str | None = None,
        seed: int = 42,
    ):
        super().__init__(dim)
        self.device = get_device(device)
        self.M = M
        self.M0 = M0 or (2 * M)
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml
        self.seed = seed
        self._rng = random.Random(seed)
        
        self._xb = torch.empty(0, dim, dtype=torch.float32, device=self.device)
        self._graphs: List[List[List[int]]] = []  # [layer][node_id][neighbors]
        self._levels: List[int] = []  # level for each node
        self._entry: int | None = None
        self._max_level: int = -1

    def _select_level(self) -> int:
        """Select layer for new point using exponential decay."""
        return int(-1.0 * self._rng.random() * self.ml)

    def _add_impl(self, xb: torch.Tensor) -> None:
        xb = xb.to(self.device)
        start = self._xb.size(0)
        if start == 0:
            self._xb = xb
        else:
            self._xb = torch.cat([self._xb, xb], dim=0)
        # Will link in build()
        for _ in range(xb.size(0)):
            self._levels.append(-1)

    def _build_impl(self) -> None:
        n = self._xb.size(0)
        if n == 0:
            return
        
        # Initialize first point
        if self._entry is None:
            self._entry = 0
            level = self._select_level()
            self._levels[0] = level
            self._max_level = max(self._max_level, level)
            # Initialize graphs up to this level
            for _ in range(level + 1):
                self._graphs.append([[] for _ in range(n)])
        
        # Insert remaining points
        for i in range(1 if self._entry == 0 else 0, n):
            if self._levels[i] >= 0:
                continue  # already inserted
            self._insert_point(i)

    def _insert_point(self, q: int) -> None:
        """Insert point q into the HNSW structure."""
        level = self._select_level()
        self._levels[q] = level
        
        # Extend graphs if needed
        while len(self._graphs) <= level:
            self._graphs.append([[] for _ in range(self._xb.size(0))])
        for lc in range(len(self._graphs)):
            while len(self._graphs[lc]) <= q:
                self._graphs[lc].append([])
        
        # Search for nearest neighbors starting from entry point
        ep = self._entry
        assert ep is not None
        
        # Phase 1: greedy search from top to level+1
        for lc in range(self._max_level, level, -1):
            ep = self._search_layer(q, [ep], 1, lc)[0]
        
        # Phase 2: find neighbors at each level from level down to 0
        for lc in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(q, [ep] if isinstance(ep, int) else ep, self.ef_construction, lc)
            M = self.M0 if lc == 0 else self.M
            
            # Select M best neighbors
            neighbors = self._select_neighbors(q, candidates, M, lc)
            
            # Add bidirectional links
            for j in neighbors:
                self._graphs[lc][q].append(j)
                self._graphs[lc][j].append(q)
                
                # Prune j's neighbors if needed
                max_neighbors = self.M0 if lc == 0 else self.M
                if len(self._graphs[lc][j]) > max_neighbors:
                    self._prune_neighbors(j, max_neighbors, lc)
            
            ep = candidates
        
        # Update entry point if needed
        if level > self._max_level:
            self._max_level = level
            self._entry = q

    def _search_layer(self, q: int, entry_points: List[int], ef: int, layer: int) -> List[int]:
        """
        Search for nearest neighbors at a specific layer.
        Returns list of candidate indices sorted by distance (best first).
        """
        visited: Set[int] = set()
        candidates: List[Tuple[float, int]] = []  # min-heap by negative distance
        w: List[Tuple[float, int]] = []  # max-heap by distance
        
        qv = self._xb[q]
        
        for ep in entry_points:
            if ep >= len(self._graphs[layer]):
                continue
            dist = float(-qv @ self._xb[ep])  # negative cosine
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(w, (-dist, ep))
            visited.add(ep)
        
        while candidates:
            c_dist, c = heapq.heappop(candidates)
            f_dist, _ = w[0]
            
            if c_dist > -f_dist:
                break
            
            for e in self._graphs[layer][c]:
                if e not in visited:
                    visited.add(e)
                    f_dist, _ = w[0]
                    d = float(-qv @ self._xb[e])
                    
                    if d < -f_dist or len(w) < ef:
                        heapq.heappush(candidates, (d, e))
                        heapq.heappush(w, (-d, e))
                        
                        if len(w) > ef:
                            heapq.heappop(w)
        
        return [idx for (_, idx) in sorted(w, reverse=True)]

    def _select_neighbors(self, q: int, candidates: List[int], M: int, layer: int) -> List[int]:
        """Select M best neighbors from candidates using heuristic or simple selection."""
        # Simple selection: take M closest
        qv = self._xb[q]
        dists = [(float(-qv @ self._xb[c]), c) for c in candidates if c < len(self._graphs[layer])]
        dists.sort()
        return [c for (_, c) in dists[:M]]

    def _prune_neighbors(self, idx: int, M: int, layer: int) -> None:
        """Prune neighbors of idx to keep only M best."""
        neighbors = self._graphs[layer][idx]
        if len(neighbors) <= M:
            return
        
        v = self._xb[idx]
        dists = [(float(-v @ self._xb[j]), j) for j in neighbors]
        dists.sort()
        self._graphs[layer][idx] = [j for (_, j) in dists[:M]]

    def _search_impl(self, xq: torch.Tensor, k: int) -> SearchResult:
        assert self._xb.size(0) > 0 and self._entry is not None
        xq = xq.to(self.device)
        
        idx_out = []
        score_out = []
        
        for qi in range(xq.size(0)):
            # Insert query temporarily for search
            q_idx = self._xb.size(0)
            self._xb = torch.cat([self._xb, xq[qi:qi+1]], dim=0)
            
            # Extend graphs
            for lc in range(len(self._graphs)):
                self._graphs[lc].append([])
            
            # Search from top layer down
            ep = self._entry
            for lc in range(self._max_level, 0, -1):
                ep = self._search_layer(q_idx, [ep], 1, lc)[0]
            
            # Search at layer 0 with ef_search
            candidates = self._search_layer(q_idx, [ep], max(self.ef_search, k), 0)
            
            # Remove query from structure
            self._xb = self._xb[:-1]
            for lc in range(len(self._graphs)):
                self._graphs[lc].pop()
            
            # Compute exact scores for candidates
            subset = torch.tensor(candidates[:k*2], device=self.device, dtype=torch.long)
            subset = subset[subset < self._xb.size(0)]
            if len(subset) == 0:
                subset = torch.arange(min(k, self._xb.size(0)), device=self.device, dtype=torch.long)
            
            sims = xq[qi] @ self._xb[subset].T
            kq = min(k, sims.numel())
            scores, order = torch.topk(sims, k=kq, dim=0, largest=True, sorted=True)
            idx = subset[order]
            
            if kq < k:
                pad = k - kq
                idx = torch.cat([idx, torch.full((pad,), -1, dtype=torch.long, device=idx.device)])
                scores = torch.cat([scores, torch.full((pad,), float("nan"), dtype=scores.dtype, device=scores.device)])
            
            idx_out.append(idx)
            score_out.append(scores)
        
        return SearchResult(indices=torch.stack(idx_out).cpu(), scores=torch.stack(score_out).cpu())

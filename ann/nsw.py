from __future__ import annotations
import heapq
from typing import List, Tuple
import torch
from .base import Index, SearchResult
from .utils import get_device


class NSWIndex(Index):
    """
    Single-layer Navigable Small World (NSW) graph index.

    - M: max neighbors per node
    - ef_construction: candidate list size during building
    - ef_search: candidate list size during query
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 100, ef_search: int = 64, device: torch.device | str | None = None):
        super().__init__(dim)
        self.device = get_device(device)
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._xb = torch.empty(0, dim, dtype=torch.float32, device=self.device)
        self._graph: List[List[int]] = []  # adjacency lists
        self._entry: int | None = None

    def _add_impl(self, xb: torch.Tensor) -> None:
        xb = xb.to(self.device)
        start = self._xb.size(0)
        if start == 0:
            self._xb = xb
        else:
            self._xb = torch.cat([self._xb, xb], dim=0)
        # extend graph with isolated nodes; will be linked in build
        self._graph.extend([[] for _ in range(xb.size(0))])

    def _build_impl(self) -> None:
        n = self._xb.size(0)
        if n == 0:
            return
        self._entry = 0
        self._graph = [[] for _ in range(n)]
        for i in range(n):
            if i == 0:
                continue
            self._link_new_point(i)

    def _link_new_point(self, idx: int) -> None:
        # Search neighbors among existing points using ef_construction
        cand = self._ef_search_internal(self._xb[idx:idx+1], self.ef_construction, entry=self._entry)[0]
        # pick top M by similarity
        neighbors = cand[: self.M]
        for j in neighbors:
            # add bidirectional edge
            if j not in self._graph[idx]:
                self._graph[idx].append(j)
            if idx not in self._graph[j]:
                self._graph[j].append(idx)
            # prune if degree exceeds M
            if len(self._graph[j]) > self.M:
                self._prune_neighbors(j)
        if len(self._graph[idx]) > self.M:
            self._prune_neighbors(idx)

    def _prune_neighbors(self, i: int) -> None:
        # Keep M most similar neighbors
        neigh = self._graph[i]
        if not neigh:
            return
        sims = (self._xb[neigh] @ self._xb[i]).tolist()
        order = sorted(range(len(neigh)), key=lambda k: sims[k], reverse=True)
        self._graph[i] = [neigh[k] for k in order[: self.M]]

    def _ef_search_internal(self, xq: torch.Tensor, ef: int, entry: int | None = None) -> List[List[int]]:
        xb = self._xb
        entry = 0 if entry is None else entry
        out: List[List[int]] = []
        for q in xq:
            # candidate max-heap by negative similarity (so largest sim first)
            cand: List[Tuple[float, int]] = []
            # visited set
            visited = set()
            # current best list as min-heap of size ef
            best: List[Tuple[float, int]] = []

            def push_best(score: float, idx: int):
                if len(best) < ef:
                    heapq.heappush(best, (score, idx))
                else:
                    if score > best[0][0]:
                        heapq.heapreplace(best, (score, idx))

            # start from entry
            start = entry
            visited.add(start)
            s0 = float(q @ xb[start])
            heapq.heappush(cand, (-s0, start))
            push_best(s0, start)

            while cand:
                negs, i = heapq.heappop(cand)
                s = -negs
                # if this candidate is worse than worst in best, we can stop early
                if best and s < best[0][0]:
                    break
                for j in self._graph[i]:
                    if j in visited:
                        continue
                    visited.add(j)
                    sj = float(q @ xb[j])
                    push_best(sj, j)
                    heapq.heappush(cand, (-sj, j))
            # return neighbors sorted by score desc
            out.append([idx for (_, idx) in sorted(best, reverse=True)])
        return out

    def _search_impl(self, xq: torch.Tensor, k: int) -> SearchResult:
        assert self._xb.size(0) > 0 and self._entry is not None
        xq = xq.to(self.device)
        cand_lists = self._ef_search_internal(xq, ef=max(self.ef_search, k), entry=self._entry)
        idx_out = []
        score_out = []
        for qi, cand in enumerate(cand_lists):
            # rerank exactly among candidates
            subset = torch.tensor(cand, device=self.device, dtype=torch.long)
            sims = xq[qi] @ self._xb[subset].T
            kq = min(k, sims.numel())
            scores, order = torch.topk(sims, k=kq, dim=0)
            top_idx = subset[order]
            if kq < k:
                pad = k - kq
                top_idx = torch.cat([top_idx, torch.full((pad,), -1, dtype=torch.long, device=top_idx.device)])
                scores = torch.cat([scores, torch.full((pad,), float("nan"), dtype=scores.dtype, device=scores.device)])
            idx_out.append(top_idx)
            score_out.append(scores)
        return SearchResult(indices=torch.stack(idx_out).cpu(), scores=torch.stack(score_out).cpu())

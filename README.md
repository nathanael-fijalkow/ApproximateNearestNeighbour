# Approximate Nearest Neighbour (ANN) algorithms for RAG

This educational repo implements several ANN indexes in readable PyTorch, using cosine similarity:

- **BruteForceIndex**: exact baseline using matrix multiplication
- **LSHIndex**: random hyperplane locality-sensitive hashing for cosine
- **IVFIndex**: IVF-Flat with a coarse k-means quantizer
- **NSWIndex**: single-layer Navigable Small World graph (simplified HNSW)
- **HNSWIndex**: Hierarchical Navigable Small World with multi-layer graph structure
- **PQIndex**: Product Quantization for memory-efficient approximate search
- **IVFPQIndex**: IVF with PQ-compressed vectors for scalable recall/speed tradeoff

All indexes accept and return PyTorch tensors. Internally, vectors are L2-normalized so cosine similarity reduces to a dot product.

## Quick start

- Install dependencies (CPU-only):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Run the quick comparison benchmark:

```bash
python scripts/compare.py
```

By default, the script auto-detects the best available device (MPS → CUDA → CPU). You can override with `--device`:

```bash
python scripts/compare.py --device mps    # Apple Silicon GPU
python scripts/compare.py --device cuda   # NVIDIA GPU
python scripts/compare.py --device cpu    # CPU only
```

## API sketch

All indexes share a minimal interface:

- add(xb): add base vectors (2D tensor, shape [n, d])
- build(): build the structure (k-means, graph, etc.)
- search(xq, k): returns top-k indices and cosine scores

Example:

```python
import torch
from ann import BruteForceIndex, LSHIndex, IVFIndex, NSWIndex, HNSWIndex, PQIndex, IVFPQIndex, get_best_device

xb = torch.randn(10_000, 768)
xb = xb / xb.norm(dim=1, keepdim=True)

xq = torch.randn(100, 768)
xq = xq / xq.norm(dim=1, keepdim=True)

# Auto-detect best device (MPS → CUDA → CPU)
device = get_best_device()
print(f"Using device: {device}")

idx = IVFIndex(768, nlist=100, n_probe=8, device=device)
# Or explicitly: device="mps", device="cuda", device="cpu", or device=None for auto
idx.add(xb)
idx.build()
res = idx.search(xq, k=5)
print(res.indices.shape, res.scores.shape)
```

## Algorithms (intuition)

- **Brute force**: multiply queries by the database matrix and take top-k.
- **LSH (cosine)**: hash vectors by the sign of dot products to random hyperplanes. Look up identical buckets across multiple tables and re-rank candidates exactly.
- **IVF-Flat**: cluster the database into `nlist` centroids (k-means). At query time, probe the `n_probe` closest centroids and re-rank their members.
- **NSW**: connect each point to a small set of neighbors. At query time, walk the graph with a best-first search, then re-rank the visited set.
- **HNSW (Hierarchical NSW)**: multi-layer graph where layer 0 contains all points and higher layers contain exponentially fewer points. Search starts from the top layer and zooms in hierarchically. Generally achieves the best recall/speed tradeoff among graph-based methods. More complex to build than NSW but faster at search time.
- **PQ (Product Quantization)**: split each vector into `m` subspaces, quantize each subspace with a small k-means codebook (2^nbits centroids). At search time, precompute query-to-centroid distances per subspace and approximate full distances by summing subspace distances. Very memory-efficient (e.g., 8 bytes per 768-d vector with m=8, nbits=8).
- **IVF-PQ**: combine IVF coarse quantization with PQ encoding. Probe top lists, then use PQ asymmetric distance to re-rank compressed vectors. Balances memory and speed with tunable recall.

These trade recall for speed and memory differently; the included script gives a side-by-side on synthetic data.

## Scripts

- `scripts/compare.py`: builds each index on synthetic unit vectors and reports recall@k and QPS. Adjust parameters for your scale.
  - Each index has a 60-second timeout to prevent hanging on large datasets
  - Use `--device` to override auto-detection (e.g., `--device cpu` if MPS is slow for certain operations)
  - Includes natural language explanations of each algorithm's characteristics and trade-offs

Example benchmark (n=10k, d=128, CPU):

```text
BruteForce: 102k QPS, 100% recall (baseline)
LSH:        7.4k QPS, 3% recall (poor on random data)
IVF-Flat:   5.9k QPS, 39% recall
NSW:        400 QPS, 72% recall (slow build)
HNSW:       359 QPS, 50% recall (slower build, hierarchical structure)
PQ:         19k QPS, 14% recall (256× compression)
IVF-PQ:     7.1k QPS, 12% recall (compression + filtering)
```

*Note: Recall is modest on random unit vectors. Real embeddings with structure yield much higher recall.*

## Notes

- **Device support**: All indexes auto-detect the best device (MPS for Apple Silicon → CUDA for NVIDIA → CPU fallback). Pass `device=None` (default) for auto-detection, or explicitly set `device="mps"`, `device="cuda"`, or `device="cpu"`.
- **When to use ANN vs Brute Force**: On small datasets (< 50k vectors), brute force is often faster and simpler. ANN methods become essential at scale (100k-1M+ vectors) where exact search becomes prohibitively slow.
- For RAG, cosine similarity is common after embedding normalization; these indexes assume that.
- The implementations are intentionally compact and not production-optimized. For large-scale systems, consider libraries like FAISS, ScaNN, or hnswlib.

## Tuning tips

- **LSH**
  - Increase n_tables, n_bits, and max_candidates to improve recall.
  - Multi-probe LSH (flip a few bits to probe nearby buckets) can boost recall.
- **IVF**
  - Larger nlist increases precision; increase n_probe for recall; both increase compute.
  - For RAG-like dimensions (e.g., 768), try nlist in [sqrt(n), 10*sqrt(n)] and n_probe in [8, 32].
- **NSW**
  - Higher M and ef_search improve recall linearly with memory/search time.
  - ef_construction mostly affects build time and graph quality.
- **HNSW**
  - M controls neighbors per layer (typically 16-48); higher M improves recall but increases memory.
  - ef_construction controls build quality; higher values (100-400) give better graphs but slower builds.
  - ef_search controls search quality; higher values improve recall at the cost of speed.
  - HNSW typically outperforms NSW due to hierarchical structure, especially on large datasets.
- **PQ**
  - More subspaces (m) or bits (nbits) improve recall but increase memory and compute.
  - For 768-d vectors, m=8 with nbits=8 gives 8 bytes per vector (256-to-1 compression).
  - PQ recall depends heavily on data structure; random vectors show lower recall than clustered embeddings.
- **IVF-PQ**
  - Combines IVF's coarse filtering with PQ's compression.
  - Tune nlist and n_probe for coarse recall, then m/nbits for fine-grained tradeoff.
  - Set `encode_residual=True` to encode residuals (x - centroid) for better PQ accuracy, at slight cost.

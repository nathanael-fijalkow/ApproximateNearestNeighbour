import time
import math
import argparse
import signal
import torch
from ann import BruteForceIndex, LSHIndex, IVFIndex, NSWIndex, HNSWIndex, PQIndex, IVFPQIndex, get_best_device


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out after 60 seconds")


def make_data(n=10000, nq=200, d=256, seed=0, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    xb = torch.randn(n, d, generator=g, device=device)
    xb = torch.nn.functional.normalize(xb, dim=1)
    xq = torch.randn(nq, d, generator=g, device=device)
    xq = torch.nn.functional.normalize(xq, dim=1)
    return xb.cpu(), xq.cpu()


def measure(index, xb, xq, k=10, build=True):
    # Set up 60-second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)
    
    try:
        t0 = time.time()
        index.add(xb)
        if build:
            index.build()
        t1 = time.time()
        res = index.search(xq, k=k)
        t2 = time.time()
        signal.alarm(0)  # Cancel the alarm
        return {
            "build_s": t1 - t0,
            "search_s": t2 - t1,
            "indices": res.indices,
            "scores": res.scores,
        }
    except TimeoutError as e:
        signal.alarm(0)  # Cancel the alarm
        print(f"  ⚠️  TIMEOUT: {e}")
        return None


def recall_at_k(pred, truth, k):
    # pred, truth: (nq, k) long
    pred = pred[:, :k]
    truth = truth[:, :k]
    match = 0
    for i in range(pred.size(0)):
        match += len(set(pred[i].tolist()) & set(truth[i].tolist()))
    return match / (pred.size(0) * k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--nq", type=int, default=1000)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--device", type=str, default=None, help="Device: 'cpu', 'cuda', 'mps', or None for auto-detect")
    args = p.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        device_obj = get_best_device()
        device_str = str(device_obj)
        print(f"Auto-detected device: {device_str}")
    else:
        device_obj = torch.device(args.device)
        device_str = args.device

    xb, xq = make_data(args.n, args.nq, args.d, device="cpu")

    print(f"Data: xb={xb.shape}, xq={xq.shape}, device={device_str}")
    print(f"\nComparing {len([BruteForceIndex, LSHIndex, IVFIndex, NSWIndex, HNSWIndex, PQIndex, IVFPQIndex])} indexes on {args.nq} queries over {args.n} vectors (dim={args.d})")
    print(f"Metrics: build time (s), search time (s), QPS (queries/sec), recall@{args.k} (fraction of true top-{args.k} found)")
    print(f"{'='*100}\n")

    bf = BruteForceIndex(args.d, device=device_obj)
    r_bf = measure(bf, xb, xq, k=args.k, build=False)
    if r_bf is None:
        print("BruteForce: FAILED (timeout or error)")
        return
    truth = r_bf["indices"]
    print(f"BruteForce: build={r_bf['build_s']:.3f}s, search={r_bf['search_s']:.3f}s, qps={args.nq/r_bf['search_s']:.1f}")
    print("  → Exact search baseline (100% recall). All approximate methods compared against this.\n")

    lsh = LSHIndex(args.d, n_tables=16, n_bits=16, max_candidates=8192, device=device_obj)
    r_lsh = measure(lsh, xb, xq, k=args.k)
    if r_lsh is not None:
        rec = recall_at_k(r_lsh["indices"], truth, args.k)
        print(f"LSH:        build={r_lsh['build_s']:.3f}s, search={r_lsh['search_s']:.3f}s, qps={args.nq/r_lsh['search_s']:.1f}, recall@{args.k}={rec:.3f}")
        print("  → Locality-sensitive hashing. Fast but low recall on random data; better on structured embeddings.\n")
    else:
        print("LSH:        FAILED (timeout or error)\n")

    ivf = IVFIndex(args.d, nlist=max(64, int(math.sqrt(args.n))), n_probe=12, device=device_obj)
    r_ivf = measure(ivf, xb, xq, k=args.k)
    if r_ivf is not None:
        rec = recall_at_k(r_ivf["indices"], truth, args.k)
        print(f"IVF-Flat:   build={r_ivf['build_s']:.3f}s, search={r_ivf['search_s']:.3f}s, qps={args.nq/r_ivf['search_s']:.1f}, recall@{args.k}={rec:.3f}")
        print("  → Inverted file with k-means clustering. Good recall/speed balance; scales well.\n")
    else:
        print("IVF-Flat:   FAILED (timeout or error)\n")

    nsw = NSWIndex(args.d, M=24, ef_construction=120, ef_search=96, device=device_obj)
    r_nsw = measure(nsw, xb, xq, k=args.k)
    if r_nsw is not None:
        rec = recall_at_k(r_nsw["indices"], truth, args.k)
        print(f"NSW:        build={r_nsw['build_s']:.3f}s, search={r_nsw['search_s']:.3f}s, qps={args.nq/r_nsw['search_s']:.1f}, recall@{args.k}={rec:.3f}")
        print("  → Single-layer graph search. High recall but slow build (O(n²)). Simpler than HNSW.\n")
    else:
        print("NSW:        FAILED (timeout or error)\n")

    hnsw = HNSWIndex(args.d, M=16, ef_construction=200, ef_search=50, device=device_obj)
    r_hnsw = measure(hnsw, xb, xq, k=args.k)
    if r_hnsw is not None:
        rec = recall_at_k(r_hnsw["indices"], truth, args.k)
        print(f"HNSW:       build={r_hnsw['build_s']:.3f}s, search={r_hnsw['search_s']:.3f}s, qps={args.nq/r_hnsw['search_s']:.1f}, recall@{args.k}={rec:.3f}")
        print("  → Hierarchical graph with multiple layers. Best graph-based method for large scale; slower build.\n")
    else:
        print("HNSW:       FAILED (timeout or error)\n")

    pq = PQIndex(args.d, m=8, nbits=8, device=device_obj)
    r_pq = measure(pq, xb, xq, k=args.k)
    if r_pq is not None:
        rec = recall_at_k(r_pq["indices"], truth, args.k)
        print(f"PQ:         build={r_pq['build_s']:.3f}s, search={r_pq['search_s']:.3f}s, qps={args.nq/r_pq['search_s']:.1f}, recall@{args.k}={rec:.3f}")
        print("  → Product Quantization. Extreme compression (256×) for memory efficiency; moderate recall.\n")
    else:
        print("PQ:         FAILED (timeout or error)\n")

    ivfpq = IVFPQIndex(args.d, nlist=max(64, int(math.sqrt(args.n))), n_probe=12, m=8, nbits=8, encode_residual=False, device=device_obj)
    r_ivfpq = measure(ivfpq, xb, xq, k=args.k)
    if r_ivfpq is not None:
        rec = recall_at_k(r_ivfpq["indices"], truth, args.k)
        print(f"IVF-PQ:     build={r_ivfpq['build_s']:.3f}s, search={r_ivfpq['search_s']:.3f}s, qps={args.nq/r_ivfpq['search_s']:.1f}, recall@{args.k}={rec:.3f}")
        print("  → IVF + PQ compression. Combines coarse filtering with memory savings; best for billion-scale.\n")
    else:
        print("IVF-PQ:     FAILED (timeout or error)\n")
    
    print(f"{'='*100}")
    print("\nSummary: Higher recall = more accurate results. Higher QPS = faster queries.")
    print("Trade-offs: Graph methods (NSW/HNSW) offer high recall but slow builds.")
    print("            Compression (PQ/IVF-PQ) trades accuracy for memory and speed.")
    print("            IVF-Flat balances all three dimensions reasonably well.")
    print(f"\nIMPORTANT: On small datasets (n < ~50k), BruteForce is often the best choice!")
    print(f"           It's fastest, most accurate, and has zero build time.")
    print(f"           ANN methods shine when n > 100k-1M+ where brute force becomes too slow.")


if __name__ == "__main__":
    main()

import torch
from ann import BruteForceIndex, LSHIndex, IVFIndex, NSWIndex, HNSWIndex, PQIndex, IVFPQIndex


def make_unit(n=1000, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return torch.nn.functional.normalize(x, dim=1)


def test_bruteforce_exact_top1():
    xb = make_unit(1000, 32)
    xq = xb[:10]  # queries are in the database
    bf = BruteForceIndex(32)
    bf.add(xb)
    res = bf.search(xq, k=1)
    assert torch.equal(res.indices.squeeze(1), torch.arange(10))


def test_lsh_reasonable_recall():
    xb = make_unit(5000, 64, seed=1)
    xq = make_unit(100, 64, seed=2)
    bf = BruteForceIndex(64)
    bf.add(xb)
    truth = bf.search(xq, k=10).indices

    lsh = LSHIndex(64, n_tables=12, n_bits=12, max_candidates=4096)
    lsh.add(xb)
    lsh.build()
    pred = lsh.search(xq, k=10).indices
    # should get some reasonable recall on random data
    inter = 0
    for i in range(xq.size(0)):
        inter += len(set(pred[i].tolist()) & set(truth[i].tolist()))
    recall = inter / (xq.size(0) * 10)
    # Simple LSH on random data needs many tables/bits to achieve high recall;
    # with modest settings we just check it's non-trivially above random.
    assert recall > 0.03


def test_ivf_reasonable_recall():
    xb = make_unit(8000, 64, seed=3)
    xq = make_unit(100, 64, seed=4)
    bf = BruteForceIndex(64)
    bf.add(xb)
    truth = bf.search(xq, k=10).indices

    ivf = IVFIndex(64, nlist=64, n_probe=8)
    ivf.add(xb)
    ivf.build()
    pred = ivf.search(xq, k=10).indices
    inter = 0
    for i in range(xq.size(0)):
        inter += len(set(pred[i].tolist()) & set(truth[i].tolist()))
    recall = inter / (xq.size(0) * 10)
    assert recall > 0.4


def test_nsw_reasonable_recall():
    xb = make_unit(6000, 64, seed=5)
    xq = make_unit(100, 64, seed=6)
    bf = BruteForceIndex(64)
    bf.add(xb)
    truth = bf.search(xq, k=10).indices

    nsw = NSWIndex(64, M=16, ef_construction=80, ef_search=64)
    nsw.add(xb)
    nsw.build()
    pred = nsw.search(xq, k=10).indices
    inter = 0
    for i in range(xq.size(0)):
        inter += len(set(pred[i].tolist()) & set(truth[i].tolist()))
    recall = inter / (xq.size(0) * 10)
    assert recall > 0.5


def test_hnsw_reasonable_recall():
    xb = make_unit(5000, 64, seed=11)
    xq = make_unit(100, 64, seed=12)
    bf = BruteForceIndex(64)
    bf.add(xb)
    truth = bf.search(xq, k=10).indices

    hnsw = HNSWIndex(64, M=16, ef_construction=100, ef_search=50)
    hnsw.add(xb)
    hnsw.build()
    pred = hnsw.search(xq, k=10).indices
    inter = 0
    for i in range(xq.size(0)):
        inter += len(set(pred[i].tolist()) & set(truth[i].tolist()))
    recall = inter / (xq.size(0) * 10)
    assert recall > 0.6


def test_pq_reasonable_recall():
    xb = make_unit(5000, 64, seed=7)
    xq = make_unit(100, 64, seed=8)
    bf = BruteForceIndex(64)
    bf.add(xb)
    truth = bf.search(xq, k=10).indices

    pq = PQIndex(64, m=8, nbits=8)
    pq.add(xb)
    pq.build()
    pred = pq.search(xq, k=10).indices
    inter = 0
    for i in range(xq.size(0)):
        inter += len(set(pred[i].tolist()) & set(truth[i].tolist()))
    recall = inter / (xq.size(0) * 10)
    # PQ loses precision, so recall may be modest
    assert recall > 0.1


def test_ivfpq_reasonable_recall():
    xb = make_unit(8000, 64, seed=9)
    xq = make_unit(100, 64, seed=10)
    bf = BruteForceIndex(64)
    bf.add(xb)
    truth = bf.search(xq, k=10).indices

    ivfpq = IVFPQIndex(64, nlist=64, n_probe=8, m=8, nbits=8, encode_residual=False)
    ivfpq.add(xb)
    ivfpq.build()
    pred = ivfpq.search(xq, k=10).indices
    inter = 0
    for i in range(xq.size(0)):
        inter += len(set(pred[i].tolist()) & set(truth[i].tolist()))
    recall = inter / (xq.size(0) * 10)
    assert recall > 0.15

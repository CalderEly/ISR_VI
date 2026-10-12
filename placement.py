"""
Forward ring search for the Tiered Synapses placement policy (Phase 2).

Finds every neuron within R forward hops of a set of source neurons and
returns each one's hop distance, plus the neurons grouped by distance
("buckets"), already in nearest-first order. No sorting is needed.

Two implementations with identical results:

  RingSearcher.search  - frontier-only BFS on CSR arrays (NumPy).
                         Cost grows with the size of the ring, not the
                         network. Best when rings are small.
  ring_torch           - one sparse matrix-vector product per hop (PyTorch).
                         Cost grows with the whole network every hop.
                         Best when rings cover a large part of the network.

Measure both on the real network before choosing.

Matches the plan:
  - Forward hops only (backward rings were dropped Oct 6).
  - Multi-source: every charged / external-input neuron starts at distance 0.
  - Residency fallback: a reached neuron whose record is not in RAM is given
    a distance but not expanded, and is returned for promotion to RAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UNREACHED = -1

# Tier codes, matching what Phase 1 `locate` / view.tier_of report.
GPU, RAM, SSD = 0, 1, 2


# ---------------------------------------------------------------------------
# Building the adjacency structure
# ---------------------------------------------------------------------------

def build_csr(senders, targets, N):
    """
    Build CSR (compressed sparse row) adjacency from a list of synapses.

    senders[i] -> targets[i] is one synapse. Only structure is stored, never
    weights, so inhibitory (negative) synapses cannot cancel out.

    Returns
      row_ptr : int64 array, length N + 1
      col_idx : int32 array, length = number of synapses
    Neuron n's targets are col_idx[row_ptr[n] : row_ptr[n + 1]].
    """
    senders = np.asarray(senders, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if senders.shape != targets.shape:
        raise ValueError("senders and targets must have the same length")
    if senders.size and (senders.min() < 0 or senders.max() >= N
                         or targets.min() < 0 or targets.max() >= N):
        raise ValueError("neuron ID out of range [0, N)")

    order = np.argsort(senders, kind="stable")      # group synapses by sender
    col_idx = targets[order].astype(np.int32)
    counts = np.bincount(senders, minlength=N)       # out-degree of each neuron
    row_ptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    return row_ptr, col_idx


# ---------------------------------------------------------------------------
# Frontier-only BFS (NumPy)
# ---------------------------------------------------------------------------

@dataclass
class RingResult:
    dist: np.ndarray                 # hop distance per neuron, -1 = outside ring
    buckets: list                    # buckets[h] = IDs at exactly h hops
    needs_ram: np.ndarray = field(   # reached but not resident: promote to RAM
        default_factory=lambda: np.empty(0, dtype=np.int64))

    def in_ring(self):
        """All IDs in the ring, nearest first (ready for 'drop the farthest')."""
        return np.concatenate(self.buckets) if self.buckets else np.empty(0, np.int64)


class RingSearcher:
    """
    Reusable searcher. Keeps a stamp array so starting a new search costs one
    increment instead of clearing N entries.

    stamp[n] == self._epoch  means neuron n was visited in the current search.
    """

    def __init__(self, row_ptr, col_idx):
        self.row_ptr = np.asarray(row_ptr, dtype=np.int64)
        self.col_idx = np.asarray(col_idx, dtype=np.int32)
        self.N = self.row_ptr.size - 1
        self._stamp = np.zeros(self.N, dtype=np.uint32)
        self._epoch = 0

    def _new_epoch(self):
        self._epoch += 1
        if self._epoch == np.iinfo(np.uint32).max:  # wraparound, ~4 billion searches
            self._stamp[:] = 0
            self._epoch = 1
        return self._epoch

    def _neighbors_of(self, frontier):
        """All targets of every neuron in `frontier`, in one vectorized gather."""
        starts = self.row_ptr[frontier]
        lengths = self.row_ptr[frontier + 1] - starts
        total = int(lengths.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        # Index of each synapse in col_idx: for each frontier neuron, the run
        # starts, starts+1, ..., starts+length-1, all concatenated.
        run_start = np.cumsum(lengths) - lengths
        idx = np.repeat(starts - run_start, lengths) + np.arange(total)
        return self.col_idx[idx].astype(np.int64)

    def search(self, sources, R, resident=None):
        """
        sources  : IDs at distance 0 (charged neurons + external inputs)
        R        : radius in forward hops (3 for the GPU ring)
        resident : optional integer array, one entry per neuron, giving the
                   tier its record is in: 0 = GPU, 1 = RAM, 2 = SSD.
                   Neurons on SSD are reached but not expanded, and are
                   returned in `needs_ram`.
        """
        if R < 0:
            raise ValueError("R must be >= 0")
        loaded = None                       # True where the record can be read
        if resident is not None:
            resident = np.asarray(resident)
            if resident.shape != (self.N,):
                raise ValueError(f"resident must have shape ({self.N},)")
            if not np.issubdtype(resident.dtype, np.integer):
                raise TypeError("resident must be integers: 0=GPU, 1=RAM, 2=SSD")
            if resident.size and (resident.min() < GPU or resident.max() > SSD):
                raise ValueError("resident values must be 0 (GPU), 1 (RAM) or 2 (SSD)")
            loaded = resident != SSD
        epoch = self._new_epoch()
        stamp = self._stamp
        dist = np.full(self.N, UNREACHED, dtype=np.int32)

        frontier = np.unique(np.asarray(sources, dtype=np.int64))
        if frontier.size and (frontier.min() < 0 or frontier.max() >= self.N):
            raise ValueError("source ID out of range [0, N)")
        stamp[frontier] = epoch
        dist[frontier] = 0
        buckets = [frontier]
        needs_ram = []

        for h in range(1, R + 1):
            # Don't expand neurons whose record isn't loaded (plan's fallback).
            if loaded is not None:
                missing = frontier[~loaded[frontier]]
                if missing.size:
                    needs_ram.append(missing)
                frontier = frontier[loaded[frontier]]

            nbrs = self._neighbors_of(frontier)
            nbrs = nbrs[stamp[nbrs] != epoch]   # drop anything already visited
            nbrs = np.unique(nbrs)              # many paths can reach one neuron
            if nbrs.size == 0:                  # ring stopped growing: early exit
                break
            stamp[nbrs] = epoch
            dist[nbrs] = h
            buckets.append(nbrs)
            frontier = nbrs

        # Neurons at the last level are never expanded, but still need RAM
        # if they're about to be promoted, so report them too.
        if loaded is not None and frontier.size:
            missing = frontier[~loaded[frontier]]
            if missing.size:
                needs_ram.append(missing)

        needs = np.unique(np.concatenate(needs_ram)) if needs_ram else np.empty(0, np.int64)
        return RingResult(dist=dist, buckets=buckets, needs_ram=needs)


# ---------------------------------------------------------------------------
# Sparse matrix version (PyTorch)
# ---------------------------------------------------------------------------

def build_A_T_torch(row_ptr, col_idx, device="cpu"):
    """
    Build the transposed adjacency A^T as a binary torch sparse CSR matrix.
    A[i, j] = 1 if i synapses onto j, so A^T[j, i] = 1. Build it once.
    """
    import torch
    import scipy.sparse as sp

    N = len(row_ptr) - 1
    A = sp.csr_matrix(
        (np.ones(len(col_idx), dtype=np.float32), col_idx, row_ptr), shape=(N, N))
    A.sum_duplicates()
    A.data[:] = 1.0                  # duplicate synapses still count once
    AT = A.T.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(AT.indptr.astype(np.int64)),
        torch.from_numpy(AT.indices.astype(np.int64)),
        torch.from_numpy(AT.data),
        size=(N, N), device=device)


def ring_torch(A_T, sources, R):
    """
    Same result as RingSearcher.search(...).dist, using one sparse
    matrix-vector product per hop: f_next = (A^T f > 0) and not visited.
    """
    import torch

    N = A_T.shape[0]
    dev = A_T.device
    dist = torch.full((N,), UNREACHED, dtype=torch.int32, device=dev)
    src = torch.as_tensor(sources, dtype=torch.long, device=dev)
    f = torch.zeros(N, dtype=A_T.dtype, device=dev)
    f[src] = 1
    dist[src] = 0
    for h in range(1, R + 1):
        reached = (A_T @ f) > 0          # counts of frontier senders per neuron
        new = reached & (dist < 0)
        if not bool(new.any()):          # one GPU->CPU sync per hop; fine at R=3
            break
        dist[new] = h
        f = new.to(A_T.dtype)
    return dist


# ---------------------------------------------------------------------------
# Tests: python ring_search.py
# ---------------------------------------------------------------------------

def _reference_bfs(row_ptr, col_idx, sources, R):
    """Plain queue BFS, used only to check the fast versions."""
    from collections import deque
    N = len(row_ptr) - 1
    dist = np.full(N, UNREACHED, dtype=np.int32)
    q = deque()
    for s in set(int(x) for x in sources):
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        if dist[u] == R:
            continue
        for w in col_idx[row_ptr[u]:row_ptr[u + 1]]:
            if dist[w] < 0:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist


def _tests():
    # 1. The 5-neuron example from the explanation: 0->1, 0->2, 1->3, 2->3, 3->4
    rp, ci = build_csr([0, 0, 1, 2, 3], [1, 2, 3, 3, 4], 5)
    r = RingSearcher(rp, ci).search([0], R=2)
    assert r.dist.tolist() == [0, 1, 1, 2, -1], r.dist

    # 2. Chain 0->1->...->20, source 0, R=3: ring is {0..3}  (spec unit test)
    N = 21
    rp, ci = build_csr(np.arange(20), np.arange(1, 21), N)
    s = RingSearcher(rp, ci)
    r = s.search([0], R=3)
    assert r.in_ring().tolist() == [0, 1, 2, 3]
    r9 = s.search([0], R=9)                     # reuse: stamp array must reset
    assert set(r9.in_ring().tolist()) == set(range(10))

    # 3. Backward edges are never followed: 1->0 does not put 1 in 0's ring
    rp, ci = build_csr([1], [0], 2)
    assert RingSearcher(rp, ci).search([0], R=3).dist.tolist() == [0, -1]

    # 4. Cycle doesn't loop forever and early exit works
    rp, ci = build_csr([0, 1, 2], [1, 2, 0], 3)
    r = RingSearcher(rp, ci).search([0], R=100)
    assert r.dist.tolist() == [0, 1, 2] and len(r.buckets) == 3

    # 5. Residency fallback: chain 0->1->2->3, neuron 1 on SSD
    rp, ci = build_csr([0, 1, 2], [1, 2, 3], 4)
    s = RingSearcher(rp, ci)
    r = s.search([0], R=3, resident=np.array([GPU, SSD, RAM, RAM]))
    assert r.dist.tolist() == [0, 1, -1, -1]
    assert r.needs_ram.tolist() == [1]
    # GPU and RAM records are both expandable
    r = s.search([0], R=3, resident=np.array([GPU, RAM, GPU, SSD]))
    assert r.dist.tolist() == [0, 1, 2, 3] and r.needs_ram.tolist() == [3]
    # Bad inputs are rejected
    for bad in (np.array([0, 1, 3, 0]),                 # 3 isn't a tier
                np.array([True, False, True, True]),    # old bool format
                np.array([0, 1, 2])):                   # wrong length
        try:
            s.search([0], R=3, resident=bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"accepted bad resident {bad}")

    # 6. Random networks: NumPy version matches the plain reference BFS
    rng = np.random.default_rng(0)
    for trial in range(20):
        N, k = 2000, int(rng.integers(1, 12))
        snd = np.repeat(np.arange(N), k)
        tgt = rng.integers(0, N, size=snd.size)
        rp, ci = build_csr(snd, tgt, N)
        src = rng.choice(N, size=int(rng.integers(1, 20)), replace=False)
        R = int(rng.integers(0, 6))
        got = RingSearcher(rp, ci).search(src, R).dist
        assert np.array_equal(got, _reference_bfs(rp, ci, src, R)), trial

    # 7. PyTorch version matches too (skipped if torch isn't installed)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not installed: skipped ring_torch test")
    else:
        rp, ci = build_csr(snd, tgt, N)
        AT = build_A_T_torch(rp, ci)
        got = ring_torch(AT, src, 3).numpy()
        assert np.array_equal(got, _reference_bfs(rp, ci, src, 3))

    print("all tests passed")


def _ring_size_report(N=1_000_000, k=100, n_sources=10, R=9, seed=0):
    """Average ring size per hop on a random network (the plan's open item)."""
    import time
    rng = np.random.default_rng(seed)
    snd = np.repeat(np.arange(N), k)
    tgt = rng.integers(0, N, size=snd.size)
    rp, ci = build_csr(snd, tgt, N)
    s = RingSearcher(rp, ci)
    t = time.perf_counter()
    r = s.search(rng.choice(N, n_sources, replace=False), R)
    dt = time.perf_counter() - t
    cum = np.cumsum([b.size for b in r.buckets])
    print(f"N={N:,} k={k} sources={n_sources}  search took {dt*1000:.0f} ms")
    for h, c in enumerate(cum):
        print(f"  within {h} hops: {c:>10,}  ({100*c/N:.2f}% of network)")


if __name__ == "__main__":
    _tests()
    _ring_size_report()